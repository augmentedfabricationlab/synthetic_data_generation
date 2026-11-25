"""
grade_brick_stock.py
=====================

Bridges Phase 1 of the synthetic pipeline (generate_brick_stock.py's
manifest.json + intact_brick.obj + brick_NNNNN.obj files) to allocate.py's
input format: an `assembly_information_model` Assembly of graded
`CellularizedPart` bricks -- i.e. the "material stock assembly model with
cellularized parts" step.

Unlike the live/rendered pipeline (dataset_from_brick_stock.py ->
BrickDamageGrader.to_canonical()), there's no camera/depth image here to
backproject: each stock brick's OBJ mesh already IS the full damaged
geometry, sitting in the exact same untouched local frame as
intact_brick.obj (both built by brick_common.create_single_brick with
identical size, centered at the origin, and round-tripped through
export_mesh_as_obj/import_mesh_from_obj's axis-neutral forward_axis='Y'/
up_axis='Z' pairing -- see that function's docstring). So instead of
to_canonical()'s backprojection step, this densely samples each damaged
mesh's own surface directly in that shared local frame and grades those
points against the same intact CAD reference used everywhere else in
BrickDamageGrader.grade().

This is actually a *cleaner* signal than a single real camera view would
give: sample_surface() sees the whole brick at once -- including a chip's
concave inner faces from every side -- rather than whatever one camera pose
happens to leave unoccluded. That's the right thing for a "ground truth"
stock meant to validate the allocation algorithm (per the project summary's
Section 4), as opposed to a sensor-realistic training/eval set.

Pure Python + trimesh + numpy + compas -- no Blender/bpy/blenderproc needed,
so this can run in a plain venv, independent of generate_brick_stock.py's
Blender-only environment.

Usage:
conda activate gen_env

python /home/afabws/synthetic_data_generation/code/allocation_rendering_pipeline/grade_brick_stock.py \
  --stock-dir /home/afabws/synthetic_data_generation/data/brick_stock \
  --out /home/afabws/brick_allocation_project/assembly_information_model/data/allocation_data/material_stock_simulated_surface.json \
  --workers 8

Default --cell-type surface ray-grades the 40 face patches. For the original
16 volume cells add: --cell-type volume --num-samples 30000 --min-pts 50
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from damage_grader import BrickDamageGrader  # noqa: E402

from assembly_information_model import Assembly  # noqa: E402
from masonry_allocation import CellularizedPart  # noqa: E402
import compas  # noqa: E402


def patch_to_bin(patch_id, grid):
    """Grader face-patch id ("x-_0_1") -> CellularizedPart surface bin ("001-x-").

    A face's two bins run along its other two axes in ascending order, both
    indexed from the box's minimum corner, on both sides. Same mapping as
    brick_perception's ros2 brick_to_assembly_recorder.patch_to_bin, where
    it was checked against the grader's patch centres; copied because that
    module needs rclpy.
    """
    face, u, v = patch_id.split("_")
    u, v = int(u), int(v)
    nx, ny, nz = grid
    hi = face[1] == "+"
    if face[0] == "x":
        ijk = (nx - 1 if hi else 0, u, v)
    elif face[0] == "y":
        ijk = (u, ny - 1 if hi else 0, v)
    else:
        ijk = (u, v, nz - 1 if hi else 0)
    return "{}-{}".format("".join(str(c) for c in ijk), face)


def _grade_one_surface(args_tuple):
    """Worker body for --cell-type surface: ray-grades one brick's 40 face
    patches with BrickDamageGrader.grade_faces_by_raycast -- the method
    behind the scanned bricks' ground truth (grade_all_scanned_bricks.py),
    so synthetic and real GT are on one scale.

    blind_faces=() because, unlike a scan's open underside, these meshes
    are closed: every face is real geometry and gets graded.
    """
    intact_mesh_path, stock_dir, brick_entry = args_tuple
    grader = BrickDamageGrader(intact_mesh_path)

    mesh = trimesh.load(os.path.join(stock_dir, brick_entry["file"]), force="mesh")
    # Damaged OBJs export as disconnected triangle soup until welded.
    mesh.merge_vertices()
    scores, detail = grader.grade_faces_by_raycast(mesh, blind_faces=(), return_detail=True)
    scores_list = [None if np.isnan(s) else float(s) for s in scores]
    no_hit = [None if np.isnan(x) else float(x) for x in detail["no_hit_frac"]]
    return scores_list, no_hit, grader.damage_face_ids(), list(grader.axis_bins)


def _grade_one(args_tuple):
    """Worker body for multiprocessing: samples + grades one stock brick.

    Re-creates the grader inside the worker rather than pickling it across
    the process pool -- trimesh objects (with their embedded proximity
    query structures) are expensive/unreliable to pickle, and a fresh
    BrickDamageGrader() build is cheap relative to sample_surface() +
    signed_distance() on tens of thousands of points.
    """
    intact_mesh_path, stock_dir, brick_entry, num_samples, min_pts, surface_tol, inside_tol = args_tuple
    grader = BrickDamageGrader(intact_mesh_path)

    mesh_path = os.path.join(stock_dir, brick_entry["file"])
    mesh = trimesh.load(mesh_path, force="mesh")
    pts, _face_idx = trimesh.sample.sample_surface(mesh, count=num_samples)

    scores, counts = grader.grade(
        pts, surface_tol=surface_tol, inside_tol=inside_tol, min_pts=min_pts, return_counts=True
    )
    # None (not NaN) for ungraded cells, matching build_cell_grid's own
    # "damage_score": None default and dataset_from_brick_stock.py's own
    # NaN->None convention when it patches coco_annotations.json.
    scores_list = [None if np.isnan(s) else float(s) for s in scores]
    counts_list = [int(c) for c in counts]
    return scores_list, counts_list, list(grader.axis_bins)


def main():
    parser = argparse.ArgumentParser(
        description="Grade a generate_brick_stock.py OBJ stock into a graded "
        "CellularizedPart Assembly, ready for allocate.allocate()."
    )
    parser.add_argument("--stock-dir", required=True,
                         help="Directory with manifest.json, intact_brick.obj, brick_NNNNN.obj.")
    parser.add_argument("--out", required=True, help="Output JSON path for the graded stock Assembly.")
    parser.add_argument("--cell-type", choices=("surface", "volume"), default="surface",
                         help="surface (default): 40 face patches, ray-graded like the scanned "
                              "bricks' ground truth. volume: the original 16 cells from sampled "
                              "surface points (--num-samples/--min-pts/--*-tol apply only here).")
    parser.add_argument("--num-samples", type=int, default=30000,
                         help="Points sampled from each damaged mesh's surface before grading. "
                              "Higher = more stable per-cell scores, especially on the smaller "
                              "of the 16 (2x4 along the long axis, 2x2 elsewhere) cells.")
    parser.add_argument("--min-pts", type=int, default=50,
                         help="Cells with fewer sampled points than this grade as ungraded "
                              "(None), same semantics as BrickDamageGrader.grade's min_pts -- "
                              "raised above its live-pipeline default of 15 since a full-mesh "
                              "sample has far more points to spend than one camera view does.")
    parser.add_argument("--surface-tol", type=float, default=0.003)
    parser.add_argument("--inside-tol", type=float, default=0.002)
    parser.add_argument("--workers", type=int, default=1,
                         help="Multiprocessing pool size. Each brick is graded independently, "
                              "so this scales close to linearly -- worth using once you're "
                              "grading the full 1,000 (or 10,000) brick stock.")
    args = parser.parse_args()

    manifest_path = os.path.join(args.stock_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    brick_size = tuple(manifest["brick_size"])
    intact_mesh_path = os.path.join(args.stock_dir, "intact_brick.obj")

    surface = args.cell_type == "surface"
    if surface:
        worker = _grade_one_surface
        task_args = [(intact_mesh_path, args.stock_dir, entry) for entry in manifest["bricks"]]
    else:
        worker = _grade_one
        task_args = [
            (intact_mesh_path, args.stock_dir, entry, args.num_samples, args.min_pts,
             args.surface_tol, args.inside_tol)
            for entry in manifest["bricks"]
        ]

    if args.workers > 1:
        import multiprocessing
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(worker, task_args)
    else:
        results = [worker(t) for t in task_args]

    stock = Assembly(name="graded_stock")
    stock.attributes["brick_size"] = brick_size
    if surface:
        stock.attributes["grading"] = {
            "method": "raycast",
            "deadband_m": BrickDamageGrader.RAY_DEADBAND_M,
            "saturate_m": BrickDamageGrader.RAY_SATURATE_M,
            "standoff_m": BrickDamageGrader.RAY_STANDOFF_M,
            "rays_per_patch": BrickDamageGrader.RAY_GRID ** 2,
        }

    damage_type_counts = {}
    mean_damages = []

    for i, (entry, result) in enumerate(zip(manifest["bricks"], results)):
        if surface:
            scores_list, no_hit, patch_ids, axis_bins = result
            grid = tuple(axis_bins)
            part = CellularizedPart.from_box(
                brick_size, grid=grid, box_mode=True, name="stock_{:05d}".format(i),
                cell_type="surface",
            )
            part.set_damage_scores({patch_to_bin(pid, grid): s for pid, s in zip(patch_ids, scores_list)})
            net = part.cell_network
            for pid, nh in zip(patch_ids, no_hit):
                net.cell_attribute(net.bin_to_ckey[patch_to_bin(pid, grid)], "no_hit_frac", nh)
        else:
            scores_list, counts_list, axis_bins = result
            part = CellularizedPart.from_box(
                brick_size, grid=tuple(axis_bins), box_mode=True, name="stock_{:05d}".format(i)
            )
            # Flat-list order matches by construction: PartCellNetwork.build_cell_grid
            # iterates itertools.product(range(nx), range(ny), range(nz)) with
            # grid=axis_bins, and BrickDamageGrader.damage_cell_bins is built with
            # the exact same nested-loop order over the same axis_bins -- see both
            # files' comments on this. Cross-checked here rather than assumed.
            part.set_damage_scores(scores_list)
            part.attributes["cell_counts"] = counts_list
        part.attributes["source_obj"] = entry["file"]
        part.attributes["damage_type"] = entry["damage_type"]
        part.attributes["source_index"] = i

        stock.add_part(part, key=i)  # key=i kept == manifest index; render_wall.py relies on this

        damage_type_counts[entry["damage_type"]] = damage_type_counts.get(entry["damage_type"], 0) + 1
        md = part.mean_damage()
        if md is not None:
            mean_damages.append(md)

    compas.json_dump(stock, args.out)

    print("Graded {} bricks -> {}".format(len(manifest["bricks"]), args.out))
    print("Damage type breakdown: {}".format(damage_type_counts))
    if mean_damages:
        print("mean_damage() across stock: min={:.3f} mean={:.3f} max={:.3f}".format(
            min(mean_damages), sum(mean_damages) / len(mean_damages), max(mean_damages)
        ))


if __name__ == "__main__":
    main()