import blenderproc as bproc

# Phase 1 of the two-phase synthetic brick pipeline: generate a stock of
# damaged brick geometries and save each as a plain OBJ file, with no
# rendering. Materials are NOT baked in here -- they're cheap to generate
# (no booleans involved), so dataset_from_brick_stock.py (Phase 2) rolls a
# fresh random material every time a stock brick gets rendered, rather
# than freezing one choice into the stock.
#
# Usage:
#     blenderproc run generate_brick_stock.py resources --num-bricks 1 --custom-blender-path ~/blender/blender-3.6.12-linux-x64
#
# generate_broken_bricks_v3.py is left untouched -- this is a separate
# entry point built on the shared geometry/damage functions in
# brick_common.py. bproc is imported (per blenderproc's own requirement
# that it be the first import) but never called -- no rendering happens
# in this script, so none of blenderproc's init/renderer/camera setup is
# needed, just Blender's own bpy running inside blenderproc's Python env.

import argparse
import json
import os
import random
import sys
import time

import bpy

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from brick_common import (
    DamageAborted,
    add_surface_irregularity,
    apply_random_damage,
    clean_boolean_debris,
    create_single_brick,
    export_mesh_as_obj,
    get_volume,
    soften_edges,
    suppress_stdout,
)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

parser = argparse.ArgumentParser()
parser.add_argument("resources_dir", nargs="?", default=os.path.join(base_dir, "resources"),
                     help="Path to the resources directory (unused right now -- kept for "
                          "consistency with the other generate scripts' positional arg).")
parser.add_argument("--num-bricks", type=int, default=1000,
                     help="Number of damaged brick geometries to generate.")
parser.add_argument("--stock-dir", default=os.path.join(base_dir, "data", "brick_stock"),
                     help="Output directory for the OBJ stock + manifest.json.")
parser.add_argument("--brick-size", type=float, nargs=3, default=[0.25, 0.12, 0.065],
                     metavar=("LENGTH", "WIDTH", "HEIGHT"),
                     help="Brick dimensions in meters. Default is Reichsformat (250x120x65mm). "
                          "Fixed across the whole stock -- Phase 2 grades every stock brick "
                          "against one shared intact reference mesh built from this size.")
parser.add_argument("--fracture-weights", type=float, nargs=5, default=[0.5, 0.1, 0.2, 0.1, 0.1],
                     metavar=("CORNER", "SURFACE", "EDGE", "CORNER_EDGE", "NONE"),
                     help="Selection weights for the 5 damage types. CORNER_EDGE applies a "
                          "corner cluster and then a long-edge spall to the same brick, so "
                          "a brick can carry two distinct damage regions.")
parser.add_argument("--seed", type=int, default=None,
                     help="Random seed. Default: time-based (different every run).")
args = parser.parse_args()

brick_size = list(args.brick_size)
os.makedirs(args.stock_dir, exist_ok=True)

if args.seed is not None:
    random.seed(args.seed)
else:
    random.seed(time.time())

# Clean start -- generate_brick_stock.py doesn't need blenderproc's
# renderer/world/camera setup at all (no rendering happens here), just a
# clean Blender scene to build meshes in.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

intact_mesh_path = os.path.join(args.stock_dir, "intact_brick.obj")
if not os.path.exists(intact_mesh_path):
    intact_obj = create_single_brick(name="IntactReference", size=brick_size, remesh=True)
    export_mesh_as_obj(intact_obj, intact_mesh_path)
    full_brick_volume = get_volume(intact_obj)
    bpy.data.objects.remove(intact_obj, do_unlink=True)
else:
    # Reused across stock-generation runs appending to the same directory
    # -- rebuild a throwaway intact brick just to get its volume, without
    # re-exporting (the file already matches this brick_size by
    # construction, since intact_brick.obj is keyed by directory not size).
    tmp_obj = create_single_brick(name="VolumeRef", size=brick_size, remesh=True)
    full_brick_volume = get_volume(tmp_obj)
    bpy.data.objects.remove(tmp_obj, do_unlink=True)

manifest_path = os.path.join(args.stock_dir, "manifest.json")
manifest = {"brick_size": brick_size, "bricks": []}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        existing = json.load(f)
    if existing.get("brick_size") != brick_size:
        raise ValueError(
            f"{manifest_path} already has brick_size={existing.get('brick_size')}, "
            f"which doesn't match --brick-size {brick_size}. Use a different --stock-dir "
            "for a different brick size."
        )
    manifest = existing

start_index = len(manifest["bricks"])
begin = time.time()
num_generated = 0
num_aborted = 0

for i in range(start_index, start_index + args.num_bricks):
    obj = create_single_brick(name=f"Brick_{i:05d}", size=brick_size, remesh=True)

    # Pass 1: bevel the pristine box's own 12 edges before any damage
    # exists -- no competing geometry nearby yet, so this width applies
    # cleanly without needing the safety net. Was 0.25-0.45 -- that read
    # as the whole brick's silhouette being rounded/eroded rather than a
    # subtle chamfer on an otherwise sharp brick. See soften_edges'
    # docstring for why this has to be a separate pass rather than
    # relying on the post-damage pass alone.
    soften_edges(obj, brick_size, width_fraction_range=(0.10, 0.16))

    try:
        damage_type = apply_random_damage(
            obj, brick_size, full_brick_volume, weights=args.fracture_weights,
        )
    except DamageAborted as e:
        print(f"[{i:05d}] aborted: {e}")
        bpy.data.objects.remove(obj, do_unlink=True)
        num_aborted += 1
        continue

    # Clean up first: the damage above ran 5-15 sequential booleans, each
    # leaving slivers and coincident vertices for the next one to build
    # on. Worth doing on its own merits, but measured against the bevel
    # below it barely helped (kept widths 0.14/0.00/0.17 -> 0.16/0.00/
    # 0.35), so it is not what was blocking it. See clean_boolean_debris.
    clean_boolean_debris(obj)

    # Pass 2: bevel the boolean-cut edges from the damage above.
    #
    # The width here is scaled to the DAMAGE, not to the brick. Pass 1
    # rounds the 12 box edges with ~4-5mm, which is right for a feature
    # the size of a brick; re-using that range here asks for a 5mm fillet
    # on a groove whose own radius is 3-15mm, i.e. a bevel as large as
    # the thing it is rounding. That cannot be done without
    # self-intersection at any mesh quality, which is why
    # _safe_bevel_apply kept halving it away to 0.15mm and why cleaning
    # the mesh first changed nothing. Measured safe width on these
    # bricks is ~0.3-0.5mm; the halving search from 5mm samples 0.61 then
    # 0.30 and steps straight over it.
    soften_edges(obj, brick_size, width_fraction_range=(0.02, 0.05))

    # Last: organic surface roughness everywhere, including the flat
    # undamaged faces/edges Remesh+Bevel alone leave looking too regular.
    add_surface_irregularity(obj, brick_size)

    fname = f"brick_{i:05d}.obj"
    export_mesh_as_obj(obj, os.path.join(args.stock_dir, fname))
    manifest["bricks"].append({"file": fname, "damage_type": damage_type})
    num_generated += 1

    old_mesh = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(old_mesh)

    if num_generated % 50 == 0:
        elapsed = time.time() - begin
        print(f"[{num_generated}/{args.num_bricks}] "
              f"({elapsed:.1f}s elapsed, {elapsed / num_generated:.2f}s/brick)")

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"{num_generated} bricks generated in {time.time() - begin:.1f} seconds "
      f"({num_aborted} aborted) -> {args.stock_dir}")
