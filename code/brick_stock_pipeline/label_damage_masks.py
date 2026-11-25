"""
Per-pixel damage-mask labeling for the segmentation-head version of the
damage-prediction project: dense 2D labels sidestep the symmetry-labeling
problem and the pose-conditioned train/inference mismatch that the
per-cell regression head kept hitting (see BrickDamageGrader in
code/allocation_rendering_pipeline/damage_grader.py -- signed_distance_image()/
damage_pixel_label()/out_of_bounds_mask() are the new pixel-level
primitives this uses).

Usage:
    # look at one frame first -- writes an overlay PNG to --out-dir,
    # nothing into the dataset itself:
    python label_damage_masks.py --visualize-one train_pbr/000000 42

    # once that looks right, label a whole chunk (or --all of them):
    python label_damage_masks.py --chunk train_pbr/000000
    python label_damage_masks.py --all
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "allocation_rendering_pipeline"))
from damage_grader import BrickDamageGrader, damage_pixel_label_classes

# NOT importing evaluation.visualization.visualize_signed_distance_image here
# even though it does the same thing: visualization.py's own top-level
# "from evaluation.evaluation import ..." forces Python to execute
# evaluation/__init__.py (-> evaluation.evaluation -> sklearn) as a side
# effect of resolving that import, regardless of how *this* script imports
# visualization.py -- and this repo's env (gen_env) doesn't have sklearn
# installed (that's brick_perception's own full env). damage_grader.py
# avoids this because it's imported bare (see the sys.path trick above,
# same one label_damage_scores.py already uses) and has no such internal
# package-qualified import itself. Duplicated here instead (small: just
# the color-ramp math) to keep this script's own dependencies down to
# cv2/numpy, matching label_damage_scores.py.
_SIGNED_DAMAGE_COLOR_STOPS = [
    (-1.0, (1.0, 0.0, 0.0)),  # red: material loss
    (0.0, (0.0, 0.8, 0.0)),   # green: matches the intact reference
    (1.0, (0.0, 0.0, 1.0)),   # blue: material addition / mis-registration
]


def visualize_signed_distance_image(bgr, sdist_img, max_dist=0.01, alpha=0.6):
    """Local copy of evaluation.visualization.visualize_signed_distance_image
    -- see that function's docstring (same color convention as
    signed_damage_cell_score_color there); duplicated rather than imported,
    see the comment above."""
    observed = ~np.isnan(sdist_img)
    normalized = np.clip(sdist_img / max_dist, -1.0, 1.0)
    stops = np.array([s for s, _ in _SIGNED_DAMAGE_COLOR_STOPS])
    colors_rgb = np.array([c for _, c in _SIGNED_DAMAGE_COLOR_STOPS])
    flat = normalized[observed]
    r = np.interp(flat, stops, colors_rgb[:, 0])
    g = np.interp(flat, stops, colors_rgb[:, 1])
    b = np.interp(flat, stops, colors_rgb[:, 2])
    colored_bgr = np.stack([b, g, r], axis=1) * 255.0
    overlay = bgr.astype(np.float64).copy()
    overlay[observed] = (1 - alpha) * overlay[observed] + alpha * colored_bgr
    return np.clip(overlay, 0, 255).astype(np.uint8)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Stored label values. 0-2 come from damage_pixel_label_classes(); 3 is
# unobserved (masked OUT of the loss); 4 is the environment band added
# here, which unlike 3 IS trained on, as a negative.
CLASS_ENV = 4

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset-dir", default=os.path.join(base_dir, "data", "dataset_from_brick_stock"),
                     help="Directory containing train_pbr/test_pbr chunk folders.")
parser.add_argument("--stock-dir", default=os.path.join(base_dir, "data", "brick_stock"),
                     help="Directory with intact_brick.obj.")
parser.add_argument("--surface-tol", type=float, default=0.003,
                     help="meters -- see BrickDamageGrader.grade()/damage_pixel_label().")
parser.add_argument("--erode-px", type=int, default=2,
                     help="shrink mask_visib by this many pixels before backprojecting, "
                          "same silhouette-noise guard as label_damage_scores.py.")
parser.add_argument("--penetration-tol", type=float, default=0.005,
                     help="meters -- see damage_pixel_label_full(): how far past the "
                          "intact brick's expected surface the observed depth must land "
                          "before a pixel counts as missing material.")
parser.add_argument("--env-band", type=float, default=0.12,
                     help="width of the environment band labelled as class 4, as a "
                          "fraction of sqrt(brick pixel area) -- scales with apparent "
                          "size, clamped to 4-40 px. These become trained NEGATIVES, so "
                          "the head learns that background beside the brick is not "
                          "damage while background seen THROUGH it (class 2) is. 0 "
                          "disables, restoring the previous all-background-unobserved "
                          "behaviour.")
parser.add_argument("--mask-dir-name", default="damage_mask",
                     help="new per-chunk subfolder name the bulk modes write into.")

mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--visualize-one", nargs=2, metavar=("CHUNK", "FRAME_ID"),
                   help="e.g. --visualize-one train_pbr/000000 42 -- writes an overlay "
                        "PNG + prints stats for one frame, touches nothing in the dataset.")
mode.add_argument("--chunk", help="label every frame in one chunk, e.g. train_pbr/000000")
mode.add_argument("--all", action="store_true", help="label every chunk under --dataset-dir")

parser.add_argument("--out-dir", default="data/dataset_from_brick_stock/damage_mask_preview",
                     help="where --visualize-one writes its overlay PNG.")
parser.add_argument("--overwrite", action="store_true",
                     help="bulk modes: recompute frames that already have a damage_mask file "
                          "(default: skip them).")
args = parser.parse_args()

grader = BrickDamageGrader(os.path.join(args.stock_dir, "intact_brick.obj"))
print(f"Damage grader: {grader.num_damage_cells} cells (unused here -- signed_distance_image() "
      f"works at pixel resolution, not cells)")


def load_frame(chunk_dir, frame_id):
    """Loads everything needed to grade one frame: rgb, clean depth (meters),
    mask_visib, intrinsics, and the object->camera pose. Same file layout
    label_damage_scores.py already reads."""
    with open(os.path.join(chunk_dir, "scene_camera.json")) as f:
        scene_camera = json.load(f)
    with open(os.path.join(chunk_dir, "scene_gt.json")) as f:
        scene_gt = json.load(f)
    frame_key = str(frame_id)

    obj_entry = next((o for o in scene_gt[frame_key] if o["obj_id"] == 1), None)
    if obj_entry is None:
        raise SystemExit(f"frame {frame_id} in {chunk_dir} has no obj_id==1 entry in scene_gt.json")

    rgb_path = os.path.join(chunk_dir, "rgb", f"{frame_id:06d}.jpg")
    depth_path = os.path.join(chunk_dir, "depth", f"{frame_id:06d}.png")
    mask_path = os.path.join(chunk_dir, "mask_visib", f"{frame_id:06d}_000000.png")
    for p in (rgb_path, depth_path, mask_path):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0  # mm -> m
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) > 0

    intr = np.array(scene_camera[frame_key]["cam_K"], dtype=np.float64).reshape(3, 3)
    R = np.array(obj_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.array(obj_entry["cam_t_m2c"], dtype=np.float64) / 1000.0  # mm -> m
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t

    return rgb, depth, mask, intr, pose


def env_band_mask(mask, band_fraction):
    """Background pixels in a band just outside the brick.

    These become class 4 (environment), a NEGATIVE the head is trained
    on, instead of being dropped as unobserved.

    Why it matters: every pixel off the brick is currently class 3, and
    the trainer masks class 3 out of the loss entirely -- 93.8% of each
    image contributes nothing. So the head is never told that background
    is not damage, while "missing material" (class 2) IS background seen
    where brick should be. At inference it therefore has no basis for
    telling those two apart, which is what `pose_damage_mask` is
    currently compensating for: the pose is supplying a decision the
    head was never taught to make.

    Only a BAND, not all background: labelling all 93.8% negative would
    swamp the loss with trivially-easy far-field pixels and drop the
    positive rate from 9.2% to 0.6%, forcing a pos_weight retune. The
    confusion lives at the brick's outline, so that is what gets
    supervised.

    The width scales with the brick's apparent size (sqrt of its pixel
    area) rather than being a fixed pixel count -- the dataset crops
    around the brick and resizes to 256, so a fixed width would mean
    very different things on a near and a far brick.

    BUILT FROM THE CONVEX HULL, not from mask_visib directly, and that
    is the whole correctness argument. Background seen THROUGH a break
    is not part of mask_visib either -- the renderer labels those pixels
    floor, not brick -- so dilating the raw mask would sweep them into
    the band and teach the head that missing material is NOT damage.
    Precisely inverted, and precisely on the pixels this class exists to
    disambiguate. A missing chunk makes the silhouette concave; the hull
    fills that notch back in, so everything inside it is excluded and
    only true outside-the-brick background is labelled.

    Limitation: damage that leaves the silhouette convex (a whole end
    sheared off square) is invisible to the hull. That is harmless here
    -- with no concavity there is no see-through region to protect, the
    brick simply reads as smaller.
    """
    area = int(mask.sum())
    if area <= 0:
        return np.zeros_like(mask, dtype=bool)

    pts = cv2.findNonZero(mask.astype(np.uint8))
    hull = cv2.convexHull(pts)
    hull_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    band_px = int(np.clip(band_fraction * np.sqrt(area), 4, 40))
    k = np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)
    dilated = cv2.dilate(hull_mask, k, iterations=1) > 0
    return dilated & ~(hull_mask > 0)


def label_one_frame(chunk_dir, frame_id):
    """Returns (rgb, sdist_img, label) with label in {-1 unobserved,
    0 intact, 1 surface damage, 2 missing material, 4 environment} --
    see damage_pixel_label_classes(). Class 2 combines out_of_bounds_mask()
    (backprojects far outside the mesh box) and penetration_depth_image()
    (observed depth lands well past where this pixel's ray would first
    hit the intact brick -- the direction-aware check that catches a
    corner broken down to the base, whose exposed floor lies ON the
    intact mesh's own bottom-face plane and so reads ~0mm signed
    distance despite tens of mm of missing material above it).

    Class 4 is added here rather than in damage_pixel_label_classes():
    it is not damage evidence at all, it is the absence of it, and it
    comes from mask_visib rather than from any of the three signed
    distance / out-of-bounds / penetration sources that function
    combines."""
    rgb, depth, mask, intr, pose = load_frame(chunk_dir, frame_id)
    sdist_img = grader.signed_distance_image(depth, mask, intr, pose, erode_px=args.erode_px)
    oob_mask = grader.out_of_bounds_mask(depth, mask, intr, pose, erode_px=args.erode_px)
    penetration_img = grader.penetration_depth_image(depth, mask, intr, pose, erode_px=args.erode_px)
    label = damage_pixel_label_classes(
        sdist_img, oob_mask, penetration_img,
        surface_tol=args.surface_tol, penetration_tol=args.penetration_tol,
    )
    if args.env_band > 0:
        # Only where nothing was observed: never overwrite a real damage
        # or intact label with "environment".
        band = env_band_mask(mask, args.env_band) & (label < 0)
        label[band] = CLASS_ENV
    return rgb, sdist_img, label


def find_chunks():
    chunks = []
    for split in ["train_pbr", "test_pbr"]:
        split_dir = os.path.join(args.dataset_dir, split)
        for chunk_dir in sorted(glob.glob(os.path.join(split_dir, "*"))):
            if os.path.isdir(chunk_dir):
                chunks.append(os.path.join(split, os.path.basename(chunk_dir)))
    return chunks


def frame_ids_in_chunk(chunk_dir):
    with open(os.path.join(chunk_dir, "scene_gt.json")) as f:
        scene_gt = json.load(f)
    return sorted(int(k) for k in scene_gt.keys())


def bulk_label_chunk(chunk_rel):
    chunk_dir = os.path.join(args.dataset_dir, chunk_rel)
    out_dir = os.path.join(chunk_dir, args.mask_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    labeled, skipped = 0, 0
    for frame_id in frame_ids_in_chunk(chunk_dir):
        out_path = os.path.join(out_dir, f"{frame_id:06d}.png")
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            continue
        _, _, label = label_one_frame(chunk_dir, frame_id)
        # label is {-1, 0, 1, 2}; store unobserved (-1) as 3 so it's a plain
        # uint8 PNG. Binary training maps {1, 2} -> damaged and ignores 3.
        stored = np.where(label < 0, 3, label).astype(np.uint8)
        cv2.imwrite(out_path, stored)
        labeled += 1
    print(f"  {chunk_rel}: labeled {labeled}, skipped {skipped} already-labeled "
          f"-> {out_dir} (0=intact, 1=surface damage, 2=missing material, "
          f"3=unobserved, {CLASS_ENV}=environment)")
    return labeled, skipped


if args.visualize_one:
    chunk_rel, frame_id_str = args.visualize_one
    chunk_dir = os.path.join(args.dataset_dir, chunk_rel)
    frame_id = int(frame_id_str)

    rgb, sdist_img, label = label_one_frame(chunk_dir, frame_id)
    # "observed" = everything the loss sees, which now includes the env
    # band; "on brick" is the subset the damage fractions are quoted
    # against, so adding the band cannot silently deflate them.
    observed = label >= 0
    on_brick = observed & (label != CLASS_ENV)
    n_obs = int(on_brick.sum())
    n_env = int((label == CLASS_ENV).sum())
    n_surface = int((label == 1).sum())
    n_missing = int((label == 2).sum())
    n_damaged = n_surface + n_missing
    print(f"{chunk_rel} frame {frame_id}: {n_obs} observed pixels, {n_damaged} damaged total "
          f"({n_surface} surface damage + {n_missing} missing material), "
          f"{100.0 * n_damaged / max(n_obs, 1):.1f}% of observed, "
          f"surface_tol={args.surface_tol * 1000:.1f}mm, "
          f"penetration_tol={args.penetration_tol * 1000:.1f}mm")
    print(f"  env band: {n_env} pixels labelled class {CLASS_ENV} (trained negatives), "
          f"{n_env / max(n_obs, 1):.2f}x the on-brick pixel count "
          f"(--env-band {args.env_band})")
    on_surface_observed = on_brick & (label != 2)
    if on_surface_observed.any():
        obs_sdist = sdist_img[on_surface_observed]
        print(f"  signed distance over on-surface pixels: min={obs_sdist.min()*1000:.1f}mm "
              f"max={obs_sdist.max()*1000:.1f}mm mean={obs_sdist.mean()*1000:.1f}mm")

    overlay = visualize_signed_distance_image(rgb, sdist_img, max_dist=0.01, alpha=0.6)
    overlay[label == 2] = (0, 165, 255)  # BGR orange: missing material
    # Surface damage as red OUTLINES here (solid fill in the class panel on
    # the right): the continuous colormap underneath is exactly what you
    # want to see inside these regions (loss vs addition, how deep), so
    # don't paint over it.
    surface_u8 = (label == 1).astype(np.uint8)
    contours, _ = cv2.findContours(surface_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    class_vis = np.zeros_like(rgb)
    class_vis[label == 0] = (0, 200, 0)    # BGR green: intact
    class_vis[label == 1] = (0, 0, 255)    # BGR red: surface damage
    class_vis[label == 2] = (0, 165, 255)  # BGR orange: missing material
    class_vis[label == CLASS_ENV] = (200, 80, 0)   # BGR blue: environment band
    # Also outline the band on the overlay, so it can be checked against
    # the brick's real outline in the RGB beside it -- the whole point of
    # this class is that it hugs the silhouette, and a band that has
    # drifted or swallowed the brick is obvious here and nowhere else.
    env_u8 = (label == CLASS_ENV).astype(np.uint8)
    # CCOMP, not EXTERNAL: the band is an annulus, and EXTERNAL returns
    # only its outer boundary -- the inner one, which is what shows
    # whether the band hugs the brick, gets silently dropped.
    env_contours, _ = cv2.findContours(env_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, env_contours, -1, (200, 80, 0), 1)
    side_by_side = np.concatenate([rgb, overlay, class_vis], axis=1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"{chunk_rel.replace('/', '_')}_frame{frame_id:06d}.png"
    )
    cv2.imwrite(out_path, side_by_side)
    print(f"  wrote rgb | signed-distance overlay (red outline = surface damage) | "
          f"class label (green=intact, red=surface damage, orange=missing material, "
          f"blue=environment band), side by side, to {out_path}")

elif args.chunk:
    bulk_label_chunk(args.chunk)

elif args.all:
    total_labeled, total_skipped = 0, 0
    for chunk_rel in find_chunks():
        labeled, skipped = bulk_label_chunk(chunk_rel)
        total_labeled += labeled
        total_skipped += skipped
    print(f"Done. Total labeled: {total_labeled}, total skipped (already labeled): {total_skipped}")
