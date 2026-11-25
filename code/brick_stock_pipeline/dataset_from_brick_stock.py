import blenderproc as bproc

# Phase 2 of the two-phase synthetic brick pipeline: load one damaged
# brick geometry from a stock built by generate_brick_stock.py, apply a
# freshly randomized material (not baked into the stock -- see that
# script's docstring), render --num-views views of it, write BOP-format
# output, and grade per-view damage cell scores into coco_annotations.json
# -- the same rendering/grading logic generate_broken_bricks_v3.py runs,
# just starting from a pre-made brick instead of building+damaging one
# fresh. Meant to be invoked once per stock brick (loop it externally to
# process the whole stock), mirroring generate_broken_bricks_v3.py's own
# one-shot-per-invocation pattern.
#
# Usage:
#     blenderproc run dataset_from_brick_stock.py resources --stock-name brick_stock --brick-index 0 --num-views 10 --custom-blender-path ~/blender/blender-3.6.12-linux-x64
#
# To render the whole stock:
# SECONDS=0
# for i in $(seq 0 999); do
#     blenderproc run dataset_from_brick_stock.py resources \
#         --stock-name brick_stock --brick-index $i --num-views 15 \
#         --custom-blender-path ~/blender/blender-3.6.12-linux-x64
# done
# echo "Total: $((SECONDS / 3600))h $(((SECONDS % 3600) / 60))m $((SECONDS % 60))s for 1000 bricks"


import argparse
import contextlib
import glob
import json
import os
import re
import sys
import time

import bpy
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))
from brick_common import assign_random_brick_texture, import_mesh_from_obj, suppress_stdout
from custom_writer import write_my

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

intr = np.array([[607.156, 0, 324.652],
                  [0, 607.194, 236.489],
                  [0, 0, 1]]).astype(np.float32)
intr = intr.reshape(-1).tolist()

catId2name = {0: 'Unknown', 1: 'Brick'}

parser = argparse.ArgumentParser()
parser.add_argument("resources_dir", nargs="?", default=os.path.join(base_dir, "resources"),
                     help="Path to the resources directory (containing brick_material_v2.blend "
                          "and ccmaterials/).")
parser.add_argument("--stock-name", default="brick_stock",
                     help="Directory generate_brick_stock.py wrote (manifest.json, "
                          "intact_brick.obj, brick_NNNNN.obj files).")
parser.add_argument("--brick-index", type=int, required=True,
                     help="Which stock brick (manifest.json index) to render this run.")
parser.add_argument("--num-views", type=int, default=10,
                     help="Number of views to render of this one brick.")

args = parser.parse_args()

resources_dir = os.path.join(base_dir, args.resources_dir)
mat_dir = {'Brick': os.path.join(resources_dir, 'brick_material_v2.blend')}
stock_dir = os.path.join(base_dir, "data", args.stock_name)
out_dir = os.path.join(base_dir, "data", f"dataset_from_{args.stock_name}")

with open(os.path.join(stock_dir, "manifest.json")) as f:
    manifest = json.load(f)
if not (0 <= args.brick_index < len(manifest["bricks"])):
    raise IndexError(f"--brick-index {args.brick_index} out of range "
                      f"(stock has {len(manifest['bricks'])} bricks)")
brick_entry = manifest["bricks"][args.brick_index]
brick_obj_path = os.path.join(stock_dir, brick_entry["file"])
intact_mesh_path = os.path.join(stock_dir, "intact_brick.obj")
brick_size = manifest["brick_size"]

print(f"Rendering stock brick {args.brick_index}: {brick_entry['file']} "
      f"(damage_type={brick_entry['damage_type']})")

################################################################
# Scene setup (same as generate_broken_bricks_v3.py's scene_setup())
################################################################

def purge_orphans():
    if bpy.app.version >= (3, 0, 0):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    else:
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            purge_orphans()

def clean_scene():
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for name in [col.name for col in bpy.data.collections]:
        bpy.data.collections.remove(bpy.data.collections[name])
    for name in [world.name for world in bpy.data.worlds]:
        bpy.data.worlds.remove(bpy.data.worlds[name])
    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]
    purge_orphans()

clean_scene()

bproc.init()
K = np.array(intr).reshape(3, 3)
bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)

with bpy.data.libraries.load(mat_dir['Brick']) as (data_from, data_to):
    data_to.materials = ['Brick']

################################################################
# Load the stock brick + apply a fresh random material
################################################################

target_objects = []

def add_object_to_target_list(obj, target_objects, category_id=1, add_to_target=True):
    bproc_obj = bproc.object.convert_to_meshes(blender_objects=[obj])[0]
    if not add_to_target:
        bproc_obj.hide()
        category_id = 0
    bproc_obj.set_cp("category_id", category_id)
    bproc_obj.set_cp("category_name", catId2name[category_id])
    target_objects.append(bproc_obj)
    return bproc_obj

begin = time.time()

brick_obj = import_mesh_from_obj(brick_obj_path, name="Brick")
# Stock bricks are centered at the origin (see create_single_brick in
# brick_common.py) -- lift onto the ground plane (z=0) here, since no
# physics simulation runs to settle it there on its own.
brick_obj.location = (0, 0, brick_size[2] / 2)

material_begin = time.time()
assign_random_brick_texture(brick_obj)
print("Material assignment in {:.3f} seconds".format(time.time() - material_begin))

bproc_brick = add_object_to_target_list(brick_obj, target_objects, category_id=1, add_to_target=True)
# Tags which manifest.json entry this is so scene_gt.json can record it
# per-frame (see custom_writer.py's get_frame_gt(), which reads this same
# way it already reads category_id/used_source_id) -- label_damage_scores.py
# reads it back to know which brick_stock/*.obj to grade against, instead
# of having to reconstruct that from render order.
bproc_brick.set_cp("stock_brick_file", brick_entry["file"])
top_layer_bricks = [bproc_brick]

################################################################
# Rendering with blenderproc (same as generate_broken_bricks_v3.py)
################################################################

begin = time.time()

for j, obj in enumerate(target_objects):
    mass, fiction_coeff = (0.4, 0.5)
    obj.enable_rigidbody(True, mass=mass, friction=mass * fiction_coeff,
                          linear_damping=1.99, angular_damping=0, collision_margin=0.0001)
    obj.set_shading_mode('auto')

room = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]

light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane.replace_materials(light_plane_material)

cc_textures_path = os.path.join(resources_dir, 'ccmaterials')
cc_textures = []
if os.path.exists(cc_textures_path):
    cc_textures = bproc.loader.load_ccmaterials(cc_textures_path, use_all_materials=True)
if cc_textures:
    for plane in room:
        random_cc_texture = np.random.choice(cc_textures)
        plane.replace_materials(random_cc_texture)

room.append(light_plane)
for plane in room:
    plane.enable_rigidbody(False, collision_shape='BOX',
                            friction=100, linear_damping=0.0, angular_damping=0.0)
    plane.set_cp('category_id', 0)
    plane.set_cp("category_name", 'Unknown')

light_point = bproc.types.Light()

bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(target_objects)

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"])

# room[0] is the floor plane (see its construction above -- the only one
# with no location/rotation override, i.e. lying flat at the origin).
floor = room[0]

i = 0
radius_min, radius_max = (0.4, 0.8)
camera_height_min, camera_height_max = (0.2, 0.7)
wall_top_z = brick_size[2] / 2

# Group consecutive views under the same floor material: VIEWS_PER_MATERIAL
# views share one material, then the next group gets a different one (never
# the same material as the immediately preceding group). Track *which* group
# we last assigned a material for (not just i % VIEWS_PER_MATERIAL) since the
# obstacle-check continues below can retry the same i without advancing it --
# a naive modulo check would re-roll the material on every failed retry.
VIEWS_PER_MATERIAL = 5
current_material = None
current_material_group = -1

# write_my/write_frames determine how many new frames to write from Blender's
# own scene frame_start/frame_end (advanced by add_camera_pose), not from the
# length of the lists passed in -- and they already support incrementally
# appending to an existing chunk on disk. So call write_my once per view here
# (right after that view's single-frame render + keyframe reset) rather than
# batching all views into one call at the end, which would leave Blender's
# frame range reflecting only the last reset (0 frames) by the time write_my
# ran, silently dropping the whole batch.
while i < args.num_views:
    material_group = i // VIEWS_PER_MATERIAL
    if material_group != current_material_group:
        next_material = np.random.choice(cc_textures)
        while len(cc_textures) > 1 and next_material is current_material:
            next_material = np.random.choice(cc_textures)
        current_material = next_material
        current_material_group = material_group
        floor.replace_materials(current_material)

        # Fresh lighting every VIEWS_PER_MATERIAL views too, same cadence
        # as the floor material.
        light_plane_material.make_emissive(
            emission_strength=np.random.uniform(3, 8),
            emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
        light_point.set_energy(np.random.uniform(0, 100))
        light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
        light_point.set_location(bproc.sampler.shell(
            center=[0, 0, 0], radius_min=1, radius_max=1.5,
            elevation_min=5, elevation_max=89, uniform_volume=False))

    inplane_rot = np.random.uniform(-0.35, 0.35)
    camera_radius = np.random.uniform(low=radius_min, high=radius_max)
    camera_height = np.random.uniform(low=camera_height_min, high=camera_height_max)
    location = bproc.sampler.part_sphere([0, 0, camera_height],
                                          radius=camera_radius, dist_above_center=0.0, mode="SURFACE")

    poi = bproc.object.compute_poi(np.random.choice(target_objects, size=min(4, len(target_objects))))
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane_rot)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

    if not bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
        continue
    if location[2] >= (wall_top_z + camera_radius - 0.05):
        continue

    bproc.camera.add_camera_pose(cam2world_matrix)
    with suppress_stdout():
        data = bproc.renderer.render()

    depth_noise = bproc.postprocessing.add_kinect_azure_noise(data["depth"], data["colors"], 10)

    write_my(out_dir,
             dataset='',
             target_objects=target_objects,
             depths=data["depth"],
             depths_noise=depth_noise,
             colors=data["colors"],
             instance_masks=data["instance_segmaps"],
             category_masks=data["category_id_segmaps"],
             instance_attribute_maps=data["instance_attribute_maps"],
             color_file_format="JPEG",
             ignore_dist_thres=10,
             frames_per_chunk=1000,
             is_shapenet=False)

    bproc.utility.reset_keyframes()
    i += 1

print("{} renders generated in {:.3f} seconds".format(args.num_views, time.time() - begin))

################################################################
# Per-view damage cell grading, same mechanism as generate_broken_bricks_v3.py
# Disabled for now -- training without damage_cell_data.
#
# NOTE: this block predates the per-view write_my() refactor above and
# references all_depths/all_category_masks, which no longer exist (write_my
# is now called once per view with that view's single-frame data, not
# accumulated). Re-enabling this as-is will NameError; it needs reworking to
# pull per-view depth/category data back from what dataset_from_brick_stock's
# write loop actually produces (e.g. re-read from disk, same as
# scene_camera.json/scene_gt.json already are below).
################################################################

# damage_cell_grader = BrickDamageGrader(intact_mesh_path)
#
# scene_dirs = sorted(glob.glob(os.path.join(out_dir, "train_pbr", "*")))
# scene_dir = max(scene_dirs, key=os.path.getmtime)
# with open(os.path.join(scene_dir, "scene_camera.json")) as f:
#     scene_camera = json.load(f)
# with open(os.path.join(scene_dir, "scene_gt.json")) as f:
#     scene_gt = json.load(f)
#
# num_views = len(all_depths)
# this_run_frame_ids = sorted((int(k) for k in scene_gt.keys()))[-num_views:]
#
# damage_cell_gt_by_frame = {}
# for view_idx, frame_id in enumerate(this_run_frame_ids):
#     frame_key = str(frame_id)
#     obj_entry = next((o for o in scene_gt[frame_key] if o["obj_id"] == 1), None)
#     if obj_entry is None:
#         continue
#
#     intr_frame = np.array(scene_camera[frame_key]["cam_K"], dtype=np.float64).reshape(3, 3)
#     R = np.array(obj_entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
#     t = np.array(obj_entry["cam_t_m2c"], dtype=np.float64) / 1000.0
#     pose = np.eye(4)
#     pose[:3, :3] = R
#     pose[:3, 3] = t
#
#     depth_view = np.asarray(all_depths[view_idx])
#     mask_view = np.asarray(all_category_masks[view_idx])
#     if mask_view.ndim == 3:
#         mask_view = mask_view[..., 0]
#     mask_view = (mask_view == 1)
#
#     pts = to_canonical(depth_view, mask_view, intr_frame, pose, erode_px=2)
#     scores, counts = damage_cell_grader.grade(pts, min_pts=15, return_counts=True)
#     damage_cell_gt_by_frame[frame_id] = (scores, counts)
#
#     score_strs = [
#         f"{b0}{b1}{b2}=" + ("n/a" if np.isnan(s) else f"{s:.2f}")
#         for (b0, b1, b2), s in zip(damage_cell_grader.damage_cell_bins, scores)
#     ]
#     print(f"Damage cell scores (frame {frame_id}, {damage_cell_grader.num_damage_cells} cells): " + " ".join(score_strs))
#
# coco_path = os.path.join(scene_dir, "coco_annotations.json")
# with open(coco_path) as f:
#     coco = json.load(f)
#
# image_id_to_frame = {}
# for img in coco["images"]:
#     m = re.search(r"(\d+)\.\w+$", img["file_name"])
#     if m:
#         image_id_to_frame[img["id"]] = int(m.group(1))
#
# patched = 0
# for ann in coco["annotations"]:
#     frame_id = image_id_to_frame.get(ann["image_id"])
#     if frame_id not in damage_cell_gt_by_frame:
#         continue
#     scores, counts = damage_cell_gt_by_frame[frame_id]
#     ann["damage_cell_scores"] = [None if np.isnan(s) else float(s) for s in scores]
#     ann["damage_cell_counts"] = [int(c) for c in counts]
#     ann["source_brick"] = brick_entry["file"]
#     ann["source_damage_type"] = brick_entry["damage_type"]
#     patched += 1
#
# with open(coco_path, "w") as f:
#     json.dump(coco, f, indent=2)
# print(f"Patched {patched} coco annotation(s) with damage cell scores -> {coco_path}")
