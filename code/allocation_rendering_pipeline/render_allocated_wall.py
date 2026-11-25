import blenderproc as bproc

# Phase 3 of the synthetic pipeline: renders build_demo_wall.py's allocated
# wall using each occupant's ACTUAL stock brick geometry -- not a generic or
# freshly-damaged brick -- positioned at its assigned slot frame, so the
# "bad" half of the wall visibly shows real chipped/damaged silhouettes and
# the "good" half visibly shows near-intact ones. This is the "simulated
# renderer" for validating the allocation algorithm end-to-end, distinct
# from dataset_from_brick_stock.py's per-brick training-data renders.
#
# Occupant -> source geometry lookup goes back through
# `occupant.attributes["source_key"]` into the ORIGINAL stock manifest.json,
# not through any attribute stored on the occupant's own CellularizedPart
# copy: CellularizedPart.copy() (see cellularized_part.py) only carries
# forward shape/mesh/frame/grid/box_mode/name -- not arbitrary custom
# attributes like the "source_obj" filename grade_brick_stock.py sets on
# each stock part -- so relying on that would silently break. `source_key`
# IS reliably carried through (allocate.py's place() sets it explicitly on
# every occupant) and, by construction, matches the stock brick's manifest
# index: grade_brick_stock.py builds the stock assembly with key=i in
# exactly manifest order. So manifest["bricks"][occupant.attributes[
# "source_key"]] is always the right OBJ file. If that assumption ever
# changes (a re-keyed or filtered stock), either fix CellularizedPart.copy()
# to preserve custom attributes, or store source_obj directly on the
# allocation result in allocate.place() instead.
#
# Occupants WITHOUT a source_key (slots no stock brick was allocated to) are
# built as a generic full brick instead: a plain box at the occupant's
# shape size (xsize/ysize/zsize), placed at its frame, with no material.
#
# Each occupant's CellularizedPart.mesh is just the plain intact box used
# for grading/allocation bookkeeping (see grade_brick_stock.py) -- not the
# real damaged geometry, and not used for rendering here. Keeping the
# allocated-wall JSON to box+grid+scores+frame (no dense mesh data) is
# deliberate: real per-brick geometry is always re-resolved from
# --stock-dir at render time via source_key, so the JSON stays small even
# for a 10,000-brick stock.
#
# occupant.frame is applied directly as the imported brick's matrix_world,
# per masonry.py's own stated convention ("Visualization/export is expected
# to place each part's geometry using its frame itself") -- valid here
# because import_mesh_from_obj loads the brick centered at its own local
# origin, exactly like brick_common.create_single_brick built it.
#
# Usage (run from this folder; data/ is two levels up, at the repo root):
#     blenderproc run render_allocated_wall.py resources \
#         --allocated-wall ../../data/gradient_allocated_long_wall.json \
#         --stock-dir ../../data/brick_stock \
#         --output ../../data/wall_render.png \
#         --custom-blender-path ~/blender/blender-3.6.12-linux-x64
#
# Each trailing "\" must be the last character on its line; on a single line,
# drop them all, or the arguments after them aren't recognised (e.g.
# --custom-blender-path is ignored and BlenderProc downloads its own Blender).
#
# Save a .blend instead of rendering (for manual camera placement):
#     ... --save-blend ../../data/gradient_allocated_long_wall.blend
#
# Add the mortar core (needs shapely:
# blenderproc pip install shapely --custom-blender-path ~/blender/blender-3.6.12-linux-x64):
#     ... --mortar
#
# Several walls into one .blend, one collection each:
#     ... --allocated-wall ../../data/wall_a.json ../../data/wall_b.json --save-blend ../../data/walls.blend

import argparse
import json
import os
import sys

import bpy
import compas
import mathutils
import numpy as np

# CellularizedPart/Part need to be imported (even though unused directly
# here) so compas's DataDecoder can resolve their dtype when json_load
# reconstructs the allocated wall below -- same reasoning as
# allocate_demo_wall.py's imports.
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))
from masonry_allocation import CellularizedPart  # noqa: F401, E402

sys.path.append(script_dir)
sys.path.append(os.path.join(os.path.dirname(script_dir), "brick_stock_pipeline"))
from brick_common import assign_random_brick_texture, import_mesh_from_obj, suppress_stdout

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

parser = argparse.ArgumentParser()
parser.add_argument("resources_dir", nargs="?", default=os.path.join(base_dir, "resources"))
parser.add_argument("--allocated-wall", required=True, nargs="+",
                     help="Output of build_demo_wall.py. Several can be given -- each wall is "
                          "built into its own collection (named after the JSON file) in the "
                          "same scene, at its own frames.")
parser.add_argument("--stock-dir", required=True, nargs="+",
                     help="generate_brick_stock.py's stock dir (for manifest.json + brick_NNNNN.obj). "
                          "Either one dir shared by every --allocated-wall, or one per wall, in order.")
parser.add_argument("--output", default=os.path.join(base_dir, "data", "wall_render.png"))
parser.add_argument("--shared-material", action="store_true",
                     help="Randomize the brick material once and reuse it for every brick, "
                          "instead of per-brick (dataset_from_brick_stock.py's default) -- "
                          "faster for a wall with hundreds of bricks, and arguably reads more "
                          "coherently as 'one wall' in a hero shot.")
parser.add_argument("--save-blend",
                     help="Instead of auto-framing and rendering, save the built scene (wall + "
                          "backdrop + light + one starting camera) to this .blend path so it can "
                          "be opened in Blender for manual camera placement and rendering.")
parser.add_argument("--mortar", action="store_true",
                     help="Add a mortar core inside each wall, extruded bottom to top from the "
                          "wall's footprint and recessed 1cm from the true brick faces so it only "
                          "shows through the gaps between bricks -- follows curved/angled walls, and "
                          "is cheaper than modeling every individual joint. Textured with resources/"
                          "mortar_material.blend's 'Mortar' material. Needs shapely "
                          "(blenderproc pip install shapely).")
args = parser.parse_args()

resources_dir = os.path.join(base_dir, args.resources_dir)
mat_path = os.path.join(resources_dir, 'brick_material_v2.blend')

if len(args.stock_dir) == 1:
    stock_dirs = args.stock_dir * len(args.allocated_wall)
elif len(args.stock_dir) == len(args.allocated_wall):
    stock_dirs = args.stock_dir
else:
    parser.error("--stock-dir takes either 1 dir or one per --allocated-wall "
                 "(got {} for {} walls)".format(len(args.stock_dir), len(args.allocated_wall)))

# (collection name, occupants, stock dir, manifest) per wall
walls = []
for wall_path, stock_dir in zip(args.allocated_wall, stock_dirs):
    with open(os.path.join(stock_dir, "manifest.json")) as f:
        manifest = json.load(f)
    wall = compas.json_load(wall_path)
    occupants = list(wall.parts())
    coll_name = os.path.splitext(os.path.basename(wall_path))[0]
    walls.append((coll_name, occupants, stock_dir, manifest))
    print("Rendering {} occupants from {}".format(len(occupants), wall_path))

################################################################
# Scene setup (same pattern as dataset_from_brick_stock.py's, minus the
# physics/rigidbody machinery -- occupants are already precisely placed by
# build_demo_wall.py, nothing needs to be dropped/settled).
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

# 3x the 640x480 training-data intrinsics' resolution, scaling the K matrix
# proportionally so the field of view/framing stays identical -- a hero
# shot benefits from more pixels than what's needed for training data.
render_scale = 3
intr = np.array([[607.156, 0, 324.652],
                  [0, 607.194, 236.489],
                  [0, 0, 1]]).astype(np.float32)
intr[:2, :] *= render_scale
bproc.camera.set_intrinsics_from_K_matrix(intr, 640 * render_scale, 480 * render_scale)

with bpy.data.libraries.load(mat_path) as (data_from, data_to):
    data_to.materials = ['Brick']

################################################################
# Frame -> Blender matrix_world
################################################################

def frame_to_matrix_world(frame):
    x, y, z = frame.xaxis, frame.yaxis, frame.zaxis
    p = frame.point
    return mathutils.Matrix((
        (x[0], y[0], z[0], p[0]),
        (x[1], y[1], z[1], p[1]),
        (x[2], y[2], z[2], p[2]),
        (0.0, 0.0, 0.0, 1.0),
    ))

def create_generic_brick(shape, name):
    """Box of shape.xsize/ysize/zsize centered at its local origin, like the stock OBJs."""
    hx, hy, hz = shape.xsize / 2.0, shape.ysize / 2.0, shape.zsize / 2.0
    verts = [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz),
             (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    faces = [(0, 1, 2, 3), (0, 3, 5, 4), (3, 2, 6, 5), (2, 1, 7, 6), (1, 0, 4, 7), (4, 5, 6, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj

################################################################
# Import every occupant's real damaged geometry at its assigned frame
# (occupants without a source_key get a generic box, see above)
################################################################

target_objects = []
shared_material_applied = False

wall_min = np.array([np.inf, np.inf, np.inf])
wall_max = np.array([-np.inf, -np.inf, -np.inf])

# True geometric bounds (actual mesh surfaces, not just brick centers).
# occupant.frame.point is each brick's CENTER, so wall_min/wall_max below
# sit half a brick-height/width/depth inside the wall's real surfaces --
# fine for camera framing (a rough size estimate is enough), but wrong for
# anything that needs to touch an actual surface: the floor has to sit at
# the true bottom of the lowest course, not at its center, or it ends up
# slicing through the middle of the first course instead of sitting under
# it. Also used by --mortar's facade inset below.
facade_min = np.array([np.inf, np.inf, np.inf])
facade_max = np.array([-np.inf, -np.inf, -np.inf])

# Per wall: (collection, [world-space bound_box corners of each brick]) --
# the input for --mortar's per-course footprints below.
wall_brick_corners = []

for coll_name, occupants, stock_dir, manifest in walls:
    wall_coll = bpy.data.collections.new(coll_name)
    bpy.context.scene.collection.children.link(wall_coll)
    brick_corners = []
    wall_brick_corners.append((wall_coll, brick_corners))

    for occupant in occupants:
        source_key = occupant.attributes.get("source_key")

        # Occupant keys restart at 0 in every wall, so prefix with the wall's
        # name when there's more than one, rather than letting Blender tack
        # on .001 suffixes.
        brick_name = "brick_{}".format(occupant.key)
        if len(walls) > 1:
            brick_name = "{}_{}".format(coll_name, brick_name)

        if source_key is None:
            # No stock brick allocated to this slot: a generic full brick
            # (box at the occupant's shape size) with no material.
            brick_obj = create_generic_brick(occupant.shape, brick_name)
        else:
            brick_entry = manifest["bricks"][source_key]
            obj_path = os.path.join(stock_dir, brick_entry["file"])
            brick_obj = import_mesh_from_obj(obj_path, name=brick_name)
        brick_obj.matrix_world = frame_to_matrix_world(occupant.frame)
        for coll in list(brick_obj.users_collection):
            coll.objects.unlink(brick_obj)
        wall_coll.objects.link(brick_obj)

        corners = np.array([brick_obj.matrix_world @ mathutils.Vector(c) for c in brick_obj.bound_box])
        brick_corners.append(corners)
        facade_min = np.minimum(facade_min, corners.min(axis=0))
        facade_max = np.maximum(facade_max, corners.max(axis=0))

        if source_key is None:
            pass
        elif args.shared_material:
            if not shared_material_applied:
                assign_random_brick_texture(brick_obj)
                shared_mat = brick_obj.active_material
                shared_material_applied = True
            else:
                brick_obj.data = brick_obj.data.copy()
                brick_obj.active_material = shared_mat
        else:
            assign_random_brick_texture(brick_obj)

        bproc_obj = bproc.object.convert_to_meshes(blender_objects=[brick_obj])[0]
        bproc_obj.set_cp("category_id", 1)
        bproc_obj.set_cp("category_name", "Brick")
        target_objects.append(bproc_obj)

        p = np.array(occupant.frame.point)
        wall_min = np.minimum(wall_min, p)
        wall_max = np.maximum(wall_max, p)

wall_center = (wall_min + wall_max) / 2.0
wall_size = wall_max - wall_min  # rough (brick-center-to-brick-center) extent, good enough to frame the shot

################################################################
# Optional: a mortar core per wall, recessed 1cm inside the bricks' true
# faces so it only shows through the small gaps between bricks -- much
# cheaper than modeling every individual joint.
#
# Built from the wall's footprint, not a bounding box, so it follows curved
# or angled walls: the footprint is the union of every brick's top-view
# outline, "closed" (grown then shrunk by joint_close) to bridge the head
# joints between neighbouring bricks, then shrunk by mortar_inset to recess
# it from both faces. That one footprint is extruded straight up from the
# wall's bottom to mortar_inset below its top.
################################################################

if args.mortar:
    import bmesh
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union

    mortar_mat_path = os.path.join(resources_dir, 'mortar_material.blend')
    with bpy.data.libraries.load(mortar_mat_path) as (data_from, data_to):
        data_to.materials = ['Mortar']
    mortar_bpy_mat = bpy.data.materials.get("Mortar")

    mortar_inset = 0.01
    joint_close = 0.02      # must exceed half the widest head joint

    def polygons_of(geom):
        if geom.is_empty:
            return []
        if geom.geom_type == "Polygon":
            return [geom]
        return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]

    def add_prism(bm, footprint, z0, z1):
        """Extrude a shapely (Multi)Polygon, holes included, from z0 to z1 into bm."""
        polys = polygons_of(footprint)
        if not polys or z1 <= z0:
            return
        # A filled 2D curve handles holes (e.g. a closed round wall) without
        # needing a triangulation library.
        cu = bpy.data.curves.new("mortar_footprint", type='CURVE')
        cu.dimensions = '2D'
        cu.fill_mode = 'BOTH'
        cu.extrude = (z1 - z0) / 2.0
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                pts = list(ring.coords)[:-1]
                spline = cu.splines.new('POLY')
                spline.points.add(len(pts) - 1)
                for point, (x, y) in zip(spline.points, pts):
                    point.co = (x, y, 0.0, 1.0)
                spline.use_cyclic_u = True
        tmp = bpy.data.objects.new("mortar_footprint", cu)
        bpy.context.scene.collection.objects.link(tmp)
        me = bpy.data.meshes.new_from_object(tmp.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        me.transform(mathutils.Matrix.Translation((0.0, 0.0, (z0 + z1) / 2.0)))
        bm.from_mesh(me)
        bpy.data.objects.remove(tmp)
        bpy.data.curves.remove(cu)
        bpy.data.meshes.remove(me)

    for wall_coll, brick_corners in wall_brick_corners:
        outline = unary_union([MultiPoint(c[:, :2]).convex_hull for c in brick_corners])
        footprint = (outline.buffer(joint_close, join_style="mitre")
                            .buffer(-(joint_close + mortar_inset), join_style="mitre")
                            .simplify(0.001))
        z0 = min(c[:, 2].min() for c in brick_corners)
        z1 = max(c[:, 2].max() for c in brick_corners) - mortar_inset

        bm = bmesh.new()
        add_prism(bm, footprint, z0, z1)

        mortar_mesh = bpy.data.meshes.new("{}_mortar".format(wall_coll.name))
        bm.to_mesh(mortar_mesh)
        bm.free()
        mortar_obj = bpy.data.objects.new(mortar_mesh.name, mortar_mesh)
        wall_coll.objects.link(mortar_obj)

        mortar = bproc.types.MeshObject(mortar_obj)
        mortar.replace_materials(bproc.types.Material(mortar_bpy_mat))
        mortar.set_cp('category_id', 2)
        mortar.set_cp('category_name', 'Mortar')

################################################################
# Camera framing math (needed up front now: the backdrop below has to be
# placed behind the camera's stand-off distance, not at a fixed spot --
# unlike dataset_from_brick_stock.py's single close-up brick, this hero
# shot's camera distance scales with the whole wall's size, so a
# fixed-position backdrop can end up in FRONT of the camera instead of
# behind the wall, which the camera then stares into the back of).
################################################################

camera_distance = max(wall_size[0], wall_size[2]) * 1.6 + 1.0
camera_height = wall_center[2] + wall_size[2] * 0.15

################################################################
# Backdrop + light (static, no physics)
################################################################

room_span = camera_distance * 2 + max(wall_size[0], wall_size[2])
room = [bproc.object.create_primitive('PLANE', scale=[room_span, room_span, 1],
                                       location=[wall_center[0], wall_center[1], facade_min[2]]),
        bproc.object.create_primitive('PLANE', scale=[room_span, room_span, 1],
                                       location=[wall_center[0], wall_center[1] + camera_distance + 1.0,
                                                 wall_center[2] + wall_size[2]],
                                       rotation=[-1.570796, 0, 0])]

white_mat = bproc.material.create('backdrop_white')
white_mat.set_principled_shader_value("Base Color", [1.0, 1.0, 1.0, 1.0])
white_mat.set_principled_shader_value("Roughness", 0.9)
for plane in room:
    plane.replace_materials(white_mat)
for plane in room:
    plane.set_cp('category_id', 0)
    plane.set_cp("category_name", 'Unknown')

light_plane = bproc.object.create_primitive('PLANE', scale=[4, 4, 1], location=[wall_center[0], wall_center[1], 10])
light_plane.set_name('light_plane')
light_mat = bproc.material.create('light_material')
light_mat.make_emissive(emission_strength=6, emission_color=[0.9, 0.9, 0.9, 1.0])
light_plane.replace_materials(light_mat)

################################################################
# Camera: one or two static "hero" shots framing the whole wall front-on,
# OR (with --save-blend) just one starting pose left for manual adjustment.
################################################################

angles = (0,) if args.save_blend else (0, 25)  # straight-on, then a slight 3/4 angle
for angle_deg in angles:
    angle = np.deg2rad(angle_deg)
    location = [
        wall_center[0] + camera_distance * np.sin(angle),
        wall_center[1] - camera_distance * np.cos(angle),
        camera_height,
    ]
    poi = np.array([wall_center[0], wall_center[1], wall_center[2]])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - np.array(location))
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

# orthographic instead of perspective, scaled to fit the wall's width/height
# (ortho_scale spans the image's wider side -- 4:3 here) plus a 10% margin
cam = bpy.context.scene.camera.data
cam.type = 'ORTHO'
cam.ortho_scale = max(max(wall_size[0], wall_size[1]), wall_size[2] * 4 / 3) * 1.1 + 0.3

bproc.renderer.set_max_amount_of_samples(128)  # a hero shot, not training data -- worth the extra samples

if args.save_blend:
    os.makedirs(os.path.dirname(os.path.abspath(args.save_blend)) or ".", exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
    print("Saved scene (no render) -> {}".format(args.save_blend))
    print("Open it in Blender, position/add a camera as you like, then Render > Render Image.")
    sys.exit(0)

with suppress_stdout():
    data = bproc.renderer.render()

os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
base, ext = os.path.splitext(args.output)
for i, color in enumerate(data["colors"]):
    out_path = args.output if len(data["colors"]) == 1 else "{}_{}{}".format(base, i, ext)
    import cv2
    cv2.imwrite(out_path, cv2.cvtColor(np.asarray(color), cv2.COLOR_RGB2BGR))
    print("Wrote {}".format(out_path))