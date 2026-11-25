"""
Shared brick geometry/damage/material helpers used by both
generate_brick_stock.py (Phase 1: create damaged brick geometries, save as
OBJ) and dataset_from_brick_stock.py (Phase 2: load a stock brick, apply a
fresh random material, render views + grade damage cells).

Deliberately forked out of generate_broken_bricks_v3.py rather than having
that script import from here -- generate_broken_bricks_v3.py stays exactly
as it is (its own working, tested single-shot pipeline). This module only
carries the --fracture damage model (corner/surface/none) and the
material system; --chipping/ChippingModifier (the geometry_script-based
small-scale surface modifier) was not ported over, since nothing recent
has used it and it drags in the whole geometry_script node-tree setup.

Pure bpy -- no blenderproc import here, so this can be used from either a
plain Blender --python run or a blenderproc run.
"""

import contextlib
import math
import os
import random
import sys

import bmesh
import bpy
import mathutils


@contextlib.contextmanager
def suppress_stdout():
    """
    Redirects the process's real stdout (file descriptor 1) to /dev/null
    for the duration of the block. Cycles' per-sample render progress and
    Blender's "Applied modifier was not first" notices are written by
    Blender's C++ code directly to the OS file descriptor, bypassing
    Python's sys.stdout entirely, so redirecting sys.stdout alone would
    not silence them.
    """
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        os.dup2(saved_fd, stdout_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


################################################################
# Mesh I/O
################################################################

def export_mesh_as_obj(obj, filepath):
    """
    Writes obj's evaluated (post-modifier) mesh to a plain OBJ file,
    vertex-for-vertex in the object's own local coordinates -- deliberately
    not using bpy's wm.obj_export operator, which applies a default axis
    remap (historically -Z-forward/Y-up) that would silently rotate the
    exported mesh relative to the object's native local frame. Everything
    that consumes this file (import_mesh_from_obj below, chip_* functions,
    damage cell grading against evaluation.damage_grader.BrickDamageGrader)
    assumes the mesh sits in that same untouched local frame as the
    object's own vertices.

    Uses bpy's wm.obj_export/wm.obj_import with forward_axis='Y',
    up_axis='Z' explicitly pinned on both ends -- that combination is
    Blender's own native axis labeling, so it's a no-op identity mapping
    (verified via a round-trip test: coordinates come back matching to
    within OBJ's text-based float precision, ~1e-8, negligible against the
    3mm grading tolerance). A first version of this function hand-wrote
    "v"/"f" lines directly to sidestep the default axis remap risk, but
    that meant no normals were ever written, so every reimport flat-shaded
    the mesh regardless of how dense/beveled it actually was -- using the
    real exporter/importer (with the axis conversion neutralized) keeps
    the frame safe *and* preserves shading (including the bevel's
    harden_normals custom split normals).
    """
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    with suppress_stdout():
        bpy.ops.wm.obj_export(
            filepath=filepath, export_selected_objects=True,
            forward_axis='Y', up_axis='Z',
            export_normals=True, export_materials=False, export_uv=False,
        )


def import_mesh_from_obj(filepath, name="Brick"):
    """
    Reads an OBJ file written by export_mesh_as_obj() back into a new
    Blender object via bpy's wm.obj_import, with the same forward_axis='Y'
    /up_axis='Z' identity mapping used on export -- see export_mesh_as_obj's
    docstring for why that pairing is safe (no rotation) despite using the
    stock importer.
    """
    bpy.ops.object.select_all(action='DESELECT')
    with suppress_stdout():
        # Pass directory/files explicitly, not just filepath -- wm.obj_import
        # is a file-browser-style operator, and Blender operators persist
        # their last-used property values across repeated calls within the
        # same session. When only filepath is given, directory/files get
        # derived from it internally on the first call, then that derived
        # state can stick and take priority over a freshly-passed filepath
        # on later calls in the same process (harmless for a script that
        # only imports once, but silently re-imports the first file forever
        # in a script that loops multiple imports in one Blender session).
        bpy.ops.wm.obj_import(filepath=filepath,
                               directory=os.path.dirname(filepath),
                               files=[{"name": os.path.basename(filepath)}],
                               forward_axis='Y', up_axis='Z')
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj


################################################################
# Mesh diagnostics
################################################################

def get_volume(obj):
    """Robust volume in m^3 using bmesh (works better on many fracture outputs)."""
    if obj.type != 'MESH':
        return 0.0

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    try:
        vol = bm.calc_volume(signed=False)
    except Exception:
        vol = 0.0

    bm.free()
    eval_obj.to_mesh_clear()
    return float(vol)


def get_evaluated_vertex_count(obj):
    """Vertex count after modifiers (Remesh etc) are evaluated."""
    if obj.type != 'MESH':
        return -1
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    count = len(mesh.vertices)
    eval_obj.to_mesh_clear()
    return count


################################################################
# Base geometry
################################################################

def create_single_brick(name="Brick", size=(0.25, 0.12, 0.065), remesh=True):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    cube_obj = bpy.context.active_object
    cube_obj.name = name
    cube_obj.scale = size[0], size[1], size[2]
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

    if remesh:
        resample_modifier = cube_obj.modifiers.new(name="Remesh", type='REMESH')
        # Was 4 (344 verts baseline) -- too coarse for add_surface_
        # irregularity's Displace modifier to look smoothly bumpy rather
        # than faceted. 5 gives ~4x the vertices (1344) for negligible
        # extra remesh time (~0.009s either way); 6 would be ~16x denser,
        # too risky given how many sequential booleans get chained onto
        # this mesh afterward (20-30+ in a heavily-damaged brick).
        resample_modifier.octree_depth = 5
        resample_modifier.mode = 'SHARP'
        # Apply immediately, as its own step -- if left active and only
        # baked later as a side effect of the first boolean's
        # modifier_apply (cascading apply), the result is a much coarser
        # mesh (confirmed empirically: 98 verts post-damage vs 436 when
        # Remesh is applied up front, for the same brick/seed). Booleans
        # need the genuinely-remeshed dense mesh to cut into, not the
        # 8-vert base cube.
        bpy.context.view_layer.objects.active = cube_obj
        with suppress_stdout():
            bpy.ops.object.modifier_apply(modifier=resample_modifier.name)

    return cube_obj


# The brick's 8 corners as +-1 sign vectors, not absolute positions.
brick_corners = [
    (sx, sy, sz)
    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
]

# The brick's 12 edges as (axis the edge runs along, (sign, sign) of the
# other two axes in ascending order).
# How far chip_edge_with_capsule insets its blob from the two faces
# meeting at the edge, as a fraction of `depth`. Must stay well below
# 1.0: at 1.0 the edge corner is sqrt(2) away in the blob's own
# normalised coordinates and never gets cut at all. 0.5 straddles the
# edge and leaves a groove about `depth` deep in each face.
EDGE_INSET_FRAC = 0.5

brick_edges = [
    (axis, (s1, s2))
    for axis in range(3)
    for s1 in (-1, 1) for s2 in (-1, 1)
]


def long_edges(brick_size):
    """The 4 edges running along the brick's LONGEST axis.

    Edge damage on a real brick concentrates on the long edges -- they
    are what a brick lands on, gets stacked against and dragged over.
    Drawing uniformly from all 12 puts two thirds of the damage on the
    short edges, where a spall along a 65 mm edge is barely
    distinguishable from a corner chip and adds nothing the corner
    mechanism does not already cover.
    """
    axis = list(brick_size).index(max(brick_size))
    return [e for e in brick_edges if e[0] == axis]


def corners_on_face(face_axis, face_sign):
    """The 4 brick_corners lying on the given face (face_axis fixed to face_sign)."""
    return [c for c in brick_corners if c[face_axis] == face_sign]


def corners_on_edge(edge_axis, other_signs):
    """The 2 brick_corners at either end of the given edge."""
    other_axes = [i for i in range(3) if i != edge_axis]
    result = []
    for free_sign in (-1, 1):
        c = [0, 0, 0]
        c[edge_axis] = free_sign
        for axis_i, sign in zip(other_axes, other_signs):
            c[axis_i] = sign
        result.append(tuple(c))
    return result


################################################################
# Boolean chipping primitives
################################################################

def _safe_boolean_apply(obj, bool_mod, min_fraction=0.5):
    """
    Applies bool_mod (already configured, e.g. DIFFERENCE) and bakes it via
    modifier_apply, but rolls back to the pre-operation mesh if this one
    operation dropped the volume by more than min_fraction -- a single
    reasonably-sized chip should never remove that much in one go, so a
    drop that large means the boolean produced a degenerate result.

    bool_mod.show_viewport is temporarily disabled while measuring
    pre_volume -- get_volume() reads the object's *evaluated* mesh (all
    active modifiers included), and bool_mod is already attached to obj by
    the time this function is called, so without disabling it first,
    "pre_volume" would already reflect this boolean's own effect. Normally
    harmless (both pre/post would just be similarly-off), but for a
    catastrophically large cut (observed: an anchor bite radius of 52.88mm
    on a 65mm-tall brick) BOTH pre_volume and post_volume read ~0, so
    `post_volume < min_fraction * pre_volume` becomes `0 < 0` -- false --
    and the rollback silently never fires no matter how degenerate the
    result is. Confirmed fixed empirically: same scenario, pre_volume now
    reads the true pre-cut value and the rollback fires correctly.

    Returns True if the operation was kept, False if it was rolled back.
    """
    backup = obj.data.copy()
    bool_mod.show_viewport = False
    pre_volume = get_volume(obj)
    bool_mod.show_viewport = True
    bpy.context.view_layer.objects.active = obj
    with suppress_stdout():
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    post_volume = get_volume(obj)

    if pre_volume > 0 and post_volume < min_fraction * pre_volume:
        old_data = obj.data
        obj.data = backup
        bpy.data.meshes.remove(old_data)
        return False

    bpy.data.meshes.remove(backup)
    return True


def clean_boolean_debris(obj, merge_distance=1e-5):
    """
    Merge near-duplicate vertices and dissolve degenerate geometry left
    behind by the damage booleans.

    Nothing in this pipeline cleaned up between booleans, and a damaged
    brick goes through 5-15 of them in sequence (3-10 edge lobes, plus
    corner bites and face chips), each EXACT-solver pass leaving its own
    slivers and coincident vertices on top of the last. The symptom is
    not in the boolean result -- which looks fine -- but in the bevel
    afterwards: soften_edges' second pass requests ~5mm, and
    _safe_bevel_apply's explosion guard halves it away to 0.15mm or
    gives up entirely at 0.00mm. A 0.15mm bevel is trivial for clean
    geometry; pass 1 takes 3-5mm on the pristine box without complaint.
    That gap is the tell that the mesh, not the width, is the problem.

    merge_distance is 10um: far below any real feature here (the
    smallest deliberate geometry is a ~3mm chip) and far above the
    float error the booleans introduce.

    Best-effort by design. If the cleanup itself fails the brick is
    still usable -- it just keeps whatever bevel the safety net allows,
    which is the behaviour before this existed.
    """
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        with suppress_stdout():
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=merge_distance)
            bpy.ops.mesh.dissolve_degenerate(threshold=merge_distance)
            bpy.ops.mesh.delete_loose()
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')
    except RuntimeError as e:
        # Never let cleanup abort a brick that damaged successfully.
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        print(f"  cleanup skipped: {e}")


def chip_corner_with_sphere(obj, corner_sign, radius, jitter=0.35):
    """
    Boolean-subtracts a small, slightly irregular sphere positioned at one
    corner of obj's local bounding box, carving a corner-local chip.
    """
    vs = [v.co for v in obj.data.vertices]
    xs = [float(v.x) for v in vs]; ys = [float(v.y) for v in vs]; zs = [float(v.z) for v in vs]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)

    sx, sy, sz = corner_sign
    corner = mathutils.Vector((
        xmax if sx > 0 else xmin,
        ymax if sy > 0 else ymin,
        zmax if sz > 0 else zmin,
    ))
    jitter_offset = mathutils.Vector([random.uniform(-jitter, jitter) * radius for _ in range(3)])
    sphere_center = obj.matrix_world @ (corner + jitter_offset)

    bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, location=sphere_center, subdivisions=2)
    sphere_obj = bpy.context.active_object
    sphere_obj.scale = tuple(1.0 + random.uniform(-jitter, jitter) for _ in range(3))
    sphere_obj.rotation_euler = tuple(random.uniform(0, 3.14159) for _ in range(3))

    bool_mod = obj.modifiers.new(name="CornerChip", type='BOOLEAN')
    bool_mod.object = sphere_obj
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.solver = 'EXACT'

    _safe_boolean_apply(obj, bool_mod)

    bpy.data.objects.remove(sphere_obj, do_unlink=True)
    return obj


def chip_face_surface(obj, face_sign, depth, coverage=0.75, jitter=0.2):
    """
    Boolean-subtracts a large, flattened, slightly irregular blob against
    one whole face of obj's local bounding box -- a broad, shallow surface
    spall, as opposed to chip_corner_with_sphere's small, localized bite.
    """
    vs = [v.co for v in obj.data.vertices]
    xs = [float(v.x) for v in vs]; ys = [float(v.y) for v in vs]; zs = [float(v.z) for v in vs]
    bounds_min = mathutils.Vector((min(xs), min(ys), min(zs)))
    bounds_max = mathutils.Vector((max(xs), max(ys), max(zs)))
    extent = bounds_max - bounds_min
    center = (bounds_min + bounds_max) / 2.0

    axis = [i for i in range(3) if face_sign[i] != 0][0]
    sign = face_sign[axis]

    patch_center = mathutils.Vector(center)
    patch_center[axis] = (bounds_max[axis] if sign > 0 else bounds_min[axis]) - sign * depth
    for i in range(3):
        if i != axis:
            patch_center[i] += random.uniform(-jitter, jitter) * extent[i]

    world_center = obj.matrix_world @ patch_center

    bpy.ops.mesh.primitive_ico_sphere_add(radius=1.0, location=world_center, subdivisions=3)
    patch_obj = bpy.context.active_object
    patch_scale = [0.0, 0.0, 0.0]
    for i in range(3):
        if i == axis:
            patch_scale[i] = depth * (1.0 + random.uniform(-jitter, jitter))
        else:
            patch_scale[i] = (extent[i] * coverage / 2.0) * (1.0 + random.uniform(-jitter, jitter))
    patch_obj.scale = tuple(patch_scale)
    patch_obj.rotation_euler = (
        random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), random.uniform(0, 6.28318)
    )

    bool_mod = obj.modifiers.new(name="SurfaceSpall", type='BOOLEAN')
    bool_mod.object = patch_obj
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.solver = 'EXACT'

    _safe_boolean_apply(obj, bool_mod)

    bpy.data.objects.remove(patch_obj, do_unlink=True)
    return obj


def chip_edge_with_capsule(obj, edge_axis, other_signs, depth, coverage=0.85, jitter=0.15):
    """
    Boolean-subtracts a long, capsule-like blob running along one whole
    edge of obj's local bounding box, leaving a rounded groove -- the
    long-edge spall that corner bites and face chips cannot produce.

    Called by apply_random_damage for damage_type == "edge", restricted
    to the brick's long edges (see long_edges).
    """
    vs = [v.co for v in obj.data.vertices]
    xs = [float(v.x) for v in vs]; ys = [float(v.y) for v in vs]; zs = [float(v.z) for v in vs]
    bounds_min = mathutils.Vector((min(xs), min(ys), min(zs)))
    bounds_max = mathutils.Vector((max(xs), max(ys), max(zs)))
    extent = bounds_max - bounds_min
    center = (bounds_min + bounds_max) / 2.0

    other_axes = [i for i in range(3) if i != edge_axis]

    # Centre the blob ON the edge, inset only slightly -- NOT by the full
    # `depth`, which is what this used to do and which cannot cut an edge
    # at all. With a full inset on both axes the edge corner sits at
    # (1, 1) in the ellipsoid's normalised coordinates, i.e. sqrt(2) =
    # 1.414 away, while the scale jitter only ever reaches 1 + jitter.
    # The corner therefore survives every cut: you get two shallow
    # dimples on the two adjoining faces with the sharp edge still
    # standing between them, and whenever the jitter factor lands below
    # 1.0 the ellipsoid is strictly inside the brick and carves an
    # internal cavity that is invisible from outside. Measured on five
    # test bricks: fragment ratios 0.972-0.994, i.e. almost nothing
    # removed, and only one of five looked damaged at all.
    #
    # At EDGE_INSET_FRAC the blob straddles the edge, a quarter of it
    # lies inside the brick, and the difference leaves a rounded groove
    # of roughly `depth` running along the edge -- an actual spall.
    patch_center = mathutils.Vector(center)
    for axis_i, sign in zip(other_axes, other_signs):
        face = bounds_max[axis_i] if sign > 0 else bounds_min[axis_i]
        patch_center[axis_i] = face - sign * depth * EDGE_INSET_FRAC
    patch_center[edge_axis] += random.uniform(-jitter, jitter) * extent[edge_axis]

    # A CHAIN of overlapping lobes, not one long ellipsoid.
    #
    # One scaled sphere subtracted along the edge reads as exactly what
    # it is: a smooth elliptical scoop, identical on every brick. Real
    # edge spall is a run of overlapping conchoidal fractures -- the
    # groove wanders, its depth varies along the edge, and the rim
    # scallops. Chaining several lobes with independent size, inset and
    # tilt produces that, and it is the same construction apply_bites
    # already uses for crumbling corners.
    span = extent[edge_axis] * coverage
    start = patch_center[edge_axis] - span / 2.0
    # Enough lobes that consecutive ones overlap rather than beading.
    n_lobes = max(3, min(10, int(round(span / max(depth * 2.2, 1e-6)))))
    spacing = span / n_lobes

    for k in range(n_lobes):
        lobe_center = mathutils.Vector(patch_center)
        lobe_center[edge_axis] = (
            start + spacing * (k + 0.5)
            + random.uniform(-0.35, 0.35) * spacing
        )
        # Per-lobe inset and radius, kept inside the band where the edge
        # still gets cut: the corner sits at sqrt(2)*inset/radius in the
        # lobe's normalised coordinates, so the worst pairing here is
        # 1.414*0.55/0.8 = 0.97 -- under 1.0, i.e. every lobe bites.
        # Varying them within that band is what makes the groove uneven
        # instead of a constant-radius fillet.
        inset_frac = random.uniform(0.30, 0.55)
        for axis_i, sign in zip(other_axes, other_signs):
            face = bounds_max[axis_i] if sign > 0 else bounds_min[axis_i]
            lobe_center[axis_i] = face - sign * depth * inset_frac

        # Size CLASS, not just jitter. Lobes all within +-30% of one size
        # read as a machined scallop -- evenly spaced, evenly deep. Real
        # spall is one or two dominant fractures with crumbs around them,
        # so most lobes are drawn small and a minority several times
        # larger.
        mag = (random.uniform(0.45, 1.0) if random.random() < 0.7
               else random.uniform(1.3, 1.9))

        # Sometimes run the bite ONTO one of the two faces the edge
        # joins, rather than keeping every chip in a narrow band hugging
        # the edge. The radius on one of the other two axes is what
        # carries the chip across the adjoining face; enlarging one and
        # not the other is what makes it spall further on one side, as
        # real chips do.
        spread = {a: 1.0 for a in other_axes}
        if random.random() < 0.35:
            spread[random.choice(other_axes)] = random.uniform(1.6, 2.4)

        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=1.0, location=obj.matrix_world @ lobe_center, subdivisions=2
        )
        lobe = bpy.context.active_object
        lobe_scale = [0.0, 0.0, 0.0]
        lobe_scale[edge_axis] = spacing * mag * random.uniform(0.7, 1.3)
        for axis_i in other_axes:
            # Cap mag*spread as a PRODUCT, not just each factor: a big
            # lobe that also spreads onto a face compounded to 9x depth,
            # which at the deep end of the depth range is a 130mm bite
            # out of a 250mm brick. That reads as a broken brick, not a
            # chipped edge -- measured fragment ratios fell to 0.92.
            eff = min(mag * spread[axis_i], 2.6)
            r = depth * eff * random.uniform(0.8, 1.3)
            # And cap against the brick's own thickness on that axis, so
            # the thin 65mm faces cannot be cut most of the way through.
            lobe_scale[axis_i] = min(r, 0.25 * extent[axis_i])
        lobe.scale = tuple(lobe_scale)
        lobe.rotation_euler = tuple(random.uniform(-0.35, 0.35) for _ in range(3))

        bool_mod = obj.modifiers.new(name=f"EdgeChip{k}", type='BOOLEAN')
        bool_mod.object = lobe
        bool_mod.operation = 'DIFFERENCE'
        bool_mod.solver = 'EXACT'
        _safe_boolean_apply(obj, bool_mod)
        bpy.data.objects.remove(lobe, do_unlink=True)

    return obj


def chop_off_big_chunk(obj, axis, sign, keep_fraction, tilt=0.12):
    """
    Slices a large, roughly flat chunk off one whole end of obj -- a
    single full-width cut across most of the cross section, keeping
    `keep_fraction` of the brick's extent along `axis` and removing the
    rest.
    """
    vs = [v.co for v in obj.data.vertices]
    xs = [float(v.x) for v in vs]; ys = [float(v.y) for v in vs]; zs = [float(v.z) for v in vs]
    bounds_min = mathutils.Vector((min(xs), min(ys), min(zs)))
    bounds_max = mathutils.Vector((max(xs), max(ys), max(zs)))
    extent = bounds_max - bounds_min
    center = (bounds_min + bounds_max) / 2.0

    if sign > 0:
        cut_pos = bounds_min[axis] + keep_fraction * extent[axis]
    else:
        cut_pos = bounds_max[axis] - keep_fraction * extent[axis]

    cutter_size = [extent[i] * 1.8 for i in range(3)]
    cutter_size[axis] = extent[axis] * 2.0

    cutter_center = mathutils.Vector(center)
    cutter_center[axis] = cut_pos + sign * (cutter_size[axis] / 2.0)

    world_center = obj.matrix_world @ cutter_center

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=world_center)
    cutter_obj = bpy.context.active_object
    cutter_obj.scale = tuple(cutter_size)

    tilt_rot = [0.0, 0.0, 0.0]
    for i in range(3):
        if i != axis:
            tilt_rot[i] = random.uniform(-tilt, tilt)
    cutter_obj.rotation_euler = tuple(tilt_rot)

    bool_mod = obj.modifiers.new(name="BigChunkCut", type='BOOLEAN')
    bool_mod.object = cutter_obj
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.solver = 'EXACT'

    _safe_boolean_apply(obj, bool_mod)

    bpy.data.objects.remove(cutter_obj, do_unlink=True)
    return obj


################################################################
# Damage model (corner / surface / none) + edge softening
################################################################

class DamageAborted(Exception):
    """Raised by apply_random_damage when a sample lost too much material
    to a degenerate boolean sequence -- the caller should discard this
    brick instance and try again, not abort the whole batch."""


def apply_random_damage(obj, brick_size, full_brick_volume, min_pts_abort_ratio=0.15,
                         weights=(0, 0, 1, 0, 0)): #0.5, 0.1, 0.2, 0.1, 0.1
    """
    Applies the same 3-way random damage model as generate_broken_bricks_v3.py's
    `if fracture:` block: 0-4 severity-clustered corner chips ("corner"),
    a whole end sheared off ("surface"), or no severe damage ("none") --
    always topped off with scattered minor corner nibbles and occasional
    small surface chips regardless of which branch ran.

    Args:
        obj: the brick object to damage (modified in place, mesh baked).
        brick_size: (length, width, height) in meters -- used to scale
            bite/depth sizes relative to the brick's own thinnest
            half-extent, and to pick which axis is "long" for the
            "surface" damage type.
        full_brick_volume: get_volume() of the pristine brick, for the
            abort-on-too-much-damage check.
        min_pts_abort_ratio: abort (raise DamageAborted) if the post-
            damage volume drops below this fraction of full_brick_volume.
        weights: 5 selection weights, in the order
            (corner, surface, edge, corner+edge, none). "corner+edge"
            applies a corner cluster and THEN a long-edge spall on the
            same brick -- the single-type choice above otherwise makes
            multi-region damage impossible, while 6 of the 10 real
            bricks carry damage on more than one face.

    Returns:
        damage_type: "corner", "surface", or "none".

    Raises:
        DamageAborted: if the damage sequence left the brick with too
            little material (a degenerate boolean result), so the caller
            should discard this sample rather than keep it.
    """
    min_half_extent = min(brick_size) / 2.0
    damage_type = random.choices(
        ["corner", "surface", "edge", "corner+edge", "none"], weights=list(weights))[0]

    def apply_bites(corner_sign, radius_fraction_range, bite_range, label, verbose=True):
        num_bites = random.randint(*bite_range)
        if verbose:
            print(f"  corner {corner_sign} [{label}]: {num_bites} bites, "
                  f"radius {radius_fraction_range[0]:.1f}-{radius_fraction_range[1]:.1f}x half-extent")
        for _ in range(num_bites):
            radius = random.uniform(*radius_fraction_range) * min_half_extent
            chip_corner_with_sphere(obj, corner_sign, radius=radius, jitter=0.35)
        return num_bites

    def apply_severe_corner(corner_sign):
        anchor_radius = random.uniform(1.0, 1.8) * min_half_extent
        chip_corner_with_sphere(obj, corner_sign, radius=anchor_radius, jitter=0.35)
        crumb_range = (0.3, 0.9)
        num_crumb = apply_bites(corner_sign, crumb_range, (3, 8), "severe/crumbling", verbose=False)
        print(f"  corner {corner_sign} [severe]: 1 anchor bite + {num_crumb} crumbling bites "
              f"(radius {crumb_range[0]:.1f}-{crumb_range[1]:.1f}x half-extent)")

    def do_corner_damage(severity_choices=(0, 1, 2, 3, 4),
                         severity_weights=(0.10, 0.20, 0.30, 0.10, 0.30)):
        """Severe corner cluster. Returns the corners it consumed."""
        num_severe = random.choices(list(severity_choices),
                                    weights=list(severity_weights))[0]
        if num_severe == 0:
            chosen = []
        elif num_severe == 1:
            chosen = random.sample(brick_corners, 1)
        elif num_severe == 2:
            edge_axis, other_signs = random.choice(brick_edges)
            chosen = corners_on_edge(edge_axis, other_signs)
        else:
            face_axis = random.choice([0, 1, 2])
            face_sign = random.choice([-1, 1])
            face_corners = corners_on_face(face_axis, face_sign)
            chosen = face_corners if num_severe == 4 else random.sample(face_corners, 3)

        print(f"Fracture: {num_severe} severe corner(s)")
        for corner_sign in chosen:
            apply_severe_corner(corner_sign)
        return chosen

    def do_edge_damage():
        """Long-edge spall. Returns the corners at that edge's ends."""
        edge_axis, other_signs = random.choice(long_edges(brick_size))
        print(f"Fracture: edge cut (axis={edge_axis} signs={other_signs})")
        depth = random.uniform(0.1, 0.45) * min_half_extent
        coverage = random.uniform(0.5, 0.9)
        chip_edge_with_capsule(obj, edge_axis, other_signs,
                               depth=depth, coverage=coverage, jitter=0.1)
        return corners_on_edge(edge_axis, other_signs)

    severe_corners = []
    if damage_type == "corner":
        severe_corners = do_corner_damage()

    elif damage_type == "corner+edge":
        # Both, corner first so the edge spall cuts across geometry the
        # corner bites already opened up -- the order a real brick wears
        # in, and the order that leaves the spall reading as the later,
        # shallower event rather than being erased by the corner cut.
        #
        # Fewer severe corners than the plain "corner" type (1-2, not
        # 0-4): a full corner cluster plus a full-length edge spall
        # removes enough material to start tripping
        # min_pts_abort_ratio, and the point of this type is a brick
        # carrying two DISTINCT damage regions, not a destroyed one.
        # Measured on the real set, 6 of 10 bricks have damage on more
        # than one face, which is what this type exists to represent.
        severe_corners = do_corner_damage(severity_choices=(1, 2),
                                          severity_weights=(0.6, 0.4))
        severe_corners = list({tuple(c) for c in severe_corners}
                              | {tuple(c) for c in do_edge_damage()})

    elif damage_type == "surface":
        axis = brick_size.index(max(brick_size))
        sign = random.choice([-1, 1])
        keep_fraction = random.uniform(0.8, 0.9)
        print(f"Fracture: surface cut off (axis={axis} sign={sign:+d}, "
              f"removing ~{100 * (1 - keep_fraction):.0f}% of the brick)")
        chop_off_big_chunk(obj, axis, sign, keep_fraction)
        severe_corners = corners_on_face(axis, sign)

    elif damage_type == "edge":
        severe_corners = do_edge_damage()

    elif damage_type == "none":
        print("Fracture: none (no severe corner or surface damage)")
        severe_corners = []

    for corner_sign in brick_corners:
        if corner_sign in severe_corners:
            continue
        if random.random() < 0.4:
            apply_bites(corner_sign, (0.15, 0.5), (1, 3), "minor")

    if random.random() < 0.4:
        for _ in range(random.randint(1, 2)):
            face_axis = random.choice([0, 1, 2])
            face_sign = [0, 0, 0]
            face_sign[face_axis] = random.choice([-1, 1])
            depth = random.uniform(0.1, 0.25) * min_half_extent
            coverage = random.uniform(0.15, 0.35)
            print(f"  surface chip: face_axis={face_axis} sign={face_sign[face_axis]:+d}")
            chip_face_surface(obj, tuple(face_sign), depth=depth, coverage=coverage, jitter=0.3)

    post_damage_volume = get_volume(obj)
    if post_damage_volume < min_pts_abort_ratio * full_brick_volume:
        raise DamageAborted(
            f"damage type '{damage_type}' removed too much material "
            f"(fragment ratio {post_damage_volume / full_brick_volume:.3f})"
        )

    fragment_ratio = post_damage_volume / full_brick_volume
    print("Post fracture fragment ratio: {:.3f}".format(fragment_ratio))
    return damage_type


def _mesh_bounds(obj):
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    return (
        mathutils.Vector((min(xs), min(ys), min(zs))),
        mathutils.Vector((max(xs), max(ys), max(zs))),
    )


def _safe_bevel_apply(obj, width, segments, harden_normals=True, max_attempts=6):
    """
    Applies a Bevel modifier with use_clamp_overlap=False -- required
    because Blender's default (use_clamp_overlap=True) doesn't just shrink
    the bevel locally wherever it would self-intersect, it was observed
    (via nearest-vertex-to-ideal-corner distance) to collapse the width to
    near-zero *everywhere on the mesh* as soon as ANY region -- e.g. a
    small crumbling-bite cut -- would otherwise overlap. But without that
    clamp, a width too large for one of those tight, densely-damaged spots
    doesn't shrink gracefully either -- it makes the bevel blow up into
    self-intersecting spikes there instead (observed directly).

    A bevel can only ever cut material away; it should never make the mesh
    extend past its own pre-bevel bounding box. That's a cheap, reliable
    explosion signal -- checked here, and if tripped, rolled back and
    retried with a smaller width (halved each attempt) until it's safe.

    Returns the width that was actually kept (may be smaller than
    requested), or 0.0 if even the smallest attempt still exploded (the
    mesh is left unbeveled in that case).
    """
    pre_min, pre_max = _mesh_bounds(obj)
    tol = 1e-4  # 0.1mm slack for floating point noise

    current_width = width
    for _ in range(max_attempts):
        backup = obj.data.copy()
        bevel_mod = obj.modifiers.new(name="EdgeSoften", type='BEVEL')
        bevel_mod.width = current_width
        bevel_mod.segments = segments
        bevel_mod.harden_normals = harden_normals
        bevel_mod.use_clamp_overlap = False
        bpy.context.view_layer.objects.active = obj
        with suppress_stdout():
            bpy.ops.object.modifier_apply(modifier=bevel_mod.name)

        post_min, post_max = _mesh_bounds(obj)
        exploded = any(
            post_min[i] < pre_min[i] - tol or post_max[i] > pre_max[i] + tol
            for i in range(3)
        )
        if not exploded:
            bpy.data.meshes.remove(backup)
            return current_width

        old_data = obj.data
        obj.data = backup
        bpy.data.meshes.remove(old_data)
        current_width *= 0.5

    return 0.0


def soften_edges(obj, brick_size, width_fraction_range=(0.12, 0.18)):
    """
    Rounds every edge currently on obj. harden_normals=True helps shading,
    but at only 4-6 segments it just fakes smoothness under gentle light
    -- real render lighting still shows visible facets/sharp shadow lines
    at that segment count (confirmed visually). More segments actually
    gives a rounded fillet profile instead of relying on normal faking to
    hide a coarse one. See _safe_bevel_apply for why the actual
    modifier_apply is wrapped in a retry loop instead of a single direct
    call.

    Meant to be called TWICE in the damage pipeline, with different
    width_fraction_range each time -- once on the pristine box (before any
    damage), where a generous width is safe since the 12 box edges are far
    apart with no competing geometry, and once again after damage, with
    the (smaller, default) range here. A single post-damage-only pass was
    tried first and didn't work: damage (even "minor" bites, especially
    where several overlap near the same corner) can leave tightly-packed,
    tiny-scale geometry, so _safe_bevel_apply's retry-and-halve safety net
    kept crushing the width down to near 0mm for the *entire* bevel
    operation on most heavily-damaged bricks (observed: 7/10 in one
    batch) -- including the plain box edges nowhere near any damage, which
    should never have been affected. Narrowing the requested width alone
    didn't fix it either (still 10/10 unsafe at ~4-5mm in testing): the
    problem isn't that the width is too big in absolute terms, it's that
    damage geometry can be arbitrarily small, so *any* single-pass width
    has a real chance of colliding with it somewhere. Isolating the box's
    own edges into their own pass, before that risk exists at all,
    sidesteps the problem instead of trying to out-tune it.
    """
    min_half_extent = min(brick_size) / 2.0
    width = random.uniform(*width_fraction_range) * min_half_extent
    segments = random.randint(10, 14)
    kept_width = _safe_bevel_apply(obj, width, segments)
    if kept_width < width:
        print(f"  bevel: requested width {width*1000:.2f}mm was unsafe, "
              f"used {kept_width*1000:.2f}mm instead")
    else:
        # Printed on success too: a silent pass is indistinguishable from
        # one that was never reached, and the whole question here is
        # whether the post-damage pass keeps a usable width.
        print(f"  bevel: kept full width {kept_width*1000:.2f}mm")


def add_surface_irregularity(obj, brick_size):
    """
    Perturbs every vertex a small amount along its own normal, driven by a
    procedural noise texture -- gives flat, undamaged surfaces/edges an
    organic, weathered irregularity instead of the perfectly regular look
    Remesh+Bevel alone produce. Meant to be called last, after damage and
    bevel, so it reaches every surface including the ones no boolean chip
    ever touched.

    A simpler, more robust replacement for the old ChippingModifier
    geometry-nodes system (see generate_broken_bricks_v3.py): that one
    needed a fragile 10x "working scale" copy of the brick just to make
    its internal mesh_to_volume/volume_to_mesh voxel remesh behave at
    Reichsformat's real-world scale, and the geometry_script DSL it was
    built on caused repeated bugs elsewhere this session (silently-wrong
    .vector/.value suffixes, wrong map_range socket names). A Displace
    modifier does the same "nudge the surface around" job directly, with
    no scale tricks and no node-tree layer in between -- and since it only
    moves existing vertices by a bounded amount (not a boolean/volume
    operation), it can't self-intersect or explode the way booleans/bevel
    could, so no safety net is needed here.
    """
    min_half_extent = min(brick_size) / 2.0

    tex = bpy.data.textures.new(name="SurfaceIrregularity", type='CLOUDS')
    # Was 0.15-0.3 -- too fine, read as almost nothing next to a reference
    # photo of clearly dimpled/rounded brick tops. Larger noise_scale here
    # means bigger, more defined bump features rather than tiny grain.
    tex.noise_scale = random.uniform(0.35, 0.55) * min_half_extent
    tex.noise_depth = random.randint(2, 4)
    tex.noise_basis = random.choice(['BLENDER_ORIGINAL', 'IMPROVED_PERLIN', 'VORONOI_F1'])

    disp_mod = obj.modifiers.new(name="SurfaceIrregularity", type='DISPLACE')
    disp_mod.texture = tex
    disp_mod.texture_coords = 'LOCAL'
    disp_mod.mid_level = 0.5  # so noise pushes both outward and inward, not just outward
    # Was 0.01-0.03 -- barely visible. Pushed up substantially to match a
    # reference photo showing clearly rounded, well-defined dimpling
    # across whole faces, not subtle grain.
    disp_mod.strength = random.uniform(0.08, 0.25) * min_half_extent

    bpy.context.view_layer.objects.active = obj
    with suppress_stdout():
        bpy.ops.object.modifier_apply(modifier=disp_mod.name)

    bpy.data.textures.remove(tex)


################################################################
# Material functions (verbatim from generate_broken_bricks_v3.py)
################################################################

def force_top_or_all(mat_name="Brick", mix_node_name="Mix.007", fac=0.0):
    """
    fac=0.0 -> TopMask (A)
    fac=1.0 -> AllAroundMask (B)
    """
    mat = bpy.data.materials[mat_name]
    nt = mat.node_tree
    node = nt.nodes[mix_node_name]
    fac_socket = node.inputs["Factor"]
    for link in list(fac_socket.links):
        nt.links.remove(link)
    fac_socket.default_value = float(fac)


def set_mortar_coverage(mat_name="Brick", mix_node_name="Color Ramp", black_pos=0.3, white_pos=0.8):
    """black_pos < white_pos, both in [0, 1]"""
    mat = bpy.data.materials[mat_name]
    ramp = mat.node_tree.nodes[mix_node_name].color_ramp
    ramp.elements[0].position = float(black_pos)
    ramp.elements[1].position = float(white_pos)


def randomize_mortar_coverage_weighted():
    r = random.random()
    
    if r < 0.10:
        set_mortar_coverage(
            black_pos=random.uniform(0.75, 0.9),
            white_pos=random.uniform(0.9, 0.98),
        )
    elif r < 0.50:
        black = random.uniform(0.5, 0.7)
        width = random.uniform(0.05, 0.15)
        set_mortar_coverage(black_pos=black, white_pos=min(black + width, 0.98))
    elif r < 0.95:
        black = random.uniform(0.25, 0.45)
        width = random.uniform(0.15, 0.3)
        set_mortar_coverage(black_pos=black, white_pos=min(black + width, 0.98))
    else:
        # Heavy coverage (rare) -- kept wide enough that it still leaves
        # visible bare patches rather than reading as fully white.
        black = random.uniform(0.15, 0.3)
        width = random.uniform(0.2, 0.35)
        set_mortar_coverage(black_pos=black, white_pos=min(black + width, 0.98))


def set_top_mortar_coverage(mat_name="Brick", mix_node_name="Mix.003", fac=1.0):
    mat = bpy.data.materials[mat_name]
    nt = mat.node_tree
    node = nt.nodes[mix_node_name]
    fac_socket = node.inputs["Factor"]
    for link in list(fac_socket.links):
        nt.links.remove(link)
    fac_socket.default_value = float(fac)


def randomize_top_mortar_coverage():
    set_top_mortar_coverage(fac=random.uniform(0.25, 0.55))


def randomize_mortar_pattern_shape(mat_name="Brick"):
    """
    AllAroundMask chain is Noise Texture.004 -> Gradient Texture (LINEAR)
    -> Color Ramp. Rewiring the noise straight into the Color Ramp keeps
    its organic blob shape (bypassing the linear-gradient smoothing), and
    Distortion is the main lever to break those blobs into more irregular,
    mortar-like patches.
    """
    mat = bpy.data.materials[mat_name]
    nt = mat.node_tree
    noise = nt.nodes["Noise Texture.004"]
    ramp = nt.nodes["Color Ramp"]

    nt.links.new(noise.outputs["Color"], ramp.inputs["Fac"])

    noise.inputs["Scale"].default_value = random.uniform(15.0, 35.0)
    noise.inputs["Detail"].default_value = random.uniform(6.0, 16.0)
    noise.inputs["Roughness"].default_value = random.uniform(0.35, 0.6)
    noise.inputs["Distortion"].default_value = random.uniform(0.6, 1.8)


def randomize_top_vs_all_blend(mat_name="Brick", mix_node_name="Mix.007"):
    """
    Combine the top-biased mask with the patchy all-around mask via a
    LIGHTEN blend (component-wise max) so the patchy mask always bleeds
    through everywhere (including the sides) instead of a hard either/or.
    """
    mat = bpy.data.materials[mat_name]
    node = mat.node_tree.nodes[mix_node_name]
    node.blend_type = 'LIGHTEN'
    fac = random.uniform(0.6, 1.0)
    force_top_or_all(mat_name=mat_name, mix_node_name=mix_node_name, fac=fac)


def randomize_top_striations(mat_name="Brick"):
    """
    Wave Texture (wave_type=BANDS) feeds Bump.002 for linear striations,
    like wire-cut/extrusion marks on a real brick's top face. Tames
    distortion, boosts strength, and multiplies the wave's height by the
    top-face mask signal so the lines mainly show up on top.
    """
    mat = bpy.data.materials[mat_name]
    nt = mat.node_tree
    wave = nt.nodes["Wave Texture"]
    bump = nt.nodes["Bump.002"]
    top_mask = nt.nodes["Bright/Contrast"]

    wave.inputs["Distortion"].default_value = random.uniform(0.5, 1.5)
    wave.inputs["Scale"].default_value = random.uniform(10.0, 20.0)
    bump.inputs["Strength"].default_value = random.uniform(0.3, 0.6)

    mask_mult = nt.nodes.get("StriationTopMask")
    if mask_mult is None:
        mask_mult = nt.nodes.new("ShaderNodeMath")
        mask_mult.name = "StriationTopMask"
        mask_mult.operation = 'MULTIPLY'
        nt.links.new(wave.outputs["Color"], mask_mult.inputs[0])
        nt.links.new(top_mask.outputs["Color"], mask_mult.inputs[1])
        nt.links.new(mask_mult.outputs["Value"], bump.inputs["Height"])


def hex_to_srgb01(h: str):
    h = h.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6 hex digits, got: {h}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def rgb_linear_to_hex(rgb_lin):
    r = max(0.0, min(1.0, linear_to_srgb(rgb_lin[0])))
    g = max(0.0, min(1.0, linear_to_srgb(rgb_lin[1])))
    b = max(0.0, min(1.0, linear_to_srgb(rgb_lin[2])))
    return f"{int(round(r*255)):02X}{int(round(g*255)):02X}{int(round(b*255)):02X}"


def srgb_hex_to_linear_rgba(hex_str: str, a: float = 1.0):
    r, g, b = hex_to_srgb01(hex_str)
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), float(a))


def hex_to_linear_rgba(hex_str: str, alpha: float = 1.0):
    h = hex_str.strip().lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0

    def _srgb_to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b), float(alpha))


def list_colorramps(mat_name="Brick"):
    mat = bpy.data.materials.get(mat_name)
    if not mat or not mat.use_nodes:
        raise RuntimeError(f"Material '{mat_name}' not found or not node-based")

    nt = mat.node_tree
    ramps = [n for n in nt.nodes if n.type == "VALTORGB"]
    print(f"Found {len(ramps)} ColorRamp nodes in '{mat_name}':\n")
    for n in ramps:
        ramp = n.color_ramp
        print(f"NODE: {n.name}  (elements={len(ramp.elements)})")
        for i, e in enumerate(ramp.elements):
            rgb_lin = (e.color[0], e.color[1], e.color[2])
            print(f"  [{i}] pos={e.position:.3f}  linear={tuple(round(x,4) for x in rgb_lin)}  approx_hex={rgb_linear_to_hex(rgb_lin)}")
        print()


def randomize_brick_color(mat_name="Brick", ramp_node_name="ColorRamp", element_index=1):
    palette = ["D4A473", "9A5838", "C06D45", "A37A64", "ADA192", "AAA499", "977B6F"]

    mat = bpy.data.materials.get(mat_name)
    if not mat or not mat.use_nodes:
        raise RuntimeError(f"Material '{mat_name}' not found or not node-based")

    node = mat.node_tree.nodes.get(ramp_node_name)
    if not node or node.type != "VALTORGB":
        raise RuntimeError(f"ColorRamp node '{ramp_node_name}' not found in material '{mat_name}'")

    ramp = node.color_ramp
    if element_index >= len(ramp.elements):
        raise RuntimeError(f"Ramp '{ramp_node_name}' has only {len(ramp.elements)} elements")

    e = ramp.elements[element_index]
    chosen = random.choice(palette)
    e.color = hex_to_linear_rgba(chosen, alpha=e.color[3])
    return chosen


def assign_random_brick_texture(obj):
    """
    Randomizes the shared "Brick" material's node values, then copies the
    freshly-randomized material and assigns the independent copy to obj --
    the copy must happen AFTER randomizing, not before, or obj ends up
    wearing the un-randomized, file-saved defaults (mat.copy() only
    snapshots whatever state "Brick" is in at copy time).
    """
    base_mat = bpy.data.materials.get("Brick")
    if not base_mat:
        print("Material 'Brick' not found.")
        return

    randomize_mortar_pattern_shape()
    randomize_top_mortar_coverage()
    randomize_top_vs_all_blend()
    randomize_top_striations()
    randomize_mortar_coverage_weighted()
    randomize_brick_color()

    mat_copy = base_mat.copy()
    obj.data = obj.data.copy()
    obj.active_material = mat_copy
