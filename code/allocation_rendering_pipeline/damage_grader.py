"""
No-training, first-pass damage grading for posed bricks.

Splits the CAD model's bounding box into a 16-cell anisotropic grid --
4 bins along the brick's longest axis, 2 bins along each of the other two
(2x2 across the short end faces, 4x2 along the long side faces) -- and
grades each damage cell 0-1 for how much material appears to be missing
there, by comparing observed (backprojected) surface points against the
intact CAD mesh's signed distance field. Finer along the length than a
uniform 2x2x2 split would be, since that's where a chip's position along
the brick matters most for downstream design decisions.

Zero dependency on torch/detectron2/ROS: takes plain numpy arrays so it can be
unit tested standalone against saved depth/mask/pose .npy files.
"""

import cv2
import numpy as np
import trimesh


def to_canonical(depth, mask, intr, pose, erode_px=0):
    """
    Backproject masked depth pixels to 3D camera-frame points and transform
    them into the CAD model's canonical (object) frame.

    Args:
        depth: (H, W) depth map in meters.
        mask: (H, W) array; pixels where mask > 0 (and depth > 0) are used.
        intr: (3, 3) camera intrinsic matrix.
        pose: (4, 4) object->camera transform, i.e. cam_pt = R @ obj_pt + t
            with R = pose[:3, :3], t = pose[:3, 3] (same convention as
            evaluation.evaluation.transform_points_withRt).
        erode_px: shrink the mask by this many pixels before backprojecting.
            Depth sensors produce unreliable "flying pixel" readings right at
            silhouette edges (interpolated between the object and whatever is
            behind it), which land inside the CAD envelope and look like
            missing material even on an intact brick. Eroding the mask drops
            that boundary ring before it ever reaches grading. 0 disables it.

    Returns:
        (N, 3) float64 array of points in the CAD model's canonical frame.
    """
    mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask_bin = cv2.erode(mask_bin, kernel)

    valid = np.logical_and(mask_bin > 0, depth > 0)
    vs, us = np.where(valid)
    if len(us) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    z = depth[vs, us].astype(np.float64)
    uv1 = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)  # (3, N)
    rays = np.linalg.inv(intr) @ uv1  # (3, N)
    cam_pts = (rays * z[None, :] / rays[2:3, :]).T  # (N, 3)

    R = pose[:3, :3]
    t = pose[:3, 3]
    return (cam_pts - t) @ R


def to_canonical_indexed(depth, mask, intr, pose, erode_px=0):
    """
    Sibling of to_canonical() that additionally returns each returned
    point's source pixel (row, col), instead of flattening that away --
    needed to scatter a per-point result (e.g. a damage classification)
    back into the original (H, W) image shape. Kept as a separate
    function rather than changing to_canonical()'s return signature,
    since many existing callers (grade()/grade_signed()/visible_counts()
    via BrickDamageGrader) already trust its current (N, 3)-only shape.
    Duplicates to_canonical()'s backprojection math exactly -- see its
    docstring for the argument/convention details.

    Returns:
        (pts, rows, cols): pts is (N, 3) canonical-frame points; rows,
        cols are (N,) int arrays of each point's source pixel, i.e.
        pts[i] came from depth[rows[i], cols[i]].
    """
    mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
    if erode_px > 0:
        kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        mask_bin = cv2.erode(mask_bin, kernel)

    valid = np.logical_and(mask_bin > 0, depth > 0)
    vs, us = np.where(valid)
    if len(us) == 0:
        return np.zeros((0, 3), dtype=np.float64), vs, us

    z = depth[vs, us].astype(np.float64)
    uv1 = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)  # (3, N)
    rays = np.linalg.inv(intr) @ uv1  # (3, N)
    cam_pts = (rays * z[None, :] / rays[2:3, :]).T  # (N, 3)

    R = pose[:3, :3]
    t = pose[:3, 3]
    pts = (cam_pts - t) @ R
    return pts, vs, us


def damage_pixel_label(sdist_img, surface_tol=0.003):
    """
    Binary per-pixel damage label from a signed-distance image (see
    BrickDamageGrader.signed_distance_image()) -- for supervising a
    segmentation head directly on pixels, instead of per-cell regression.

    A pixel counts as damaged if the observed surface deviates from the
    intact reference by more than surface_tol in *either* direction --
    both missing material (receded past the intact surface) and extra
    material (e.g. mortar residue on an old brick, or a raised patch)
    read out as damage here; severity/direction is meant to be recovered
    geometrically afterward from the signed distance itself, not learned,
    so the segmentation target only needs to say where, not how or which
    way. Same surface_tol dead zone as grade()/grade_signed(), so a
    pixel graded "damaged" here is consistent with what those would call
    nonzero.

    Args:
        sdist_img: (H, W) float array from signed_distance_image(), NaN
            where unobserved.
        surface_tol: meters; see grade().

    Returns:
        (H, W) int8 array: -1 = unobserved (exclude from loss), 0 =
        matches intact reference, 1 = damaged.
    """
    label = np.full(sdist_img.shape, -1, dtype=np.int8)
    observed = ~np.isnan(sdist_img)
    label[observed] = (np.abs(sdist_img[observed]) > surface_tol).astype(np.int8)
    return label


def damage_pixel_label_with_silhouette(sdist_img, silhouette_gap_mask, surface_tol=0.003):
    """
    Extends damage_pixel_label() with the silhouette-gap signal: pixels
    where material is missing entirely -- background visible through a
    chip/break, e.g. a broken-off corner -- also count as damaged, not
    just pixels on the remaining surface that deviate from the intact
    reference. Without this, a break severe enough to change the
    silhouette contributes zero labeled pixels (there's no depth point on
    the *damaged* brick there at all for signed_distance_image() to see),
    so exactly the most visually obvious kind of damage would otherwise
    go completely unlabeled while subtle on-surface pitting is caught.

    Args:
        sdist_img: (H, W) from signed_distance_image().
        silhouette_gap_mask: (H, W) bool, True where the intact mesh's
            silhouette says brick material should be but the observed
            (damaged) mask_visib says it isn't -- see
            BrickDamageGrader.intact_silhouette_mask(). Compute this as
            intact_silhouette_mask(...) & ~mask_visib, using the *raw*
            (non-eroded) mask_visib -- not signed_distance_image()'s own
            eroded backprojection mask, which would falsely read its own
            noise-avoidance erosion ring as a silhouette gap around every
            brick, damaged or not.
        surface_tol: see damage_pixel_label().

    Returns:
        (H, W) int8: -1 = unobserved (neither on the remaining surface
        nor a silhouette gap), 0 = matches intact, 1 = damaged (surface
        deviation, silhouette gap, or both).

    NOT the recommended way to catch missing-material pixels on this
    dataset -- checked against every one of the 15 views of the single
    most severely damaged brick in the dataset (max cell score 1.0), and
    the gap this measures never rose above the same few-hundred-pixel,
    many-tiny-scattered-components noise floor seen on an undamaged
    brick (rasterization mismatch between the convex-hull projection and
    mask_visib's own rasterization, present on every frame regardless of
    damage). The reason: this dataset's damage rarely removes material
    at the mesh's own true outer envelope, so the convex hull's
    silhouette barely changes even for severe damage. See
    damage_pixel_label_with_out_of_bounds() instead -- it catches the
    same "hole with something else showing through" case via a much more
    direct signal (confirmed present: 15% of one real damaged frame's
    mask_visib points landed up to 95mm outside the mesh's own box).
    Left here since it's still correct for a genuinely convex mesh and
    may matter for real (not synthetic) captures, or a different damage
    style.
    """
    label = damage_pixel_label(sdist_img, surface_tol=surface_tol)
    label[silhouette_gap_mask] = 1
    return label


def damage_pixel_label_with_out_of_bounds(sdist_img, out_of_bounds_mask, surface_tol=0.003):
    """
    Extends damage_pixel_label() with the out-of-bounds signal: pixels
    that mask_visib claims are "the brick" but that backproject nowhere
    near the intact mesh's actual bounding box also count as damaged --
    not just pixels close enough to the mesh to be scored by signed
    distance. This is the validated fix for the same problem
    damage_pixel_label_with_silhouette() was meant to solve (a break
    severe enough to expose whatever's behind the brick contributing zero
    labeled pixels): signed_distance_image()'s bounds_margin filter
    silently drops such points as "unobserved" (correctly -- signed
    distance isn't meaningful that far from the mesh), but for labeling
    purposes that's exactly backwards. A mask_visib pixel that far from
    where the brick's own geometry could plausibly be IS the evidence
    that real material is missing there (floor or other background
    showing through a hole, or a mesh-generation artifact left behind by
    however the break was cut) -- discarding it throws away the clearest
    signal available instead of using it.

    Args:
        sdist_img: (H, W) from signed_distance_image(). Must have been
            computed with the same bounds_margin that produced
            out_of_bounds_mask, so the two partition mask_visib's pixels
            cleanly (in-bounds-and-scored vs. out-of-bounds) with no gap
            or double-count between them.
        out_of_bounds_mask: (H, W) bool from
            BrickDamageGrader.out_of_bounds_mask().
        surface_tol: see damage_pixel_label().

    Returns:
        (H, W) int8: -1 = unobserved (no valid depth/mask at all there),
        0 = matches intact, 1 = damaged (surface deviation, out-of-bounds,
        or both).
    """
    label = damage_pixel_label(sdist_img, surface_tol=surface_tol)
    label[out_of_bounds_mask] = 1
    return label


def damage_pixel_label_full(sdist_img, out_of_bounds_mask, penetration_img,
                            surface_tol=0.003, penetration_tol=0.005):
    """
    The complete per-pixel damage label, combining all three evidence
    sources. Supersedes damage_pixel_label_with_out_of_bounds() after a
    real missed-damage case showed why the first two signals alone are
    not enough:

      1. on-surface deviation (sdist_img beyond surface_tol either way)
         -- scratches, pitting, shallow scoops, residue.
      2. out-of-bounds (mask_visib pixels backprojecting far outside the
         mesh's own bounding box) -- background/floor visible through a
         break, landing nowhere near the brick.
      3. deep penetration (penetration_img beyond penetration_tol) --
         the case both of the above are BLIND to, found on a real frame:
         a corner broken off down to the brick's base exposes a surface
         at the same height as the intact mesh's own BOTTOM face.
         Signed distance is direction-blind -- a point lying on the
         bottom-face plane measures ~0mm from the intact mesh (its own
         bottom!) and reads "matches intact", even though the camera is
         seeing it from ABOVE through tens of mm of brick that should
         be in the way, which is impossible on an intact brick. It is
         also within the bounding box, so out_of_bounds_mask can't
         catch it either. The penetration signal asks the question the
         other two don't: could this surface even be VISIBLE from this
         camera on an intact brick? See
         BrickDamageGrader.penetration_depth_image().

    Args:
        sdist_img: (H, W) from signed_distance_image().
        out_of_bounds_mask: (H, W) bool from out_of_bounds_mask().
        penetration_img: (H, W) from penetration_depth_image() -- must
            share erode_px with the other two so they cover the same
            pixels.
        surface_tol: see damage_pixel_label().
        penetration_tol: meters of perpendicular-equivalent penetration
            past the intact entry surface before a pixel counts as
            damaged. Default 5mm -- deliberately looser than
            surface_tol's 3mm, since this signal targets deep missing
            material (real cases are tens of mm) and along-ray depth
            comparisons near grazing incidence are noisier than the
            perpendicular signed distance; shallow damage is still
            caught by the sdist path.

    Returns:
        (H, W) int8: -1 = unobserved, 0 = matches intact, 1 = damaged
        (any of the three sources).
    """
    label = damage_pixel_label(sdist_img, surface_tol=surface_tol)
    label[out_of_bounds_mask] = 1
    deep = ~np.isnan(penetration_img) & (penetration_img > penetration_tol)
    label[deep] = 1
    return label


def damage_pixel_label_classes(sdist_img, out_of_bounds_mask, penetration_img,
                               surface_tol=0.003, penetration_tol=0.005):
    """
    Multi-class sibling of damage_pixel_label_full(): same three evidence
    sources and thresholds, but the two damage kinds stay distinguishable
    instead of collapsing to one value -- store THIS, and map {1, 2} ->
    damaged when training a binary head. The two classes have genuinely
    different visual signatures (subtle on-surface texture change vs.
    background/floor showing where brick should be), so keeping them
    separate in the stored label preserves the option of a 3-class head
    or per-class loss weighting later without relabeling the dataset;
    collapsing early would throw that away for zero savings.

    Precedence: a pixel that trips both checks is classed as missing
    material (2) -- deep penetration/out-of-bounds is the stronger, less
    ambiguous evidence, and "surface damage" is meant to read as
    "deviates but the surface itself is still the brick's".

    Args: see damage_pixel_label_full().

    Returns:
        (H, W) int8: -1 = unobserved, 0 = matches intact, 1 = surface
        damage (|sdist| beyond surface_tol, either direction -- loss or
        addition/residue), 2 = missing material (out-of-bounds or
        penetration beyond penetration_tol).
    """
    label = damage_pixel_label(sdist_img, surface_tol=surface_tol)
    missing = out_of_bounds_mask | (~np.isnan(penetration_img) & (penetration_img > penetration_tol))
    label[missing] = 2
    return label


def _damage_cell_permutation_under_flip(axis_bins, damage_cell_bins, flip_axis):
    """
    Which flat damage cell index ends up holding a given cell's data if the
    canonical frame was secretly rotated 180 degrees about `flip_axis`
    (or left alone, if flip_axis is None) before grading.

    A 180-degree rotation about an axis leaves that axis's coordinate sign
    unchanged and flips the other two -- in bin-index terms, that means
    the bin index on `flip_axis` is unchanged, and the bin order is
    reversed on the other two axes (bin b -> (n-1-b), same operation
    regardless of how many bins that axis has). This generalizes the old
    fixed 2x2x2 sign-flip logic to the anisotropic grid.
    """
    new_bins = damage_cell_bins.copy()
    if flip_axis is not None:
        for axis in range(3):
            if axis != flip_axis:
                new_bins[:, axis] = (axis_bins[axis] - 1) - damage_cell_bins[:, axis]
    multipliers = np.array([axis_bins[1] * axis_bins[2], axis_bins[2], 1])
    return new_bins @ multipliers


class BrickDamageGrader:
    """Grades per-damage-cell material loss against an intact CAD mesh,
    using a 16-cell grid (4 bins along the longest axis, 2 along each
    other axis) by default -- pass axis_bins to override.

    Two independent grading paths, for two different situations:
      - grade()/visible_counts(): from real *observed* points (e.g.
        to_canonical() applied to an actual depth camera + estimated
        pose). Use this for real perception, where the true damaged
        geometry isn't known and has to be estimated from a sensor.
      - grade_mesh(): from a *known* damaged mesh, sampled directly and
        compared to the intact one -- no camera/pose/depth involved. Use
        this only when the true damaged geometry is already known outright
        (e.g. synthetic-data labeling from the mesh that generated the
        render), never as a stand-in for real sensor-based grading.
    """

    def __init__(self, mesh_path, axis_bins=None):
        """
        :param axis_bins: optional [bins_x, bins_y, bins_z], 3 positive ints,
            overriding the default auto layout (2 bins on every axis except
            the longest, which gets 4).
        """
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        # Some CAD exports leave duplicate vertices along UV/normal seams,
        # which makes an otherwise-closed solid look non-watertight to
        # trimesh. Merging by position only (ignoring texture/normal) fixes
        # that without changing the geometry.
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight:
            raise ValueError(
                f"CAD mesh at {mesh_path!r} is not watertight even after "
                "vertex-merge/hole-fill repair, so trimesh.proximity."
                "signed_distance cannot be trusted. Use a voxel-occupancy "
                "fallback (rasterize the mesh to a voxel grid and compare "
                "occupancy) instead of signed distance for this mesh."
            )
        self.mesh = mesh
        self.centroid = mesh.bounds.mean(axis=0)

        if axis_bins is not None:
            if len(axis_bins) != 3 or any(b <= 0 for b in axis_bins):
                raise ValueError(f"axis_bins must be 3 positive ints, got {axis_bins!r}")
            self.axis_bins = [int(b) for b in axis_bins]
        else:
            # Which axis is "long" is mesh-dependent (not a fixed convention),
            # so the grid layout is built per-instance here rather than as a
            # class-level constant.
            long_axis = int(np.argmax(mesh.extents))
            self.axis_bins = [2, 2, 2]
            self.axis_bins[long_axis] = 4
        self.num_damage_cells = self.axis_bins[0] * self.axis_bins[1] * self.axis_bins[2]

        # One row per damage cell: (b0, b1, b2) bin index along each axis,
        # in the same flat order damage_cell_index() packs points into
        # (axis 0 is the most significant / outermost, matching the old
        # x/y/z bit-pack ordering).
        self.damage_cell_bins = np.array(
            [
                (b0, b1, b2)
                for b0 in range(self.axis_bins[0])
                for b1 in range(self.axis_bins[1])
                for b2 in range(self.axis_bins[2])
            ],
            dtype=np.int64,
        )

        # Size of one grid cell along each axis -- constant per axis (a
        # uniform subdivision within that axis), used both to bin points
        # in damage_cell_index() and to size/position each cell for
        # visualization (see damage_cell_centers()).
        self.damage_cell_extents = mesh.extents / np.array(self.axis_bins, dtype=np.float64)
        # Per-face patch table (see the FACE section below).
        self._build_face_patches()

        # The brick's own rotational symmetry group: identity, plus a
        # 180-degree flip about each of its 3 principal axes. See
        # resolve_symmetry() below. Built per-instance since it depends on
        # this mesh's own axis_bins/damage_cell_bins.
        self.SYM_ROTATIONS = [
            np.eye(3),
            trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0))[:3, :3],
            trimesh.transformations.rotation_matrix(np.pi, (0, 1, 0))[:3, :3],
            trimesh.transformations.rotation_matrix(np.pi, (0, 0, 1))[:3, :3],
        ]
        self.SYM_PERMUTATIONS = [
            _damage_cell_permutation_under_flip(self.axis_bins, self.damage_cell_bins, None),
            _damage_cell_permutation_under_flip(self.axis_bins, self.damage_cell_bins, 0),
            _damage_cell_permutation_under_flip(self.axis_bins, self.damage_cell_bins, 1),
            _damage_cell_permutation_under_flip(self.axis_bins, self.damage_cell_bins, 2),
        ]

    def damage_cell_centers(self):
        """
        Returns:
            (num_damage_cells, 3) canonical-frame centers of each grid
            cell, in the same index order as damage_cell_bins /
            grade()'s cell axis, for visualizing per-cell scores (image
            overlays, RViz markers). Together the num_damage_cells
            sub-boxes (each damage_cell_extents in size) exactly tile the
            CAD bounding box.
        """
        bounds_min = self.mesh.bounds[0]
        return bounds_min + (self.damage_cell_bins.astype(np.float64) + 0.5) * self.damage_cell_extents

    def damage_cell_vis_anchors(self):
        """
        Returns:
            (num_damage_cells, 3) canonical-frame anchor points for
            *visualizing* per-cell scores -- distinct from
            damage_cell_centers(), which is for grading (tiling/binning
            observed points) and places every cell at its bin's midpoint,
            clustering all cells toward the object's centerline.

            Here, bin 0 and bin (N-1) of each axis are placed exactly at
            that axis's two extremes, with intermediate bins evenly
            spaced between -- so a corner cell (extreme bin on every
            axis) lands on the CAD box's actual corner, and a run of
            cells along one axis reads as an evenly spaced line
            (o--o--o--o) along that edge, matching how the grid actually
            looks laid over the object instead of bunching near the
            middle.
        """
        bounds_min, bounds_max = self.mesh.bounds
        extent = bounds_max - bounds_min
        anchors = np.empty_like(self.damage_cell_bins, dtype=np.float64)
        for axis in range(3):
            n = self.axis_bins[axis]
            if n > 1:
                frac = self.damage_cell_bins[:, axis].astype(np.float64) / (n - 1)
            else:
                frac = np.full(len(self.damage_cell_bins), 0.5)
            anchors[:, axis] = bounds_min[axis] + frac * extent[axis]
        return anchors

    # ---------------------------------------------------------------
    # Per-FACE patches.
    #
    # The 16-cell grid scores VOLUMES, which pools together points from
    # several faces of the same cell. That is a real defect, not just
    # coarseness: a corner cell whose top face is large, camera-facing
    # and clean, and whose end face is destroyed but nearly edge-on, has
    # thousands of intact top-face pixels and a few hundred damaged
    # end-face ones. The clean face SWAMPS the broken one and the cell
    # reads healthy while a whole face of it is missing.
    #
    # Scoring each face patch separately is what makes the "fraction of
    # observed pixels that are damaged" normalisation sound: within one
    # flat patch, seeing only a sliver but finding all of it damaged
    # genuinely does mean that patch is destroyed. Across mixed faces it
    # does not.
    #
    # Layout, from the existing axis_bins (so the two stay aligned):
    # each of the 6 faces is subdivided by the two axes it spans. For a
    # 4x2x2 grid that is 8+8 (top/bottom) + 8+8 (long sides) + 4+4
    # (ends) = 40 patches. A face nobody looked at scores NaN rather
    # than silently averaging into a neighbour.
    # ---------------------------------------------------------------
    FACE_AXES = (0, 0, 1, 1, 2, 2)      # which axis each face is normal to
    FACE_SIGNS = (0, 1, 0, 1, 0, 1)     # 0 = min side, 1 = max side
    FACE_NAMES = ("x-", "x+", "y-", "y+", "z-", "z+")

    def _build_face_patches(self):
        """Flat list of (face, u_bin, v_bin) for every patch, plus the
        offset where each face's patches start. u/v are the two axes the
        face spans, in ascending axis order."""
        patches, offsets = [], []
        for f in range(6):
            offsets.append(len(patches))
            ua, va = self._face_uv_axes(f)
            for u in range(self.axis_bins[ua]):
                for v in range(self.axis_bins[va]):
                    patches.append((f, u, v))
        self.damage_face_patches = np.array(patches, dtype=np.int64)
        self.damage_face_offsets = np.array(offsets, dtype=np.int64)
        self.num_damage_faces = len(patches)

    def _face_uv_axes(self, face):
        """The two axes a face spans, ascending."""
        normal = self.FACE_AXES[face]
        return tuple(a for a in range(3) if a != normal)

    def damage_face_ids(self):
        """Human-readable id per patch, e.g. 'z+_1_0' -- face, then the
        two bin indices along the axes that face spans."""
        return [
            f"{self.FACE_NAMES[f]}_{u}_{v}"
            for f, u, v in self.damage_face_patches
        ]

    def _patch_index(self, face, u_bin, v_bin):
        ua, va = self._face_uv_axes(face)
        return self.damage_face_offsets[face] + u_bin * self.axis_bins[va] + v_bin

    def damage_face_index(self, pts):
        """Which face patch each canonical-frame point belongs to.

        A point is assigned to the face whose plane it is CLOSEST to --
        the surface it was observed on. Points are expected to lie on or
        near the box surface (they are depth returns off the brick), so
        nearest-plane is the natural reading of "which face is this".

        Returns: (N,) int64 in [0, num_damage_faces).
        """
        pts = np.asarray(pts, dtype=np.float64)
        bounds_min, bounds_max = self.mesh.bounds
        # Distance to each of the 6 planes, in FACE order.
        d = np.empty((len(pts), 6), dtype=np.float64)
        for axis in range(3):
            d[:, 2 * axis] = np.abs(pts[:, axis] - bounds_min[axis])
            d[:, 2 * axis + 1] = np.abs(bounds_max[axis] - pts[:, axis])
        face = np.argmin(d, axis=1).astype(np.int64)

        out = np.empty(len(pts), dtype=np.int64)
        for f in range(6):
            sel = face == f
            if not sel.any():
                continue
            ua, va = self._face_uv_axes(f)
            ub = np.clip(
                np.floor((pts[sel, ua] - bounds_min[ua]) / self.damage_cell_extents[ua]),
                0, self.axis_bins[ua] - 1,
            ).astype(np.int64)
            vb = np.clip(
                np.floor((pts[sel, va] - bounds_min[va]) / self.damage_cell_extents[va]),
                0, self.axis_bins[va] - 1,
            ).astype(np.int64)
            out[sel] = self.damage_face_offsets[f] + ub * self.axis_bins[va] + vb
        return out

    def damage_face_centers(self):
        """(num_damage_faces, 3) canonical-frame centre of each patch,
        ON its face plane -- the anchor for drawing a coloured quad."""
        bounds_min, bounds_max = self.mesh.bounds
        out = np.zeros((self.num_damage_faces, 3), dtype=np.float64)
        for i, (f, u, v) in enumerate(self.damage_face_patches):
            normal = self.FACE_AXES[f]
            ua, va = self._face_uv_axes(f)
            out[i, normal] = bounds_max[normal] if self.FACE_SIGNS[f] else bounds_min[normal]
            out[i, ua] = bounds_min[ua] + (u + 0.5) * self.damage_cell_extents[ua]
            out[i, va] = bounds_min[va] + (v + 0.5) * self.damage_cell_extents[va]
        return out

    def damage_face_quads(self):
        """(num_damage_faces, 4, 3) canonical-frame corners of each patch,
        wound consistently, for drawing filled surfaces."""
        bounds_min, bounds_max = self.mesh.bounds
        quads = np.zeros((self.num_damage_faces, 4, 3), dtype=np.float64)
        for i, (f, u, v) in enumerate(self.damage_face_patches):
            normal = self.FACE_AXES[f]
            ua, va = self._face_uv_axes(f)
            n_val = bounds_max[normal] if self.FACE_SIGNS[f] else bounds_min[normal]
            u0 = bounds_min[ua] + u * self.damage_cell_extents[ua]
            u1 = u0 + self.damage_cell_extents[ua]
            v0 = bounds_min[va] + v * self.damage_cell_extents[va]
            v1 = v0 + self.damage_cell_extents[va]
            for j, (uu, vv) in enumerate(((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
                quads[i, j, normal] = n_val
                quads[i, j, ua] = uu
                quads[i, j, va] = vv
            # Wind so the face normal points OUTWARD.
            #
            # Emitting the same (u, v) order for every face gives an
            # outward normal on the max-side faces and an inward one on
            # the min-side faces, because the (u, v) axes keep their
            # handedness while the normal flips. Renderers that light
            # triangle lists (RViz does) draw inward-facing triangles
            # BLACK, so half the patches vanish into a black brick with
            # perfectly correct colours attached.
            e1 = quads[i, 1] - quads[i, 0]
            e2 = quads[i, 3] - quads[i, 0]
            outward = 1.0 if self.FACE_SIGNS[f] else -1.0
            if np.cross(e1, e2)[normal] * outward < 0:
                quads[i] = quads[i, ::-1]
        return quads

    # Ray grading, per face patch. See grade_faces_by_raycast().
    RAY_DEADBAND_M = 0.003      # void shallower than this reads as intact
    RAY_SATURATE_M = 0.015      # void this deep or deeper reads as fully missing
    RAY_STANDOFF_M = 0.040      # how far outside the CAD face rays start
    RAY_GRID = 18               # rays per patch edge (18x18 = 324 per patch)
    RAY_BLIND_FACES = ("z+",)   # faces the scanner cannot see; scored None

    def grade_faces_by_raycast(self, scan_mesh, deadband=None, saturate=None,
                               standoff=None, grid=None, blind_faces=None,
                               return_detail=False):
        """Grade each face patch by how deep the void behind it is.

        Rays are fired INWARD along each patch's own face normal, from a
        standoff outside the CAD face plane, and the score is how far
        each ray travels past where intact material should be before it
        hits the scan:

            depth = hit_distance - standoff      (<=0 intact, >0 void)
            score = mean over rays of clip((depth - deadband) /
                                           (saturate - deadband), 0, 1)

        A ray that hits nothing counts as fully missing (1.0).

        Why this and not grade()'s signed distance. grade() samples the
        SCAN and asks how far each point lies inside the intact envelope,
        using distance to the nearest CAD surface in ANY direction. Two
        consequences:

        - A scan that is a few mm oversize on one axis -- these are, by
          12 mm in y -- makes every point on those faces read as outside
          the envelope, which subtracts from each damage measurement and
          biases the whole face toward "clean". Firing along the face
          normal instead just stops every ray on that face early by the
          same offset, so it cancels rather than accumulating. Measured
          on brick 1, clean patches come back at -0.02 mm and +0.26 mm.
        - Where material is entirely gone there are no scan points to
          sample, so the region contributes nothing or drops below
          min_pts. A ray gets a no-hit there and correctly scores 1.0.

        The method cannot tell MISSING material from UNSCANNED surface:
        on the open underside the rays pass straight through and hit the
        top from inside, reading ~65 mm and a confident, wrong 1.00.
        Those faces are therefore scored None rather than graded --
        excluding them is load-bearing here, not just tidy.

        Args:
            scan_mesh: trimesh mesh ALREADY aligned into this frame.
            return_detail: also return per-patch diagnostics.

        Returns:
            scores: (num_damage_faces,) float, NaN for blind faces.
            detail (optional): dict of (num_damage_faces,) arrays --
                median_depth_m, no_hit_frac, n_rays.
        """
        deadband = self.RAY_DEADBAND_M if deadband is None else deadband
        saturate = self.RAY_SATURATE_M if saturate is None else saturate
        standoff = self.RAY_STANDOFF_M if standoff is None else standoff
        grid = self.RAY_GRID if grid is None else grid
        blind = self.RAY_BLIND_FACES if blind_faces is None else tuple(blind_faces)
        if not saturate > deadband:
            raise ValueError(f"saturate ({saturate}) must exceed deadband ({deadband})")

        quads = self.damage_face_quads()
        scores = np.full(self.num_damage_faces, np.nan)
        med = np.full(self.num_damage_faces, np.nan)
        nohit = np.full(self.num_damage_faces, np.nan)
        nray = np.zeros(self.num_damage_faces, dtype=np.int64)

        # Inset from the patch edge: a ray exactly on the boundary is a
        # coin flip between this patch and its neighbour, and on a convex
        # corner it can miss the brick entirely and read as a void.
        t = np.linspace(0.08, 0.92, grid)

        for i, (f, _u, _v) in enumerate(self.damage_face_patches):
            if self.FACE_NAMES[f] in blind:
                continue
            axis = self.FACE_AXES[f]
            normal = np.zeros(3)
            normal[axis] = 1.0 if self.FACE_SIGNS[f] else -1.0

            q = quads[i]
            e1, e2 = q[1] - q[0], q[3] - q[0]
            pts = (q[0][None, None, :]
                   + t[:, None, None] * e1[None, None, :]
                   + t[None, :, None] * e2[None, None, :]).reshape(-1, 3)

            origins = pts + normal * standoff
            dirs = np.repeat(-normal[None, :], len(origins), axis=0)
            loc, ray_idx, _ = scan_mesh.ray.intersects_location(
                origins, dirs, multiple_hits=False)

            depth = np.full(len(origins), np.inf)
            if len(ray_idx):
                depth[ray_idx] = (np.linalg.norm(loc - origins[ray_idx], axis=1)
                                  - standoff)
            missing = np.where(np.isinf(depth), 1.0,
                               np.clip((depth - deadband) / (saturate - deadband),
                                       0.0, 1.0))
            scores[i] = float(missing.mean())
            nray[i] = len(origins)
            nohit[i] = float(np.isinf(depth).mean())
            finite = depth[~np.isinf(depth)]
            if len(finite):
                med[i] = float(np.median(finite))

        if return_detail:
            return scores, {"median_depth_m": med, "no_hit_frac": nohit,
                            "n_rays": nray}
        return scores

    # Kinds of neighbour touch, in increasing order of how localised the
    # shared geometry is. See damage_face_adjacency.
    TOUCH_SAME_FACE = 0     # on the same face (edge- or corner-adjacent)
    TOUCH_CROSS_EDGE = 1    # different faces, sharing a whole edge segment
    TOUCH_CROSS_CORNER = 2  # different faces, touching at a single point --
                            # three faces of the brick converging there

    def damage_face_adjacency(self, eps=1e-6, return_kind=False):
        """Neighbour lists over the face patches -- the graph that lets a
        patch be informed by the surfaces around it.

        Adjacency is decided GEOMETRICALLY: two patches are neighbours
        when their quads touch, tested by intersecting their axis-aligned
        bounds with a small tolerance. That single rule produces three
        kinds of neighbour without special-casing any of them:

          - on the same face, the 8 surrounding patches (edge-sharing
            neighbours touch along an edge, diagonal ones at a corner),
          - across a face edge, the patches on the adjoining face that
            run along the same physical edge (their quads overlap along
            a whole line segment),
          - across a face CORNER, the patches on a third face that meet
            this one at a single point -- the physical corner of the
            brick, where three faces converge (their quads overlap in
            only one dimension: a point, not a segment).

        The cross-face links are the important ones. Damage does not
        respect face boundaries -- a broken corner spans the three faces
        that meet there -- so a patch on one face genuinely is evidence
        about its neighbour on the next. A corner-point neighbour is
        stronger evidence than a same-face one for the same reason a
        broken corner IS evidence about all three faces at once: it is
        the single most concentrated, most damage-prone piece of
        geometry a rectangular brick has. This is what
        evaluation.smoothing's `corner` strategy weights on.

        Args:
            return_kind: also return, for each neighbour list, an int
                array of TOUCH_* codes in the same order.

        Returns: list of (num_damage_faces,) int arrays, or
            (neighbours, kinds) if return_kind.
        """
        quads = self.damage_face_quads()
        lo = quads.min(axis=1)   # (N, 3)
        hi = quads.max(axis=1)
        n = len(quads)
        faces = self.damage_face_patches[:, 0]
        neighbours, kinds = [], []
        for i in range(n):
            touch = np.all(
                (lo <= hi[i][None, :] + eps) & (hi >= lo[i][None, :] - eps), axis=1
            )
            touch[i] = False
            js = np.nonzero(touch)[0]
            neighbours.append(js)
            if return_kind:
                overlap_lo = np.maximum(lo[i][None, :], lo[js])
                overlap_hi = np.minimum(hi[i][None, :], hi[js])
                # Dimensions where the shared region has nonzero extent:
                # 2 for two patches on the same plane (same face), 1 for
                # a shared edge line, 0 for a single shared point.
                ndim = np.sum((overlap_hi - overlap_lo) > eps, axis=1)
                same_face = faces[js] == faces[i]
                kind = np.where(same_face, self.TOUCH_SAME_FACE,
                                np.where(ndim == 0, self.TOUCH_CROSS_CORNER,
                                        self.TOUCH_CROSS_EDGE))
                kinds.append(kind)
        return (neighbours, kinds) if return_kind else neighbours

    def smooth_face_scores(self, scores, weights, lam=0.5, fill_unobserved=True,
                           neighbours=None, lam_down=None, iters=1):
        """Confidence-weighted diffusion over the patch graph.

            s'_i = (w_i*s_i + lam * sum_j w_j*s_j) / (w_i + lam * sum_j w_j)

        A patch backed by strong evidence barely moves; one backed by
        little leans on its neighbours. That is the useful asymmetry --
        it is exactly the thin-evidence patches whose raw numbers deserve
        least trust, and they are the ones this corrects.

        ASYMMETRY (lam_down). Coupled symmetrically, cleanliness spreads
        faster than damage, and not by accident: damage destroys the
        evidence that would prove it. A broken patch returns few depth
        points -- there is less surface left to return any -- while a
        pristine face-on patch floods the sensor. Measured on a real
        two-view capture: damaged patches (score > 0.5) carried 3.4x
        LESS weight than clean ones (1369 vs 4709 points), correlation
        between score and evidence -0.29. Confidence weighting then lets
        the well-observed clean patches outvote the sparsely-observed
        damaged ones, so smoothing drags the damage away -- exactly what
        it should not do.

        lam_down is the coupling strength for neighbours that would pull
        a patch DOWN (toward clean); lam stays the strength for those
        pulling it UP (toward damage). Setting lam_down < lam encodes
        the real asymmetry in the evidence: "my neighbour is destroyed"
        is strong evidence about me, because damage is contiguous, while
        "my neighbour is fine" is weak evidence, because I might be the
        broken corner. lam_down=None keeps the symmetric behaviour.

        fill_unobserved additionally gives a NaN patch the weighted mean
        of whatever neighbours were observed. This INVENTS a value, so
        the result is returned separately and must never be presented as
        measured: a patch nobody looked at has no score, it has an
        inference. Patches with no observed neighbour at all stay NaN.

        Args:
            scores: (num_damage_faces,) with NaN for unobserved.
            weights: (num_damage_faces,) evidence weight per patch
                (effective counts).
            lam: neighbour influence. 0 reproduces `scores` exactly.

        iters applies the whole operation repeatedly, feeding each pass
        its own output. One pass moves damage exactly ONE hop, to
        immediate neighbours -- raising lam makes that hop stronger, not
        longer, and it saturates. Reach needs passes: measured on a real
        capture at lam=1, patches scoring >0.3 went 16 -> 24 -> 30 over
        three passes. Weights are held fixed across passes, since the
        evidence behind a patch does not change just because its
        neighbours were consulted again.

        Returns: (smoothed, filled_mask) -- filled_mask marks patches
            whose value came only from neighbours.
        """
        result = np.asarray(scores, dtype=np.float64)
        filled_any = np.zeros(len(result), dtype=bool)
        for _ in range(max(1, int(iters))):
            result, filled = self._smooth_face_scores_once(
                result, weights, lam, fill_unobserved, neighbours, lam_down
            )
            filled_any |= filled
        return result, filled_any

    def _smooth_face_scores_once(self, scores, weights, lam, fill_unobserved,
                                 neighbours, lam_down):
        """One diffusion pass; see smooth_face_scores()."""
        scores = np.asarray(scores, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if neighbours is None:
            neighbours = self.damage_face_adjacency()
        observed = ~np.isnan(scores)
        out = scores.copy()
        filled = np.zeros(len(scores), dtype=bool)

        for i in range(len(scores)):
            nb = neighbours[i]
            nb = nb[observed[nb]] if len(nb) else nb
            if len(nb) == 0:
                continue
            w_nb = weights[nb]
            if observed[i]:
                w_i = float(weights[i])
                # Per-neighbour coupling strength: neighbours above this
                # patch pull it up at `lam`, neighbours below pull it
                # down at `lam_down`.
                if lam_down is None:
                    lam_j = np.full(len(nb), lam, dtype=np.float64)
                else:
                    lam_j = np.where(scores[nb] >= scores[i], lam, lam_down)
                num_nb = float((lam_j * w_nb * scores[nb]).sum())
                den_nb = float((lam_j * w_nb).sum())
                out[i] = (w_i * scores[i] + num_nb) / max(w_i + den_nb, 1e-9)
            elif fill_unobserved and float(w_nb.sum()) > 0:
                # An unobserved patch has no value of its own to be
                # pulled from, so the asymmetry has nothing to act on
                # here: it takes the plain weighted mean of whatever was
                # actually seen around it.
                out[i] = float((w_nb * scores[nb]).sum()) / float(w_nb.sum())
                filled[i] = True
        return out, filled

    def ray_entry_face_index(self, origins, directions):
        """Which face PATCH a ray first enters, for rays that see past
        where the brick should be.

        The slab method already determines which axis produced the entry
        -- that axis and the sign of the ray tell you the face -- so the
        entry face is available for free. This is what gives missing
        material a well-defined home: a missing corner removes material
        from several faces at once, but the ray came through exactly
        one, and that is the patch whose material is gone.

        Returns: (idx, hit), same contract as ray_entry_cell_index().
        """
        box_volume = float(np.prod(self.mesh.extents))
        if not np.isclose(self.mesh.volume, box_volume, rtol=1e-3):
            raise ValueError(
                "ray_entry_face_index() requires the intact mesh to be exactly its "
                f"bounding box (volume {self.mesh.volume:.6g} vs box {box_volume:.6g})."
            )
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(directions, dtype=np.float64)
        bounds_min, bounds_max = self.mesh.bounds

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (bounds_min[None, :] - origins) / dirs
            t2 = (bounds_max[None, :] - origins) / dirs
        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)
        parallel = dirs == 0.0
        inside = (origins >= bounds_min[None, :]) & (origins <= bounds_max[None, :])
        t_near = np.where(parallel, np.where(inside, -np.inf, np.inf), t_near)
        t_far = np.where(parallel, np.where(inside, np.inf, -np.inf), t_far)

        entry_axis = np.argmax(t_near, axis=1)
        t_entry = np.max(t_near, axis=1)
        t_exit = np.min(t_far, axis=1)
        hit = (t_exit >= t_entry) & (t_exit > 0) & np.isfinite(t_entry)

        idx = np.zeros(len(origins), dtype=np.int64)
        if hit.any():
            pts = origins[hit] + t_entry[hit, None] * dirs[hit]
            ax = entry_axis[hit]
            # Entered through the min plane when the ray travels in the
            # +axis direction, and vice versa.
            comp = dirs[hit][np.arange(len(ax)), ax]
            face = 2 * ax + (comp < 0).astype(np.int64)
            sub = np.zeros(len(ax), dtype=np.int64)
            for f in range(6):
                m = face == f
                if not m.any():
                    continue
                ua, va = self._face_uv_axes(f)
                ub = np.clip(
                    np.floor((pts[m, ua] - bounds_min[ua]) / self.damage_cell_extents[ua]),
                    0, self.axis_bins[ua] - 1,
                ).astype(np.int64)
                vb = np.clip(
                    np.floor((pts[m, va] - bounds_min[va]) / self.damage_cell_extents[va]),
                    0, self.axis_bins[va] - 1,
                ).astype(np.int64)
                sub[m] = self.damage_face_offsets[f] + ub * self.axis_bins[va] + vb
            idx[hit] = sub
        return idx, hit

    def damage_cell_index(self, pts):
        """
        Args:
            pts: (N, 3) canonical-frame points.

        Returns:
            (N,) int64 array in [0, num_damage_cells), indexing
            damage_cell_bins.
        """
        pts = np.asarray(pts, dtype=np.float64)
        bounds_min = self.mesh.bounds[0]
        bins = np.zeros((len(pts), 3), dtype=np.int64)
        for axis in range(3):
            b = np.floor((pts[:, axis] - bounds_min[axis]) / self.damage_cell_extents[axis])
            bins[:, axis] = np.clip(b, 0, self.axis_bins[axis] - 1).astype(np.int64)
        return (
            bins[:, 0] * (self.axis_bins[1] * self.axis_bins[2])
            + bins[:, 1] * self.axis_bins[2]
            + bins[:, 2]
        )

    def grade(
        self, pts, surface_tol=0.003, inside_tol=0.002, min_pts=15,
        return_counts=False, bounds_margin=0.005, by_face=False,
    ):
        """
        Args:
            pts: (N, 3) canonical-frame points, e.g. from to_canonical().
            surface_tol: points within this signed distance (meters) of the
                intact surface count as matching it (no damage).
            inside_tol: width (meters) of the ramp from "matches surface" to
                "confidently missing material", starting at surface_tol.
                A point at signed_distance >= surface_tol + inside_tol is
                fully counted as missing.
            min_pts: minimum number of points a damage cell needs before it
                gets a score; otherwise it's reported as NaN.
            return_counts: if True, also return how many observed points
                landed in each damage cell, so a score can be judged
                against how much evidence it's actually based on.
            bounds_margin: points farther than this (meters) outside the
                mesh's own bounding box, on any axis, are dropped before
                binning. damage_cell_index() clips whatever bin index it's
                given into [0, axis_bins), so without this filter a point
                that isn't part of the object at all -- e.g. a silhouette-
                edge "flying pixel" that landed on the floor behind the
                brick, tens of mm outside the box -- still gets silently
                assigned to whichever edge/corner cell it clips into,
                inflating that cell's count with points that have nothing
                to do with it. Points within the margin (real surface noise
                right at the true boundary) are kept and clipped as before.

        Returns:
            (num_damage_cells,) float64 array of per-cell damage scores in
            [0, 1], or NaN for cells with fewer than min_pts observed
            points. If return_counts is True, also returns a
            (num_damage_cells,) int64 array of the per-cell point counts.
        """
        # by_face scores the 40 SURFACE patches instead of the 16
        # volumetric cells, matching what the ROS learned path produces
        # with score_by_face -- so ground truth can be compared at the
        # same granularity as the prediction rather than through a
        # lossy cell mapping.
        n_out = self.num_damage_faces if by_face else self.num_damage_cells
        _bin_index = self.damage_face_index if by_face else self.damage_cell_index

        pts = np.asarray(pts, dtype=np.float64)
        scores = np.full(n_out, np.nan, dtype=np.float64)
        counts = np.zeros(n_out, dtype=np.int64)
        if len(pts) == 0:
            return (scores, counts) if return_counts else scores

        if bounds_margin is not None:
            bounds_min, bounds_max = self.mesh.bounds
            in_bounds = np.all(
                (pts >= bounds_min - bounds_margin) & (pts <= bounds_max + bounds_margin),
                axis=1,
            )
            pts = pts[in_bounds]
            if len(pts) == 0:
                return (scores, counts) if return_counts else scores

        idx = _bin_index(pts)
        # signed_distance: >0 inside the intact solid, ~0 on its surface,
        # <0 outside it. A point sitting well inside the intact envelope
        # means the observed surface has receded past where intact material
        # should be, i.e. that material is missing.
        sdist = trimesh.proximity.signed_distance(self.mesh, pts)
        missing_frac = np.clip((sdist - surface_tol) / inside_tol, 0.0, 1.0)

        for cell in range(n_out):
            sel = idx == cell
            n = int(sel.sum())
            counts[cell] = n
            if n < min_pts:
                continue
            scores[cell] = missing_frac[sel].mean()
        return (scores, counts) if return_counts else scores

    def grade_signed(
        self, pts, surface_tol=0.003, max_dist=0.05, min_pts=15,
        return_counts=False, bounds_margin=None,
    ):
        """
        Diagnostic sibling of grade() -- NOT the production scoring
        function (nothing else in the pipeline reads this), built to
        investigate a specific real-world symptom: a real capture graded
        as "damaged" almost everywhere via grade(), while visual
        inspection of the same point cloud showed it sitting *outside*
        the intact reference, not receded into it. grade()'s
        missing_frac is one-sided by construction (clip(...,0,1) maps
        every outside point to exactly 0, indistinguishable from "exactly
        on the surface"), so it can't actually show you which of those
        two is happening. This can, by scoring the full signed distance
        instead of just its positive side:

            -1 = material loss (observed surface receded into the intact
                 envelope by max_dist or more)
             0 = matches the intact reference (within surface_tol)
            +1 = material addition / the observed surface sits outside
                 the intact envelope by max_dist or more (real extra
                 material, e.g. mortar residue -- or a mis-registered
                 point cloud makes an intact surface look like this)

        Note the sign flip from grade()'s own convention: this module's
        signed_distance is >0 *inside* the intact solid (grade()'s
        "missing"), so mapping that directly would put loss on the
        positive side -- flipped here to put loss at -1 per how this
        method's caller wants to read it.

        Args:
            pts: (N, 3) canonical-frame points, e.g. from to_canonical().
            surface_tol: dead zone (meters) around 0, same role as in
                grade() -- keeps sensor noise right at the true surface
                from reading as nonzero loss/addition.
            max_dist: distance (meters) at which the score saturates at
                -1/+1 in each direction. grade()'s equivalent (inside_tol)
                is 2mm; this defaults far wider (5cm) since the point of
                this method is investigating registration-scale (mm-cm)
                discrepancies, not fine surface detail.
            min_pts: see grade().
            return_counts: see grade().
            bounds_margin: see grade() -- but None (no filtering) by
                default here, unlike grade()'s 5mm default. grade()'s
                filter exists to drop points that plainly aren't part of
                the object at all; this method's whole purpose is to see
                *how far* a cell's points sit from the reference, up to
                max_dist, so a tight default margin would silently
                discard exactly the "addition" points this is meant to
                reveal. Pass a value if you do want the flying-pixel
                filter grade() has.

        Returns: same shape/NaN convention as grade(), but values in
        [-1, 1] instead of [0, 1].
        """
        pts = np.asarray(pts, dtype=np.float64)
        scores = np.full(self.num_damage_cells, np.nan, dtype=np.float64)
        counts = np.zeros(self.num_damage_cells, dtype=np.int64)
        if len(pts) == 0:
            return (scores, counts) if return_counts else scores

        if bounds_margin is not None:
            bounds_min, bounds_max = self.mesh.bounds
            in_bounds = np.all(
                (pts >= bounds_min - bounds_margin) & (pts <= bounds_max + bounds_margin),
                axis=1,
            )
            pts = pts[in_bounds]
            if len(pts) == 0:
                return (scores, counts) if return_counts else scores

        idx = self.damage_cell_index(pts)
        sdist = trimesh.proximity.signed_distance(self.mesh, pts)  # >0 inside (loss), <0 outside (addition)
        beyond_dead_zone = np.sign(sdist) * np.clip(np.abs(sdist) - surface_tol, 0.0, None)
        signed_score = np.clip(-beyond_dead_zone / max_dist, -1.0, 1.0)

        for cell in range(self.num_damage_cells):
            sel = idx == cell
            n = int(sel.sum())
            counts[cell] = n
            if n < min_pts:
                continue
            scores[cell] = signed_score[sel].mean()
        return (scores, counts) if return_counts else scores

    def signed_distance_image(self, depth, mask, intr, pose, erode_px=2, bounds_margin=0.005):
        """
        Per-pixel signed distance (meters) from the intact reference mesh,
        for every pixel with valid depth inside `mask` -- the same
        quantity grade_signed() aggregates per cell, kept here at full
        pixel resolution as a dense label for a segmentation head (see
        damage_pixel_label()), instead of collapsed into 16 cell scores.

        Convention matches grade()/grade_signed(): >0 = observed surface
        sits inside the intact envelope (material missing), <0 = outside
        it (material added / mis-registration), ~0 = matches the intact
        surface.

        Args:
            depth: (H, W) depth map in meters -- pass the *clean* render,
                not a noise-augmented one, when labeling synthetic data:
                this is meant to be as close to true geometry as possible,
                and injecting sensor-noise realism into the label itself
                (rather than into whatever the network actually trains on
                as input) only adds label noise for no benefit.
            mask: (H, W) visibility mask, e.g. mask_visib.
            intr: (3, 3) camera intrinsics.
            pose: (4, 4) object->camera transform, see to_canonical().
            erode_px: see to_canonical() -- default 2 to drop silhouette-
                edge flying-pixel noise, same as label_damage_scores.py.
            bounds_margin: see grade() -- points farther than this
                (meters) outside the mesh's own bounding box, on any
                axis, are left unobserved (NaN) rather than scored.
                Confirmed necessary here (not just theoretical): on a
                real rendered frame, ~1% of mask_visib pixels survived
                erode_px=2 as background/floor flying-pixel noise tens
                of mm off the object (observed up to -65mm, versus the
                brick's own 65mm total short-axis extent) -- without this
                filter those would poison the label as false "damaged"
                pixels right along the silhouette boundary. grade()'s
                5mm default is reused here for the same reason.

        Returns:
            (H, W) float64 array, NaN where unobserved (outside mask, no
            valid depth, or outside bounds_margin of the mesh's own box).
        """
        H, W = depth.shape
        img = np.full((H, W), np.nan, dtype=np.float64)
        pts, rows, cols = to_canonical_indexed(depth, mask, intr, pose, erode_px=erode_px)
        if len(pts) == 0:
            return img

        if bounds_margin is not None:
            bounds_min, bounds_max = self.mesh.bounds
            in_bounds = np.all(
                (pts >= bounds_min - bounds_margin) & (pts <= bounds_max + bounds_margin),
                axis=1,
            )
            pts, rows, cols = pts[in_bounds], rows[in_bounds], cols[in_bounds]
            if len(pts) == 0:
                return img

        sdist = trimesh.proximity.signed_distance(self.mesh, pts)
        img[rows, cols] = sdist
        return img

    def out_of_bounds_mask(self, depth, mask, intr, pose, erode_px=2, bounds_margin=0.005):
        """
        (H, W) bool mask -- True where a pixel has valid depth within
        `mask` (after erode_px) but backprojects more than bounds_margin
        outside the intact mesh's own bounding box. signed_distance_image()
        silently drops these points (NaN, "unobserved") since signed
        distance isn't meaningful that far from the mesh -- but for
        labeling purposes, a mask_visib pixel claiming to be "the brick"
        that lands nowhere near the brick's actual geometry is itself
        direct evidence that real material is missing there. See
        damage_pixel_label_with_out_of_bounds().

        Confirmed on a real frame (the most severely damaged brick in the
        dataset, its worst view): 5664 of 38291 mask_visib points landed
        up to 95mm outside the mesh's own ~250mm-long bounding box,
        concentrated almost entirely (3192/4117, 78%) in the rows
        covering the visible break.

        Args must match whatever produced the sdist_img this gets paired
        with in damage_pixel_label_with_out_of_bounds() (same erode_px,
        same bounds_margin) so the two partition mask_visib's points
        cleanly.

        Returns:
            (H, W) bool array.
        """
        H, W = depth.shape
        out_mask = np.zeros((H, W), dtype=bool)
        pts, rows, cols = to_canonical_indexed(depth, mask, intr, pose, erode_px=erode_px)
        if len(pts) == 0:
            return out_mask
        bounds_min, bounds_max = self.mesh.bounds
        in_bounds = np.all(
            (pts >= bounds_min - bounds_margin) & (pts <= bounds_max + bounds_margin),
            axis=1,
        )
        out_rows, out_cols = rows[~in_bounds], cols[~in_bounds]
        out_mask[out_rows, out_cols] = True
        return out_mask

    def penetration_depth_image(self, depth, mask, intr, pose, erode_px=2):
        """
        Per-pixel penetration (meters, perpendicular-equivalent) of the
        observed surface PAST where this pixel's camera ray would first
        hit the intact reference. Positive = the ray reached deeper than
        any intact surface could allow (material missing along the ray);
        ~0 = observed surface sits where the intact one would; negative =
        observed surface sits in front of it (addition/residue).

        Why this exists when signed_distance_image() already measures
        deviation: signed distance is DIRECTION-BLIND. Found on a real
        frame: a corner broken off down to the brick's base exposes a
        surface at the same height as the intact mesh's own bottom face
        -- that point measures ~0mm signed distance (it lies on the
        intact mesh's bottom surface!) and reads "matches intact", even
        though the camera sees it from above through tens of mm of brick
        that should block the view. This method asks the question signed
        distance can't: given where the camera is, could this surface
        even be visible on an intact brick?

        Implementation: analytic ray/axis-aligned-box intersection (slab
        method) against self.mesh.bounds, comparing the box entry depth
        with the observed depth. EXACT only because intact_brick.obj is
        literally a rectangular box (verified: mesh volume equals its
        bounding-box volume to 4 decimal places) -- the method checks
        this and refuses to run otherwise; for a non-box reference mesh
        you'd need a real per-pixel raycast (or an intact-depth render
        exported at data-generation time) instead.

        The along-ray depth difference is scaled by the entry-face
        obliquity (|dir . entry_axis|) so the returned value is a
        perpendicular-equivalent distance, directly comparable to
        surface_tol-style thresholds and NOT inflated at grazing
        incidence.

        Args: same as signed_distance_image(); erode_px must match the
        one used for the sdist/out-of-bounds images this gets combined
        with (see damage_pixel_label_full()).

        Returns:
            (H, W) float64, NaN where unobserved or where the pixel ray
            misses the intact box entirely (near-silhouette rays -- those
            are out_of_bounds_mask()'s territory, not this method's).
        """
        box_volume = float(np.prod(self.mesh.extents))
        if not np.isclose(self.mesh.volume, box_volume, rtol=1e-3):
            raise ValueError(
                "penetration_depth_image()'s analytic ray-box shortcut requires the "
                f"intact mesh to be exactly its bounding box (mesh volume {self.mesh.volume:.6g} "
                f"vs box volume {box_volume:.6g}) -- use a real raycast for this mesh."
            )

        H, W = depth.shape
        img = np.full((H, W), np.nan, dtype=np.float64)

        mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
        if erode_px > 0:
            kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
            mask_bin = cv2.erode(mask_bin, kernel)
        valid = np.logical_and(mask_bin > 0, depth > 0)
        vs, us = np.where(valid)
        if len(us) == 0:
            return img
        z_obs = depth[vs, us].astype(np.float64)

        # Per-pixel ray, parameterized by DEPTH z: p_cam(z) = d_cam * z with
        # d_cam normalized to third component 1, so the slab test's t is
        # directly comparable to the observed depth value.
        uv1 = np.stack([us, vs, np.ones_like(us)], axis=0).astype(np.float64)
        d_cam = np.linalg.inv(intr) @ uv1  # (3, N)
        d_cam = (d_cam / d_cam[2:3, :]).T  # (N, 3), third component 1

        R = pose[:3, :3]
        t = pose[:3, 3]
        origin = -t @ R  # camera center in canonical frame (row-vector form of R^T @ -t)
        dirs = d_cam @ R  # (N, 3) canonical ray direction per unit depth

        bounds_min, bounds_max = self.mesh.bounds
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (bounds_min[None, :] - origin[None, :]) / dirs
            t2 = (bounds_max[None, :] - origin[None, :]) / dirs
        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)
        # Axis-parallel rays (dirs[:, a] == 0): inside that slab -> -inf/+inf
        # (never the binding constraint), outside -> +inf/-inf (guaranteed miss).
        parallel = dirs == 0.0
        inside_slab = (origin[None, :] >= bounds_min[None, :]) & (origin[None, :] <= bounds_max[None, :])
        t_near = np.where(parallel, np.where(inside_slab, -np.inf, np.inf), t_near)
        t_far = np.where(parallel, np.where(inside_slab, np.inf, -np.inf), t_far)

        entry_axis = np.argmax(t_near, axis=1)
        z_entry = np.max(t_near, axis=1)
        z_exit = np.min(t_far, axis=1)
        hit = (z_entry <= z_exit) & (z_entry > 0)

        # Perpendicular-equivalent scaling: along-ray 3D length for a depth
        # difference dz is dz*|dirs|, and the angle to the entry face's
        # normal has cos = |dirs[axis]|/|dirs| -- the |dirs| cancels, so
        # multiplying dz by the |entry-axis component| alone gives the
        # perpendicular distance directly.
        axis_component = np.abs(dirs[np.arange(len(dirs)), entry_axis])
        penetration = (z_obs - z_entry) * axis_component
        img[vs[hit], us[hit]] = penetration[hit]
        return img

    def intact_silhouette_mask(self, intr, pose, image_shape, erode_px=0):
        """
        Rasterized silhouette (H, W) bool mask of the INTACT reference
        mesh projected into the image at the given camera intrinsics/
        pose -- the full footprint the brick would occupy if it had no
        damage at all. Paired with the *observed* (damaged) mask_visib,
        (this silhouette) & ~(observed mask) finds pixels where material
        is missing entirely (background visible through a chip/break),
        which signed_distance_image() alone can't see -- that only
        covers pixels still on the remaining surface, so a break severe
        enough to remove material at the silhouette boundary contributes
        no depth points there at all.

        Computed via the 2D convex hull of the intact mesh's own convex
        hull vertices projected into the image, not a full render --
        exact (not an approximation) for a genuinely convex mesh: verified
        for intact_brick.obj specifically (mesh.convex_hull.volume ==
        mesh.volume, i.e. it already is a convex solid). Would
        systematically OVER-estimate the true silhouette on a non-convex
        reference mesh (e.g. one with a real concave notch) -- check
        convexity before trusting this on a different mesh.

        Args:
            intr: (3, 3) camera intrinsics.
            pose: (4, 4) object->camera transform, cam_pt = R @ obj_pt + t
                (same convention as to_canonical()).
            image_shape: (H, W) of the target image.
            erode_px: shrinks the rasterized silhouette by this many
                pixels before returning it, to guard against the
                projection/rasterization boundary mismatch described
                above. Defaults to 0 -- checked empirically on a real
                frame with a visible corner chip: at erode_px=0 the true
                mask_visib already sits within 5px total of this
                silhouette (negligible against ~13700 silhouette pixels,
                and far smaller than the real ~193px gap the chip
                produces), but even erode_px=1 erases that entire real
                gap while also cutting hundreds of pixels of genuinely
                intact surface -- the object's own pixel footprint here
                is small/thin enough (a box's side face projects to a
                fairly narrow sliver at typical capture distance) that a
                uniform few-px erosion is disproportionately destructive,
                unlike the much larger surfaces/point counts erode_px
                guards elsewhere in this module. Re-check before raising
                this on a different dataset/camera distance.

        Returns:
            (H, W) bool array, True where the intact brick would project to.
        """
        H, W = image_shape
        verts = self.mesh.convex_hull.vertices
        R = pose[:3, :3]
        t = pose[:3, 3]
        cam_pts = verts @ R.T + t
        if np.any(cam_pts[:, 2] <= 0):
            raise ValueError(
                "intact mesh has convex-hull vertices behind the camera -- "
                "pose/intrinsics look wrong (expected cam_pt.z > 0 for all of them)"
            )
        uv1 = intr @ cam_pts.T  # (3, M)
        uv = (uv1[:2] / uv1[2:3]).T  # (M, 2)

        hull_2d = cv2.convexHull(uv.astype(np.float32))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_2d.astype(np.int32), 1)
        if erode_px > 0:
            kernel = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
            mask = cv2.erode(mask, kernel)
        return mask.astype(bool)

    def grade_from_segmentation(
        self, depth, mask, intr, pose, damage_probs, threshold=0.82,
        pos_weight=5.0, erode_px=2, min_pts=200, bounds_margin=None,
        return_counts=False, return_confidence=False,
    ):
        """
        Per-cell damage scores from a learned per-PIXEL segmentation mask
        instead of from geometry -- the learned counterpart of grade(),
        deliberately returning the same quantity on the same scale so the
        two can be compared directly on the same capture.

        A cell's score is the FRACTION of its observed pixels flagged
        damaged. That is the faithful analogue of what grade() computes,
        which is worth being explicit about because it is easy to assume
        otherwise: grade()'s per-point missing_frac ramp saturates at
        surface_tol + inside_tol = 5mm, so a point 5mm inside the intact
        envelope and one 30mm inside both score 1.0. grade() is therefore
        already an areal "what fraction of this cell reads damaged"
        measure rather than a depth-weighted severity, and matching it
        keeps both paths on the scale the allocator already consumes.

        The network is only asked WHERE damage is, never how deep -- it
        was never trained to estimate depth. Pixels seen "through" a
        break need no special severity convention here either: such a
        pixel is simply a damaged pixel, counted like any other.

        Args:
            depth: (H, W) depth map in meters.
            mask: (H, W) detection/visibility mask for this object.
            intr: (3, 3) camera intrinsics.
            pose: (4, 4) object->camera transform (see to_canonical()).
                Pass the FUSED world-derived pose when aggregating across
                views, so every view's pixels land in one consistent cell
                frame and no per-view symmetry resolution is needed.
            damage_probs: (H, W) RAW sigmoid output of the segmentation
                head, aligned with depth/mask. NaN where it made no
                prediction.
            threshold: decision threshold on the RAW probability (~0.82,
                not 0.5 -- see calibrate_damage_seg_probability in
                evaluation/visualization.py).
            pos_weight: training-time damage_seg_pos_weight, used only
                for the optional calibrated-confidence output.
            erode_px: see to_canonical(); must match what the labels were
                generated with, since the head is unsupervised on the
                boundary rim that erosion removed.
            min_pts: cells with fewer observed pixels than this are NaN
                rather than scored off near-nothing. 200, NOT grade()'s
                15: there a "point" is a backprojected depth sample, here
                it is a pixel, and a legitimately-observed cell face at
                normal capture distance is thousands of them. Measured on
                a real 5-view capture, score is strongly anti-correlated
                with count (r = -0.43 against log10 count): cells holding
                1-100 pixels averaged 0.29 and peaked at 0.74, while
                cells holding 3000+ averaged 0.023. The sparse ones are
                grazing slivers clipped at a cell boundary, where both
                depth and segmentation are least reliable -- 20 of 29
                pixels tripping the threshold reads as a confident 0.69
                from a view whose image shows no damage at all. Such a
                cell should have no opinion, not a loud one.
            bounds_margin: see grade(). None by default here (unlike
                grade()'s 5mm): pixels landing outside the intact
                envelope are exactly the see-through evidence of missing
                material, and dropping them would discard the strongest
                damage signal available.

        Returns:
            (num_damage_cells,) float64 fractions in [0, 1], NaN for
            cells with fewer than min_pts observed pixels. Plus, in
            order, the (num_damage_cells,) int64 observed-pixel counts if
            return_counts, and the (num_damage_cells,) mean CALIBRATED
            damage probability if return_confidence -- the latter is a
            diagnostic, not a score: a cell reading 0.3 from marginal
            pixels is a different situation from 0.3 from confident ones.
        """
        scores = np.full(self.num_damage_cells, np.nan, dtype=np.float64)
        counts = np.zeros(self.num_damage_cells, dtype=np.int64)
        confidence = np.full(self.num_damage_cells, np.nan, dtype=np.float64)

        pts, rows, cols = to_canonical_indexed(depth, mask, intr, pose, erode_px=erode_px)
        if len(pts) == 0:
            return _pack_seg_grade(scores, counts, confidence, return_counts,
                                   return_confidence, weight_sums, return_weight_sums)

        probs = np.asarray(damage_probs)[rows, cols]
        predicted = np.isfinite(probs)
        pts, probs = pts[predicted], probs[predicted]
        if len(pts) == 0:
            return _pack_seg_grade(scores, counts, confidence, return_counts,
                                   return_confidence, weight_sums, return_weight_sums)

        if bounds_margin is not None:
            bounds_min, bounds_max = self.mesh.bounds
            in_bounds = np.all(
                (pts >= bounds_min - bounds_margin) & (pts <= bounds_max + bounds_margin),
                axis=1,
            )
            pts, probs = pts[in_bounds], probs[in_bounds]
            if len(pts) == 0:
                return _pack_seg_grade(scores, counts, confidence, return_counts,
                                   return_confidence, weight_sums, return_weight_sums)

        return self.grade_points_from_segmentation(
            pts, probs, threshold=threshold, pos_weight=pos_weight, min_pts=min_pts,
            return_counts=return_counts, return_confidence=return_confidence,
        )

    def ray_entry_cell_index(self, origins, directions):
        """Which cell a ray would first enter, for rays that pass through
        the intact mesh's box -- used to place pixels that see PAST where
        the brick should be.

        A see-through pixel's own depth is the table behind the brick, so
        it cannot say which cell is missing. Its ray still can: whatever
        cell the ray would have entered on an intact brick is the cell
        whose material is gone.

        Analytic ray/axis-aligned-box (slab method), exact only because
        the intact reference is a box -- same restriction and check as
        penetration_depth_image().

        Args:
            origins: (N, 3) ray origins in the canonical frame.
            directions: (N, 3) ray directions in the canonical frame
                (need not be normalised).

        Returns:
            (idx, hit): idx is (N,) int64 cell indices, valid only where
            hit is True; hit is (N,) bool marking rays that actually
            cross the box in front of the origin.
        """
        box_volume = float(np.prod(self.mesh.extents))
        if not np.isclose(self.mesh.volume, box_volume, rtol=1e-3):
            raise ValueError(
                "ray_entry_cell_index() requires the intact mesh to be exactly its "
                f"bounding box (volume {self.mesh.volume:.6g} vs box {box_volume:.6g})."
            )
        origins = np.asarray(origins, dtype=np.float64)
        dirs = np.asarray(directions, dtype=np.float64)
        bounds_min, bounds_max = self.mesh.bounds

        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (bounds_min[None, :] - origins) / dirs
            t2 = (bounds_max[None, :] - origins) / dirs
        t_near = np.minimum(t1, t2)
        t_far = np.maximum(t1, t2)
        parallel = dirs == 0.0
        inside = (origins >= bounds_min[None, :]) & (origins <= bounds_max[None, :])
        t_near = np.where(parallel, np.where(inside, -np.inf, np.inf), t_near)
        t_far = np.where(parallel, np.where(inside, np.inf, -np.inf), t_far)

        t_entry = np.max(t_near, axis=1)
        t_exit = np.min(t_far, axis=1)
        hit = (t_entry <= t_exit) & (t_exit > 0)
        # Nudge just inside the surface so a point landing exactly on a
        # face doesn't bin unpredictably.
        t_use = np.where(t_entry > 0, t_entry, 0.0)
        entry = origins + dirs * (t_use[:, None] + 1e-6)
        idx = self.damage_cell_index(entry)
        return idx, hit

    def grade_points_from_segmentation(
        self, canon_pts, point_probs, threshold=0.82, pos_weight=5.0,
        min_pts=200, bounds_margin=0.005, return_counts=False, return_confidence=False,
        missing_origins=None, missing_dirs=None, missing_probs=None,
        ray_origins=None, ray_dirs=None, by_face=False,
        missing_weight=1.0, point_weights=None, return_weight_sums=False,
    ):
        """
        The scoring core of grade_from_segmentation(), operating on points
        that are ALREADY in the canonical frame with a damage probability
        attached to each.

        Split out because the two callers arrive with different things in
        hand: a single-frame caller has a depth image and backprojects
        (grade_from_segmentation above), while the multi-view fusion node
        has already accumulated canonical points across several views and
        must NOT re-backproject -- its points come from different cameras
        and are only mutually consistent because they were transformed
        through the fused world pose. Scoring here, after that transform,
        is what lets every view share one cell frame and removes the need
        for per-view symmetry resolution entirely.

        Args:
            canon_pts: (N, 3) points in the CAD model's canonical frame.
            point_probs: (N,) raw sigmoid damage probability per point.
            bounds_margin: meters; points farther than this outside the
                mesh's own bounding box are REJECTED, not scored. This is
                load-bearing, not hygiene: damage_cell_index() CLIPS an
                out-of-range bin index to the nearest valid bin, so
                without this filter a point tens of mm off the brick is
                silently counted as a full-authority observation of
                whichever corner cell it clips into. Measured on a real
                5-view capture: 55% of accumulated points fell outside
                the box (pose disagreement between views), and the
                (x_max, y_max, z_min) corner cell became a drain holding
                thousands of points that were 58-98% out-of-box -- two
                views appeared to "both see it clearly" when neither was
                observing that surface at all. A point that far from the
                brick says nothing about any cell, so it should vote for
                nothing rather than for its nearest neighbour.
            others: see grade_from_segmentation().

        Returns: see grade_from_segmentation().
        """
        # by_face swaps the 16 volumetric cells for the 40 surface
        # patches. Everything else -- filtering, thresholding, the
        # missing-material append, the score definition -- is identical,
        # which is the point: the two differ ONLY in what a point is
        # grouped with. See damage_face_index() for why grouping by
        # surface rather than by volume matters.
        n_out = self.num_damage_faces if by_face else self.num_damage_cells
        _bin_index = self.damage_face_index if by_face else self.damage_cell_index
        _ray_index = self.ray_entry_face_index if by_face else self.ray_entry_cell_index

        scores = np.full(n_out, np.nan, dtype=np.float64)
        counts = np.zeros(n_out, dtype=np.int64)
        confidence = np.full(n_out, np.nan, dtype=np.float64)
        # Per-bin sum of observation weights -- the EFFECTIVE sample
        # count. With point_weights supplied this is what fusion should
        # weight by instead of the raw count: 500 distant, grazing
        # points carry genuinely less information than 300 close,
        # face-on ones, and the raw count cannot express that.
        weight_sums = np.zeros(n_out, dtype=np.float64)

        canon_pts = np.asarray(canon_pts, dtype=np.float64)
        probs = np.asarray(point_probs, dtype=np.float64)
        if len(canon_pts) == 0:
            return _pack_seg_grade(scores, counts, confidence, return_counts,
                                   return_confidence, weight_sums, return_weight_sums)
        finite = np.isfinite(probs)
        canon_pts, probs = canon_pts[finite], probs[finite]
        if point_weights is not None:
            point_weights = np.asarray(point_weights, dtype=np.float64)[finite]
        if ray_origins is not None:
            ray_origins = np.asarray(ray_origins)[finite]
            ray_dirs = np.asarray(ray_dirs)[finite]
        if len(canon_pts) == 0:
            return _pack_seg_grade(scores, counts, confidence, return_counts,
                                   return_confidence, weight_sums, return_weight_sums)

        if bounds_margin is not None:
            bounds_min, bounds_max = self.mesh.bounds
            in_bounds = np.all(
                (canon_pts >= bounds_min - bounds_margin)
                & (canon_pts <= bounds_max + bounds_margin),
                axis=1,
            )
            canon_pts, probs = canon_pts[in_bounds], probs[in_bounds]
            if point_weights is not None:
                point_weights = point_weights[in_bounds]
            if ray_origins is not None:
                ray_origins = np.asarray(ray_origins)[in_bounds]
                ray_dirs = np.asarray(ray_dirs)[in_bounds]
            if len(canon_pts) == 0:
                return _pack_seg_grade(
                    scores, counts, confidence, return_counts, return_confidence,
                    weight_sums, return_weight_sums
                )

        if ray_origins is not None:
            # Ray placement: put each pixel in the cell its ray ENTERS on
            # the intact brick, rather than where its observed surface
            # happens to sit.
            #
            # More consistent with what the score actually measures. It is
            # an areal fraction over a cell's SURFACE -- "what fraction of
            # this cell reads damaged" -- and the surface in question is
            # the intact one. Depth placement puts a damaged pixel wherever
            # the receded surface ended up, which for a deep gouge can be a
            # different cell than the one the material was removed from.
            # It also makes ordinary and missing-material pixels obey one
            # rule instead of two, and removes depth noise from placement.
            #
            # The cost is that placement becomes entirely pose-dependent:
            # depth placement retains some independence from pose error,
            # this does not. Hence a switch rather than a replacement.
            keep_ray = np.asarray(ray_origins).shape[0] == len(canon_pts)
            if not keep_ray:
                raise ValueError(
                    f"ray_origins has {np.asarray(ray_origins).shape[0]} entries but "
                    f"there are {len(canon_pts)} points -- they must correspond 1:1 "
                    "(the bounds filter above must be applied to both or neither)."
                )
            idx, hit = _ray_index(ray_origins, ray_dirs)
            # A ray that misses the box has no cell; drop rather than let
            # damage_cell_index()'s clipping invent one.
            probs = probs[hit]
            idx = idx[hit]
            if point_weights is not None:
                point_weights = point_weights[hit]
        else:
            idx = _bin_index(canon_pts)
        damaged = probs > threshold

        # Optional missing-material contribution: pixels that see PAST
        # where the brick should be. They are appended as ordinary
        # observations of their ray-entry cell, so the cell's score stays
        # the same quantity (fraction of its observed pixels reading
        # damaged) whether or not this is enabled -- the switch changes
        # what is observable, not what is measured.
        n_missing = 0
        if missing_origins is not None and len(missing_origins) > 0:
            m_idx, m_hit = _ray_index(missing_origins, missing_dirs)
            if m_hit.any():
                m_probs = (
                    np.ones(int(m_hit.sum()))
                    if missing_probs is None
                    else np.asarray(missing_probs, dtype=np.float64)[m_hit]
                )
                idx = np.concatenate([idx, m_idx[m_hit]])
                # Seeing through the brick IS the damage evidence, so
                # these count as damaged regardless of the head's opinion
                # on background texture it was never reliably trained on.
                damaged = np.concatenate([damaged, np.ones(int(m_hit.sum()), dtype=bool)])
                probs = np.concatenate([probs, m_probs])
                n_missing = int(m_hit.sum())

        # Per-observation weight. Surface damage counts 1.0; missing
        # material counts `missing_weight`, so missing_weight=0.5 makes
        # observed surface damage count DOUBLE a material-loss ray.
        #
        # They are not equally trustworthy evidence. A surface-damage
        # pixel is a real measured return off the brick, placed by its
        # own depth. A missing-material ray is an inference -- "nothing
        # is here, and the silhouette says something should be" -- which
        # depends on the pose being right, and a misplaced silhouette
        # manufactures them along every edge. Weighting the two equally
        # lets the less reliable signal set a patch's score.
        #
        # The score stays a fraction, now a weighted one:
        #     sum(weight * damaged) / sum(weight)
        # so at missing_weight=1.0 it is exactly the old mean.
        if point_weights is None:
            weights = np.ones(len(probs), dtype=np.float64)
        else:
            # Missing-material rays were appended after the observed
            # points, so the supplied per-point weights cover the
            # leading block; the rays get a neutral 1.0 before the
            # missing_weight factor below.
            weights = np.ones(len(probs), dtype=np.float64)
            weights[: len(point_weights)] = point_weights
        if n_missing:
            weights[-n_missing:] *= float(missing_weight)

        calibrated = probs / np.maximum(pos_weight - (pos_weight - 1.0) * probs, 1e-9)

        for cell in range(n_out):
            sel = idx == cell
            n = int(sel.sum())
            counts[cell] = n
            if n < min_pts:
                continue
            w = weights[sel]
            weight_sums[cell] = float(w.sum())
            scores[cell] = float((w * damaged[sel]).sum() / max(w.sum(), 1e-9))
            confidence[cell] = float(np.clip(calibrated[sel], 0.0, 1.0).mean())
        return _pack_seg_grade(scores, counts, confidence, return_counts,
                               return_confidence, weight_sums, return_weight_sums)

    def visible_counts(self, pts, bounds_margin=0.005):
        """
        Per-cell observed-point counts only, skipping the signed-distance
        query grade() needs for scores. Use this when pts feed a per-frame
        visibility/confidence signal alongside scores computed separately
        (e.g. by grade_mesh()), since signed_distance is the expensive
        part of grade() and is wasted work when the per-frame score
        itself won't be used.

        Args:
            pts: (N, 3) canonical-frame points, e.g. from to_canonical().
            bounds_margin: see grade().

        Returns: (num_damage_cells,) int64 array of per-cell point counts.
        """
        pts = np.asarray(pts, dtype=np.float64)
        counts = np.zeros(self.num_damage_cells, dtype=np.int64)
        if len(pts) == 0:
            return counts
        if bounds_margin is not None:
            bounds_min, bounds_max = self.mesh.bounds
            in_bounds = np.all(
                (pts >= bounds_min - bounds_margin) & (pts <= bounds_max + bounds_margin),
                axis=1,
            )
            pts = pts[in_bounds]
            if len(pts) == 0:
                return counts
        idx = self.damage_cell_index(pts)
        for cell in range(self.num_damage_cells):
            counts[cell] = int((idx == cell).sum())
        return counts

    def grade_mesh(self, damaged_mesh_path, num_samples=50_000, min_pts=15,
                    surface_tol=0.003, inside_tol=0.002, return_counts=False):
        """
        True, view-independent damage score for one damaged brick instance,
        computed directly from its mesh against this grader's intact
        reference -- no camera, pose, or depth observation involved.

        Densely samples the damaged mesh's own surface and grades those
        points exactly like grade() does for observed points, so the same
        physical brick always yields the same score regardless of which
        (or how many) camera views of it happen to exist. Use this for
        synthetic-data labeling, where the true damaged geometry is known
        outright; reserve grade()/to_canonical() for grading a real,
        physically observed brick from an actual depth camera, where no
        such ground truth is available.

        Args:
            damaged_mesh_path: path to the damaged instance mesh, in the
                same local frame as this grader's intact mesh (i.e. not
                re-centered/re-oriented relative to it).
            num_samples: how many points to sample from the damaged mesh's
                surface. Cheap relative to a per-frame depth grade, and
                only needs to be done once per instance -- err high.

        Returns: same as grade().
        """
        mesh = trimesh.load(damaged_mesh_path, force="mesh", process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, num_samples)
        return self.grade(
            pts, surface_tol=surface_tol, inside_tol=inside_tol,
            min_pts=min_pts, return_counts=return_counts, bounds_margin=None,
        )

    def resolve_symmetry(self, rot_world, reference_rot, scores, counts=None):
        """
        Re-labels one snapshot's damage cell scores so they use the same
        physical-position convention as `reference_rot`.

        An unmarked rectangular brick looks identical after a 180-degree
        flip about any of its 3 principal axes, so a single-view pose
        estimate can't tell those apart from the true orientation. A
        snapshot that landed on a flipped orientation ends up with its
        damage cell scores silently permuted (see SYM_PERMUTATIONS): the
        same physical spot reads out under a different cell index than it
        would from a snapshot that didn't flip. This finds which of the 4
        possible flips (including "none") best explains this snapshot's
        rotation relative to a trusted reference, and undoes the resulting
        relabeling.

        Args:
            rot_world: (3, 3) this snapshot's estimated rotation, expressed
                in the same common frame as reference_rot (e.g. robot base,
                via T_world_camera @ pose_local).
            reference_rot: (3, 3) rotation to align against -- e.g. another
                snapshot's rot_world, taken as the trusted convention.
            scores: (num_damage_cells,) cell scores from this snapshot's
                grade() call.
            counts: optional (num_damage_cells,) cell point counts from
                grade().

        Returns:
            (scores_aligned, counts_aligned, sym_idx, residual_deg):
            counts_aligned is None if counts was None. sym_idx indexes
            SYM_ROTATIONS/SYM_PERMUTATIONS -- 0 means no flip was needed,
            useful for logging how often this is actually kicking in.
            residual_deg is how far off (in degrees) even the best-matching
            candidate still is from reference_rot -- large even after
            trying all 4 symmetry options means this snapshot's pose isn't
            a symmetry-flip of the reference, it's just a bad estimate.
            Callers can use this to reject outlier snapshots outright.
        """
        best_idx, best_dist = 0, np.inf
        for i, rot_sym in enumerate(self.SYM_ROTATIONS):
            candidate = rot_world @ rot_sym
            r = candidate.T @ reference_rot
            trace = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
            dist = np.arccos(trace)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        perm = self.SYM_PERMUTATIONS[best_idx]
        scores_aligned = np.asarray(scores)[perm]
        counts_aligned = np.asarray(counts)[perm] if counts is not None else None
        residual_deg = float(np.degrees(best_dist))
        return scores_aligned, counts_aligned, best_idx, residual_deg

    def estimate_cell_visibility(self, pose, margin=0.0):
        """
        Purely geometric estimate of which damage cells face the camera
        under a given pose -- no depth image needed, unlike grade()/
        visible_counts(), which need an actual observed point cloud.

        Approximates each cell's outward-facing direction as the vector
        from the mesh's bounding-box center to that cell's anchor point
        (damage_cell_vis_anchors()) -- a good approximation for a roughly
        box-shaped convex solid; not exact for edge/corner cells that
        face two or three outer surfaces at once, but good enough for a
        confidence signal, not a precise visibility mask. A cell counts
        as facing the camera if that direction, rotated (not translated)
        into camera frame by pose, points back toward the camera -- i.e.
        negative Z in the cam_pt = R @ obj_pt + t convention this whole
        module uses (see to_canonical()).

        Use this to give the learned damage pipeline (models/corners/
        corner_damage_predictor.py's Corner3DDamagePredictor) a real,
        non-learned per-cell confidence signal -- unlike grade()'s real
        point counts, a forward pass never touches a point cloud at all,
        so pose_estimator_node_online.py's "learned" damage_source
        currently has nothing to put in damage_cell_counts except a -1
        sentinel. Computing this straight from the model's own predicted
        pose (already reasonably accurate) is what makes a single
        learned-pipeline frame's prediction meaningfully fusable across
        multiple views via fuse_damage_cell_scores(), the same way
        grade()'s real counts already allow for the geometric pipeline.

        Args:
            pose: (4, 4) or (3, 4) object->camera transform, same
                convention as to_canonical().
            margin: cells whose facing direction is within this cosine
                margin of grazing (0 = exactly edge-on) are marked
                not-visible too, to exclude near-silhouette cells whose
                few real observed points would be unreliable anyway.
                0.0 = only exclude cells strictly facing away.

        Returns:
            (num_damage_cells,) float64 array in {0.0, 1.0} -- 1.0 if
            that cell's approximate outward direction faces the camera
            under this pose, 0.0 otherwise. Deliberately binary (not a
            continuous confidence), matching grade()'s own hard
            DAMAGE_CONFIDENCE_MIN_COUNT gate rather than trying to
            calibrate a continuous scale against real point counts'
            genuinely different units.
        """
        R = np.asarray(pose)[:3, :3]
        anchors = self.damage_cell_vis_anchors()  # (num_damage_cells, 3), object frame
        centroid = self.mesh.bounds.mean(axis=0)
        outward_obj = anchors - centroid[None, :]
        outward_obj = outward_obj / np.clip(
            np.linalg.norm(outward_obj, axis=1, keepdims=True), 1e-9, None
        )
        outward_cam = outward_obj @ R.T  # rotate each direction into camera frame
        visible = outward_cam[:, 2] < -margin
        return visible.astype(np.float64)

    def resolve_symmetry_by_scores(self, scores, reference_scores, counts=None, reference_counts=None):
        """
        Same idea as resolve_symmetry() -- re-labels `scores` to match
        `reference_scores`'s convention -- but matches by score similarity
        directly instead of comparing rotations. Use this when there's no
        rotation to trust for the alignment, e.g. comparing a learned
        model's pred_damage_scores (whose cell-index convention can be any
        of the 4 symmetric branches, independent of whatever its own pose
        head predicted -- see corner_damage_predictor.py) against a real
        brick's true grade_mesh() ground truth, which has no camera pose at
        all to compare against.

        Args:
            scores: (num_damage_cells,) scores to re-label, e.g. one
                frame's pred_damage_scores. May contain NaN.
            reference_scores: (num_damage_cells,) scores already in the
                trusted canonical convention, e.g. from grade_mesh(). May
                contain NaN.
            counts: optional (num_damage_cells,) per-cell weight for
                `scores` (e.g. a visibility count) -- cells with more
                weight influence which branch is picked more.
            reference_counts: optional (num_damage_cells,) same, for
                reference_scores.

        Returns:
            (scores_aligned, counts_aligned, best_idx, best_error):
            counts_aligned is None if counts was None. best_idx indexes
            SYM_ROTATIONS/SYM_PERMUTATIONS -- 0 means no flip was needed.
            best_error is the weighted MSE of the best-matching candidate
            against reference_scores, over cells valid (non-NaN, nonzero
            weight) in both -- large even after trying all 4 candidates
            means `scores` doesn't actually resemble reference_scores at
            all (wrong brick, bad prediction), not just a labeling
            mismatch. NaN if no cell was ever valid in both.
        """
        scores = np.asarray(scores, dtype=np.float64)
        reference_scores = np.asarray(reference_scores, dtype=np.float64)
        counts = np.ones_like(scores) if counts is None else np.asarray(counts, dtype=np.float64)
        ref_weight = (
            np.ones_like(reference_scores) if reference_counts is None
            else np.asarray(reference_counts, dtype=np.float64)
        )

        best_idx, best_error = 0, np.inf
        best_candidate, best_counts_candidate = scores, counts
        for i, perm in enumerate(self.SYM_PERMUTATIONS):
            candidate = scores[perm]
            counts_candidate = counts[perm]
            valid = ~np.isnan(candidate) & ~np.isnan(reference_scores)
            weight = np.where(valid, counts_candidate * ref_weight, 0.0)
            if weight.sum() <= 0:
                continue
            error = float(np.sum(weight * (candidate - reference_scores) ** 2) / weight.sum())
            if error < best_error:
                best_error = error
                best_idx = i
                best_candidate = candidate
                best_counts_candidate = counts_candidate

        counts_aligned = best_counts_candidate if counts is not None else None
        best_error = best_error if np.isfinite(best_error) else float("nan")
        return best_candidate, counts_aligned, best_idx, best_error


def _pack_seg_grade(scores, counts, confidence, return_counts, return_confidence,
                    weight_sums=None, return_weight_sums=False):
    """Shapes grade_from_segmentation()'s return to match grade()'s
    optional-extras convention (bare array, or a tuple growing in a fixed
    order) from its single set of computed arrays."""
    if not return_counts and not return_confidence and not return_weight_sums:
        return scores
    out = [scores]
    if return_counts:
        out.append(counts)
    if return_confidence:
        out.append(confidence)
    if return_weight_sums:
        out.append(weight_sums)
    return tuple(out)


def fuse_damage_cell_scores(scores_list, counts_list):
    """
    Combine per-damage-cell scores from multiple views into one set of
    cell scores. The views must already share the same cell-labeling
    convention -- run each one through BrickDamageGrader.resolve_symmetry()
    against a common reference first, or this will silently average
    unrelated cells.

    Each cell's views are averaged weighted by how many points supported
    that reading, so a well-observed view dominates a barely-observed one.

    Args:
        scores_list: sequence of (num_damage_cells,) score arrays (NaN allowed).
        counts_list: sequence of (num_damage_cells,) point-count arrays, same order.

    Returns:
        (fused_scores, total_counts): (num_damage_cells,) arrays. A cell
        with zero total points across every view stays NaN.
    """
    scores = np.stack(scores_list, axis=0).astype(np.float64)  # (V, num_damage_cells)
    counts = np.stack(counts_list, axis=0).astype(np.float64)  # (V, num_damage_cells)
    weights = np.where(np.isnan(scores), 0.0, counts)
    total_counts = weights.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        fused = np.sum(np.where(np.isnan(scores), 0.0, scores * weights), axis=0)
        fused = fused / total_counts
    fused[total_counts == 0] = np.nan
    return fused, total_counts.astype(np.int64)


def ground_mask(world_pts, band=0.005):
    """True for points that are NOT the board the brick stands on.

    The board is found from the data, not from the pose: it is the
    densest height level more than 40 mm below the top face (world +z is
    up). Everything less than `band` above it is dropped -- on bricks 1
    and 4 that was 11-14% of the cloud, sitting right where the side
    faces' bottom patches are.
    """
    z = world_pts[:, 2]
    top = np.percentile(z, 95)
    low = z[z < top - 0.04]
    if len(low) < 100:
        return np.ones(len(z), bool)
    hist, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.001, 0.001))
    board = edges[np.argmax(hist)]
    return z > board + band


def depth_face_scores(canon_pts, grader, deadband=0.003, saturate=0.015, min_pts=20,
                      box_margin=0.02, box=None):
    """Per face patch: mean depth of a view's points below the intact box,
    0 at `deadband` or less, 1 at `saturate` or more. NaN below `min_pts`.

    canon_pts are in the brick's canonical frame, through the view's own
    pose. Points further than `box_margin` outside the box are ignored.
    box=(lo, hi) measures depth against those bounds instead (e.g. the
    adjusted box); cropping and patch assignment stay on the intact box.
    """
    lo, hi = grader.mesh.bounds
    blo, bhi = (lo, hi) if box is None else box
    n = grader.num_damage_faces
    c = canon_pts[np.all((canon_pts > lo - box_margin) & (canon_pts < hi + box_margin), axis=1)]
    depth = np.clip(np.minimum(c - blo, bhi - c).min(axis=1), 0.0, None)
    ramp = np.clip((depth - deadband) / (saturate - deadband), 0.0, 1.0)
    idx = grader.damage_face_index(c)
    cnt = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        geo = np.bincount(idx, weights=ramp, minlength=n) / cnt
    geo[cnt < min_pts] = np.nan
    return geo


def learned_depth_combo(learned, geo, grader, threshold=0.25, neighbours=True):
    """pw11 decides where, depth decides how much -- for one view.

    Patches with learned >= threshold, plus their neighbours if
    `neighbours`, take the depth score; the rest keep the learned score.
    A flagged patch with no depth score (NaN) also keeps the learned one.
    """
    flag = np.nan_to_num(learned, nan=0.0) >= threshold
    if neighbours:
        adj = grader.damage_face_adjacency()
        for i in np.flatnonzero(flag.copy()):
            flag[list(adj[i])] = True
    return np.where(flag & ~np.isnan(geo), geo, learned)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone smoke test: grade damage cells from saved "
        "depth/mask/intrinsics/pose .npy files."
    )
    parser.add_argument("--mesh", default="pretrained/birck_simp2.obj")
    parser.add_argument("--depth", required=True, help=".npy depth map (H, W), meters")
    parser.add_argument("--mask", required=True, help=".npy detection mask (H, W)")
    parser.add_argument("--intr", required=True, help=".npy camera intrinsics (3, 3)")
    parser.add_argument("--pose", required=True, help=".npy object->camera pose (4, 4)")
    parser.add_argument("--surface-tol", type=float, default=0.003)
    parser.add_argument("--inside-tol", type=float, default=0.002)
    parser.add_argument("--min-pts", type=int, default=15)
    parser.add_argument(
        "--erode-px", type=int, default=0,
        help="shrink the mask by this many pixels to drop silhouette-edge noise",
    )
    args = parser.parse_args()

    depth = np.load(args.depth)
    mask = np.load(args.mask)
    intr = np.load(args.intr)
    pose = np.load(args.pose)

    pts = to_canonical(depth, mask, intr, pose, erode_px=args.erode_px)
    grader = BrickDamageGrader(args.mesh)
    scores, counts = grader.grade(
        pts,
        surface_tol=args.surface_tol,
        inside_tol=args.inside_tol,
        min_pts=args.min_pts,
        return_counts=True,
    )

    print(f"backprojected {len(pts)} canonical-frame points")
    print(f"grid: {grader.axis_bins[0]}x{grader.axis_bins[1]}x{grader.axis_bins[2]} "
          f"= {grader.num_damage_cells} damage cells (axis_bins are per x/y/z)")
    for i, (s, n) in enumerate(zip(scores, counts)):
        b0, b1, b2 = grader.damage_cell_bins[i]
        label = "nan (too few points)" if np.isnan(s) else f"{s:.3f}"
        print(f"damage cell {i} (bin x={b0} y={b1} z={b2}): {label}  [n={n}]")
