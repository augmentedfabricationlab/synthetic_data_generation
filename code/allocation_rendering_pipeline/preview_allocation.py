import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import ConvexHull
from compas.geometry import Transformation, cross_vectors, normalize_vector, dot_vectors, transform_points


# 0 = good (green), 1 = bad (red); grey = no value
GREEN = (192 / 255, 192 / 255, 0 / 255)
RED = (220 / 255, 0 / 255, 40 / 255)
cmap = LinearSegmentedColormap.from_list("good_bad", [GREEN, RED])


def outline(points, right):
    """Outline of 3D points as seen from the front."""
    uv = [(dot_vectors(pt, right), pt[2]) for pt in points]
    return [uv[i] for i in ConvexHull(uv).vertices]


def draw(ax, polygons, values, title):
    colors = [cmap(v) if v is not None else "lightgrey" for v in values]
    ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="black", linewidths=0.2))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


def preview_allocation(container_assembly, allocated_assembly, out_path=None):
    """Front view of the labeled container (front_label) next to the allocated wall (front cells' damage_score)."""

    # view direction = the front face's outward normal (front = local minus side of front_axis)
    front_parts = [p for p in container_assembly.parts() if p.attributes.get("front_axis")]
    normals = [[-c for c in getattr(p.frame, p.attributes["front_axis"] + "axis")] for p in front_parts]
    normal = normalize_vector([sum(n[i] for n in normals) for i in range(3)])
    right = normalize_vector(cross_vectors([0, 0, 1], normal))

    # container: one outline per front-exposed slot, coloured by its label
    container_polygons, container_values = [], []
    for part in container_assembly.parts():
        if part.attributes.get("front_axis"):
            T = Transformation.from_frame(part.frame)
            container_polygons.append(outline(part.mesh.transformed(T).vertices_attributes("xyz"), right))
            container_values.append(part.attributes.get("front_label"))

    # allocated: every front-facing cell of a front-exposed brick, coloured by its damage score
    allocated_polygons, allocated_values = [], []
    for part in allocated_assembly.parts():
        if not part.attributes.get("front_axis"):
            continue
        T = Transformation.from_frame(part.frame)
        net = getattr(part, "cell_network", None)
        if net is None:  # unlabeled slot carried over without a stock brick
            allocated_polygons.append(outline(part.mesh.transformed(T).vertices_attributes("xyz"), right))
            allocated_values.append(None)
            continue
        for ckey in net.active_cells():
            face = net.cell_attribute(ckey, "face")  # e.g. "y-"
            sign = -1 if face[1] == "-" else 1
            face_normal = [sign * c for c in getattr(part.frame, face[0] + "axis")]
            if dot_vectors(face_normal, normal) > 0.5:
                points = transform_points(net.cell_points(ckey), T)
                allocated_polygons.append(outline(points, right))
                allocated_values.append(net.cell_attribute(ckey, "damage_score"))

    fig, (ax_container, ax_allocated) = plt.subplots(1, 2, figsize=(14, 7))
    draw(ax_container, container_polygons, container_values, "Labeled container structure")
    draw(ax_allocated, allocated_polygons, allocated_values, "Allocated target structure")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), ax=[ax_container, ax_allocated], shrink=0.6)
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.show()
