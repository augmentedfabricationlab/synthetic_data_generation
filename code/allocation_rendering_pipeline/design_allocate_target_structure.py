import os
import sys

from compas.geometry import Line, Frame, Box
from assembly_information_model import Part, Assembly

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from masonry_allocation import CellularizedPart, build_wall, connect_wall, label_facade, allocate
from preview_allocation import preview_allocation


# Make a basic brick part to use in the assembly.

brick_width = 0.12
brick_length = 0.25
brick_height = 0.065

tool_frame = Frame((0, 0, 0), (1, 0, 0), (0, -1, 0))
brick_geometry = Box(brick_length, brick_width, brick_height, Frame.worldXY(), 'box')
brick = Part.from_shape(brick_geometry, frame=tool_frame, name='brick')

# Design a container target structure to use in the allocation pipeline.

base_curve = Line((0, 0, 0), (2, 0, 0))

container_assembly = build_wall(brick, 
                                curve=base_curve, 
                                joint=0.02, 
                                bond_type="flemish", 
                                num_courses=20,
                                num_layers=1)

connect_wall(container_assembly)

# Label the facade of the container assembly with a design logic.

label_facade(container_assembly, 
             facade="front",
             style="rows",
             amount=4,
             alternate=True,
             gradient=False,
             floors=1,
             midpoint=0.5)

# Import the synthetic digital stock.

filename = "material_stock_simulated.json"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "allocation_data")
material_stock = Assembly.from_json(os.path.join(DATA, filename))

# Allocate from the material stock to the target structure.

allocated_assembly = allocate(container_assembly, material_stock)

# Export the allocated assembly to a JSON file.

allocated_name = "allocated_assembly.json"
allocated_assembly.to_json(os.path.join(DATA, allocated_name))

# Preview the labeled container next to the allocated target structure.

preview_allocation(container_assembly, allocated_assembly, os.path.join(DATA, "allocation_preview.png"))
