import sapien.core as sapien
import numpy as np
from pathlib import Path
import transforms3d as t3d
import sapien.physx as sapienp
import json
import os, re
import tempfile

from .actor_utils import Actor, ArticulationActor

# Stock RoboTwin library vs custom RoboDyna meshes. create_actor prefers dyna_assets.
OBJECTS_ROOT = Path("assets/objects")
DYNA_ASSETS_ROOT = Path("assets/dyna_assets")
DYNA_TEXTURES_ROOT = Path("assets/dyna_textures")
BACKGROUND_TEXTURE_ROOT = Path("assets/background_texture")


def resolve_model_dir(modelname: str) -> Path:
    """Return the object folder, preferring ``assets/dyna_assets`` over stock ``assets/objects``."""
    name = str(modelname)
    custom = DYNA_ASSETS_ROOT / name
    if custom.exists():
        return custom
    return OBJECTS_ROOT / name


def resolve_texture_file(texture_id: str, *, sphere: bool = False) -> Path | None:
    """Resolve a PNG/JPG for boxes, tables, and spheres.

    Custom files live in ``assets/dyna_textures``. Stock wall/table maps stay
    under ``assets/background_texture/{seen,unseen}/``.
    """
    stem = str(texture_id)
    names = [stem] if Path(stem).suffix else [f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg"]
    roots = [DYNA_TEXTURES_ROOT] if sphere else [DYNA_TEXTURES_ROOT, BACKGROUND_TEXTURE_ROOT]
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path
    return None


class UnStableError(Exception):

    def __init__(self, msg):
        super().__init__(msg)


def preprocess(scene, pose: sapien.Pose) -> tuple[sapien.Scene, sapien.Pose]:
    """Add entity to scene. Add bias to z axis if scene is not sapien.Scene."""
    if isinstance(scene, sapien.Scene):
        return scene, pose
    else:
        return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)


# create box
def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        tex = resolve_texture_file(texture_id)
        texturepath = str(tex) if tex is not None else f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    boxtype="default",
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
    )
    if boxtype == "default":
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data)


def create_hollow_box_with_holes(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=True,
    name="",
    hole_rows=None,
    hole_cols=None,
    hole_count=None,
    hole_size=None,
    wall_thickness=0.02,
    top_thickness=0.02,
    bottom_thickness=0.0,
    bar_thickness=0.02,
    top_transparent=False,
    panel_overhang=0.0,
) -> Actor:
    scene, pose = preprocess(scene, pose)
    x_half, y_half, z_half = half_size
    bottom_thickness = float(max(0.0, bottom_thickness))
    panel_overhang = float(max(0.0, panel_overhang))
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if is_static else "dynamic")

    if hole_rows is None and hole_cols is None:
        if hole_count is None:
            hole_rows = 5
            hole_cols = 3
        else:
            hole_cols = int(np.ceil(np.sqrt(hole_count)))
            hole_rows = int(np.ceil(hole_count / hole_cols))
    elif hole_rows is None:
        if hole_count is not None:
            hole_rows = int(np.ceil(hole_count / hole_cols))
        else:
            raise ValueError("hole_rows is missing")
    elif hole_cols is None:
        if hole_count is not None:
            hole_cols = int(np.ceil(hole_count / hole_rows))
        else:
            raise ValueError("hole_cols is missing")
    elif hole_count is not None and hole_rows * hole_cols != hole_count:
        raise ValueError("hole_count must equal hole_rows * hole_cols")

    if hole_rows < 1 or hole_cols < 1:
        raise ValueError("hole_rows and hole_cols must be >= 1")

    total_x = 2 * x_half
    total_y = 2 * y_half
    if hole_size is None:
        hole_space_x = total_x - (hole_cols + 1) * bar_thickness
        hole_space_y = total_y - (hole_rows + 1) * bar_thickness
        if hole_space_x <= 0 or hole_space_y <= 0:
            raise ValueError("Board is too small for the requested hole layout")
        hole_size = min(hole_space_x / hole_cols, hole_space_y / hole_rows)
    else:
        if hole_size <= 0:
            raise ValueError("hole_size must be > 0")
        if bar_thickness <= 0:
            raise ValueError("bar_thickness must be > 0")
        if hole_size * hole_cols + bar_thickness * (hole_cols + 1) > total_x:
            raise ValueError("Requested hole_size is too large for the board width")
        if hole_size * hole_rows + bar_thickness * (hole_rows + 1) > total_y:
            raise ValueError("Requested hole_size is too large for the board depth")

    # Side walls between optional floor and top lattice.
    # ±X walls span the full Y footprint; ±Y walls meet their inner faces
    # (half-length x_half - wall_thickness) so corners join with no gap.
    side_height = 2 * z_half - top_thickness - bottom_thickness
    if side_height <= 1e-6:
        raise ValueError("top_thickness + bottom_thickness exceed board height")
    side_z = -z_half + bottom_thickness + side_height / 2
    side_y_half_x = x_half - wall_thickness  # ±Y panels butt into ±X panels
    builder.add_box_collision(
        pose=sapien.Pose([-x_half + wall_thickness / 2, 0, side_z]),
        half_size=[wall_thickness / 2, y_half, side_height / 2],
        material=scene.default_physical_material,
    )
    builder.add_box_collision(
        pose=sapien.Pose([x_half - wall_thickness / 2, 0, side_z]),
        half_size=[wall_thickness / 2, y_half, side_height / 2],
        material=scene.default_physical_material,
    )
    builder.add_box_collision(
        pose=sapien.Pose([0, -y_half + wall_thickness / 2, side_z]),
        half_size=[side_y_half_x, wall_thickness / 2, side_height / 2],
        material=scene.default_physical_material,
    )
    builder.add_box_collision(
        pose=sapien.Pose([0, y_half - wall_thickness / 2, side_z]),
        half_size=[side_y_half_x, wall_thickness / 2, side_height / 2],
        material=scene.default_physical_material,
    )

    if hole_size is None:
        hole_size_x = (2 * x_half - (hole_cols + 1) * bar_thickness) / hole_cols
        hole_size_y = (2 * y_half - (hole_rows + 1) * bar_thickness) / hole_rows
        hole_size = min(hole_size_x, hole_size_y)
        if hole_size <= 0:
            raise ValueError("Board is too small for the requested hole layout")

    gap_x = (2 * x_half - hole_cols * hole_size) / (hole_cols + 1)
    gap_y = (2 * y_half - hole_rows * hole_size) / (hole_rows + 1)
    if gap_x < bar_thickness or gap_y < bar_thickness:
        raise ValueError("Requested hole_size is too large for the board top")

    # Top / bottom panels can overhang the walls slightly (crate lid / floor look).
    panel_x = x_half + panel_overhang
    panel_y = y_half + panel_overhang

    if bottom_thickness > 0.0:
        bottom_pose = sapien.Pose([0, 0, -z_half + bottom_thickness / 2])
        bottom_half = [panel_x, panel_y, bottom_thickness / 2]
        builder.add_box_collision(
            pose=bottom_pose,
            half_size=bottom_half,
            material=scene.default_physical_material,
        )

    # fill the top surface around the holes with a lattice of filled strips
    # horizontal strips: between hole rows and at the top/bottom margins
    # Outer ±Y strips absorb panel_overhang so the lid matches the floor.
    y = -y_half + gap_y / 2
    for i in range(hole_rows + 1):
        strip_y = y
        strip_hy = gap_y / 2
        if panel_overhang > 0.0 and i == 0:
            strip_hy = gap_y / 2 + panel_overhang
            strip_y = -panel_y + strip_hy
        elif panel_overhang > 0.0 and i == hole_rows:
            strip_hy = gap_y / 2 + panel_overhang
            strip_y = panel_y - strip_hy
        builder.add_box_collision(
            pose=sapien.Pose([0, strip_y, z_half - top_thickness / 2]),
            half_size=[panel_x, strip_hy, top_thickness / 2],
            material=scene.default_physical_material,
        )
        y += hole_size + gap_y

    # vertical strip fillers for each hole band
    y = -y_half + gap_y + hole_size / 2
    for _ in range(hole_rows):
        x = -x_half + gap_x / 2
        for j in range(hole_cols + 1):
            strip_x = x
            strip_hx = gap_x / 2
            if panel_overhang > 0.0 and j == 0:
                strip_hx = gap_x / 2 + panel_overhang
                strip_x = -panel_x + strip_hx
            elif panel_overhang > 0.0 and j == hole_cols:
                strip_hx = gap_x / 2 + panel_overhang
                strip_x = panel_x - strip_hx
            builder.add_box_collision(
                pose=sapien.Pose([strip_x, y, z_half - top_thickness / 2]),
                half_size=[strip_hx, hole_size / 2, top_thickness / 2],
                material=scene.default_physical_material,
            )
            x += hole_size + gap_x
        y += hole_size + gap_y

    board_color = color if color is not None else [1.0, 1.0, 1.0]
    if bottom_thickness > 0.0:
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, -z_half + bottom_thickness / 2]),
            half_size=[panel_x, panel_y, bottom_thickness / 2],
            material=board_color,
        )
    builder.add_box_visual(
        pose=sapien.Pose([-x_half + wall_thickness / 2, 0, side_z]),
        half_size=[wall_thickness / 2, y_half, side_height / 2],
        material=board_color,
    )
    builder.add_box_visual(
        pose=sapien.Pose([x_half - wall_thickness / 2, 0, side_z]),
        half_size=[wall_thickness / 2, y_half, side_height / 2],
        material=board_color,
    )
    builder.add_box_visual(
        pose=sapien.Pose([0, -y_half + wall_thickness / 2, side_z]),
        half_size=[side_y_half_x, wall_thickness / 2, side_height / 2],
        material=board_color,
    )
    builder.add_box_visual(
        pose=sapien.Pose([0, y_half - wall_thickness / 2, side_z]),
        half_size=[side_y_half_x, wall_thickness / 2, side_height / 2],
        material=board_color,
    )

    # Opaque top lattice via ActorBuilder; glass top is attached after build
    # (same RenderBodyComponent + transmission path as dispense_gummy tubes).
    top_strip_specs = []  # (local_pose, half_size)
    y = -y_half + gap_y / 2
    for i in range(hole_rows + 1):
        strip_y = y
        strip_hy = gap_y / 2
        if panel_overhang > 0.0 and i == 0:
            strip_hy = gap_y / 2 + panel_overhang
            strip_y = -panel_y + strip_hy
        elif panel_overhang > 0.0 and i == hole_rows:
            strip_hy = gap_y / 2 + panel_overhang
            strip_y = panel_y - strip_hy
        top_strip_specs.append((
            sapien.Pose([0, strip_y, z_half - top_thickness / 2]),
            [panel_x, strip_hy, top_thickness / 2],
        ))
        y += hole_size + gap_y

    y = -y_half + gap_y + hole_size / 2
    for _ in range(hole_rows):
        x = -x_half + gap_x / 2
        for j in range(hole_cols + 1):
            strip_x = x
            strip_hx = gap_x / 2
            if panel_overhang > 0.0 and j == 0:
                strip_hx = gap_x / 2 + panel_overhang
                strip_x = -panel_x + strip_hx
            elif panel_overhang > 0.0 and j == hole_cols:
                strip_hx = gap_x / 2 + panel_overhang
                strip_x = panel_x - strip_hx
            top_strip_specs.append((
                sapien.Pose([strip_x, y, z_half - top_thickness / 2]),
                [strip_hx, hole_size / 2, top_thickness / 2],
            ))
            x += hole_size + gap_x
        y += hole_size + gap_y

    if not top_transparent:
        for strip_pose, strip_half in top_strip_specs:
            builder.add_box_visual(
                pose=strip_pose, half_size=strip_half, material=board_color)

    builder.set_initial_pose(pose)
    entity = builder.build(name=name)

    if top_transparent and top_strip_specs:
        # Window glass (RT demos): light-blue tint + full transmission.
        # Default raster GUI ignores transmission (may look opaque).
        glass = sapien.render.RenderMaterial(base_color=[0.93, 0.97, 1.0, 0.25])
        glass.set_transmission(1.0)
        glass.set_transmission_roughness(0.0)
        glass.set_roughness(0.02)
        glass.set_metallic(0.0)
        try:
            glass.set_ior(1.45)
        except Exception:
            glass.ior = 1.45
        glass_entity = sapien.Entity()
        glass_entity.set_name(f"{name}_glass_top" if name else "glass_top")
        glass_entity.set_pose(pose)
        render_body = sapien.render.RenderBodyComponent()
        for strip_pose, strip_half in top_strip_specs:
            shape = sapien.render.RenderShapeBox(strip_half, glass)
            shape.set_local_pose(strip_pose)
            render_body.attach(shape)
        glass_entity.add_component(render_body)
        scene.add_entity(glass_entity)

    model_data = {
        "center": [0, 0, 0],
        "extents": half_size,
        "scale": half_size,
        "target_pose": [np.eye(4).tolist()],
        "contact_points_pose": [],
        "transform_matrix": np.eye(4).tolist(),
        "functional_matrix": [],
    }
    return Actor(entity, model_data)


def create_box_with_circular_holes(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=True,
    name="",
    hole_rows=None,
    hole_cols=None,
    hole_count=None,
    hole_size=None,
    wall_thickness=0.02,
    top_thickness=0.02,
    bottom_thickness=0.0,
    bar_thickness=0.02,
    hole_bands=48,
    hole_ring_width=0.006,
    hole_ring_height=0.0025,
    hole_ring_color=None,
    body_color=None,
) -> Actor:
    """Hollow box with a flat top deck and true circular hole cutouts.

    Unlike ``create_hollow_box_with_holes`` (square lattice bars), the play
    surface is a continuous plate with round openings — same layout math as the
    square grid (row-major hole centers), but openings are disks of diameter
    ``hole_size``.

    ``color`` tints the top deck; ``body_color`` tints the side walls (and
    optional floor). When ``body_color`` is None, walls use ``color``.
    """
    from shapely.geometry import Point, Polygon
    from trimesh.creation import extrude_polygon

    scene, pose = preprocess(scene, pose)
    x_half, y_half, z_half = half_size
    bottom_thickness = float(max(0.0, bottom_thickness))
    top_thickness = float(top_thickness)
    wall_thickness = float(wall_thickness)
    bar_thickness = float(bar_thickness)
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if is_static else "dynamic")

    if hole_rows is None and hole_cols is None:
        if hole_count is None:
            hole_rows = 3
            hole_cols = 3
        else:
            hole_cols = int(np.ceil(np.sqrt(hole_count)))
            hole_rows = int(np.ceil(hole_count / hole_cols))
    elif hole_rows is None:
        if hole_count is not None:
            hole_rows = int(np.ceil(hole_count / hole_cols))
        else:
            raise ValueError("hole_rows is missing")
    elif hole_cols is None:
        if hole_count is not None:
            hole_cols = int(np.ceil(hole_count / hole_rows))
        else:
            raise ValueError("hole_cols is missing")
    elif hole_count is not None and hole_rows * hole_cols != hole_count:
        raise ValueError("hole_count must equal hole_rows * hole_cols")

    if hole_rows < 1 or hole_cols < 1:
        raise ValueError("hole_rows and hole_cols must be >= 1")
    if hole_size is None or float(hole_size) <= 0:
        raise ValueError("hole_size (circular hole diameter) must be > 0")
    hole_size = float(hole_size)
    hole_radius = 0.5 * hole_size

    total_x = 2.0 * x_half
    total_y = 2.0 * y_half
    if hole_size * hole_cols + bar_thickness * (hole_cols + 1) > total_x:
        raise ValueError("Requested hole_size is too large for the board width")
    if hole_size * hole_rows + bar_thickness * (hole_rows + 1) > total_y:
        raise ValueError("Requested hole_size is too large for the board depth")

    gap_x = (total_x - hole_cols * hole_size) / (hole_cols + 1)
    gap_y = (total_y - hole_rows * hole_size) / (hole_rows + 1)
    if gap_x < bar_thickness or gap_y < bar_thickness:
        raise ValueError("Requested hole_size is too large for the board top")

    x_centers = np.linspace(
        -x_half + gap_x + hole_radius,
        x_half - gap_x - hole_radius,
        hole_cols,
    )
    y_centers = np.linspace(
        -y_half + gap_y + hole_radius,
        y_half - gap_y - hole_radius,
        hole_rows,
    )
    n_holes = hole_rows * hole_cols if hole_count is None else int(hole_count)
    hole_local = []
    for r, dy in enumerate(y_centers):
        for c, cx in enumerate(x_centers):
            if len(hole_local) >= n_holes:
                break
            hole_local.append((float(cx), float(dy)))

    # Side walls between optional floor and top deck.
    side_height = 2.0 * z_half - top_thickness - bottom_thickness
    if side_height <= 1e-6:
        raise ValueError("top_thickness + bottom_thickness exceed board height")
    side_z = -z_half + bottom_thickness + side_height / 2.0
    side_y_half_x = x_half - wall_thickness
    wall_specs = [
        ([-x_half + wall_thickness / 2.0, 0.0, side_z],
         [wall_thickness / 2.0, y_half, side_height / 2.0]),
        ([x_half - wall_thickness / 2.0, 0.0, side_z],
         [wall_thickness / 2.0, y_half, side_height / 2.0]),
        ([0.0, -y_half + wall_thickness / 2.0, side_z],
         [side_y_half_x, wall_thickness / 2.0, side_height / 2.0]),
        ([0.0, y_half - wall_thickness / 2.0, side_z],
         [side_y_half_x, wall_thickness / 2.0, side_height / 2.0]),
    ]
    for wall_p, wall_half in wall_specs:
        builder.add_box_collision(
            pose=sapien.Pose(wall_p),
            half_size=wall_half,
            material=scene.default_physical_material,
        )

    if bottom_thickness > 0.0:
        bottom_pose = sapien.Pose([0.0, 0.0, -z_half + bottom_thickness / 2.0])
        bottom_half = [x_half, y_half, bottom_thickness / 2.0]
        builder.add_box_collision(
            pose=bottom_pose,
            half_size=bottom_half,
            material=scene.default_physical_material,
        )

    # Flat top deck = rectangle minus circular buffers, extruded to a mesh.
    # Collision uses a thick slab; RT demos render triangle-mesh albedo poorly when
    # the same thick OBJ is also the visual (reads near-black). Match play_billiard:
    # collision mesh + a separate thin felt-like visual plate.
    resolution = max(24, int(hole_bands))
    poly = Polygon([
        (-x_half, -y_half), (x_half, -y_half), (x_half, y_half), (-x_half, y_half),
    ])
    for cx, cy in hole_local:
        cut = Point(cx, cy).buffer(hole_radius, resolution=resolution)
        poly = poly.difference(cut)
    if poly.is_empty:
        raise RuntimeError("circular hole cutouts removed the entire board top")
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    coll_mesh = extrude_polygon(poly, height=top_thickness)
    coll_mesh.apply_translation([0.0, 0.0, -0.5 * top_thickness])
    try:
        coll_mesh.fix_normals()
    except Exception:
        pass
    coll_path = os.path.join(
        tempfile.gettempdir(),
        f"whack_board_circular_holes_coll_{hole_rows}x{hole_cols}_{name or 'board'}.obj",
    )
    coll_mesh.export(coll_path, include_color=False, include_normals=True)

    # Thin visual plate flush with the deck top (billiard-style felt lid).
    vis_thickness = float(min(0.004, max(0.002, 0.15 * top_thickness)))
    vis_mesh = extrude_polygon(poly, height=vis_thickness)
    vis_mesh.apply_translation([0.0, 0.0, -0.5 * vis_thickness])
    try:
        vis_mesh.fix_normals()
    except Exception:
        pass
    vis_path = os.path.join(
        tempfile.gettempdir(),
        f"whack_board_circular_holes_vis_{hole_rows}x{hole_cols}_{name or 'board'}.obj",
    )
    vis_mesh.export(vis_path, include_color=False, include_normals=True)

    board_color = list(color) if color is not None else [0.10, 0.45, 0.22]
    if len(board_color) == 3:
        board_rgba = [*board_color, 1.0]
    else:
        board_rgba = board_color[:4]
    render_mat = sapien.render.RenderMaterial(base_color=board_rgba)

    wall_color = list(body_color) if body_color is not None else list(board_color)
    wall_rgb = wall_color[:3]

    for wall_p, wall_half in wall_specs:
        builder.add_box_visual(
            pose=sapien.Pose(wall_p), half_size=wall_half, material=wall_rgb)
    if bottom_thickness > 0.0:
        builder.add_box_visual(
            pose=sapien.Pose([0.0, 0.0, -z_half + bottom_thickness / 2.0]),
            half_size=[x_half, y_half, bottom_thickness / 2.0],
            material=wall_rgb,
        )

    top_z = z_half - top_thickness / 2.0
    top_pose = sapien.Pose([0.0, 0.0, top_z])
    builder.add_nonconvex_collision_from_file(filename=coll_path, pose=top_pose)
    # Visual plate: top face flush with board top (z_half).
    vis_pose = sapien.Pose([0.0, 0.0, z_half - 0.5 * vis_thickness])
    builder.add_visual_from_file(
        filename=vis_path, pose=vis_pose, material=render_mat)

    # Black rim rings around each hole (visual only) so openings read clearly.
    ring_width = float(max(0.0, hole_ring_width))
    ring_height = float(max(0.0, hole_ring_height))
    if ring_width > 1e-6 and ring_height > 1e-6 and hole_local:
        import trimesh

        ring_meshes = []
        for cx, cy in hole_local:
            outer = Point(cx, cy).buffer(
                hole_radius + ring_width, resolution=resolution)
            inner = Point(cx, cy).buffer(hole_radius, resolution=resolution)
            annulus = outer.difference(inner)
            if annulus.is_empty:
                continue
            geoms = list(annulus.geoms) if annulus.geom_type == "MultiPolygon" else [annulus]
            for geom in geoms:
                if geom.is_empty or geom.area <= 0:
                    continue
                rm = extrude_polygon(geom, height=ring_height)
                rm.apply_translation([0.0, 0.0, -0.5 * ring_height])
                ring_meshes.append(rm)
        if ring_meshes:
            ring_mesh = (
                ring_meshes[0] if len(ring_meshes) == 1
                else trimesh.util.concatenate(ring_meshes)
            )
            try:
                ring_mesh.fix_normals()
            except Exception:
                pass
            ring_path = os.path.join(
                tempfile.gettempdir(),
                f"whack_board_hole_rings_{hole_rows}x{hole_cols}_{name or 'board'}.obj",
            )
            ring_mesh.export(ring_path, include_color=False, include_normals=True)
            ring_rgb = list(hole_ring_color) if hole_ring_color is not None else [0.04, 0.04, 0.04]
            ring_rgba = [*ring_rgb[:3], 1.0]
            ring_mat = sapien.render.RenderMaterial(base_color=ring_rgba)
            try:
                ring_mat.set_metallic(0.0)
                ring_mat.set_roughness(0.95)
            except Exception:
                pass
            # Sit the lip on the deck top so the rim reads from above.
            ring_pose = sapien.Pose([0.0, 0.0, z_half + ring_height / 2.0])
            builder.add_visual_from_file(
                filename=ring_path, pose=ring_pose, material=ring_mat)

    builder.set_initial_pose(pose)
    entity = builder.build(name=name)

    model_data = {
        "center": [0, 0, 0],
        "extents": half_size,
        "scale": half_size,
        "target_pose": [np.eye(4).tolist()],
        "contact_points_pose": [],
        "transform_matrix": np.eye(4).tolist(),
        "functional_matrix": [],
        "hole_centers_local": hole_local,
        "hole_size": hole_size,
    }
    return Actor(entity, model_data)


# create spere
def create_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeSphere(radius=radius, material=scene.default_physical_material))

    # Add texture (classic soccer panel map under assets/dyna_textures/<id>.png).
    if texture_id is not None:
        tex = resolve_texture_file(texture_id, sphere=True)
        texturepath = str(tex) if tex is not None else ""
        if texturepath and os.path.isfile(texturepath):
            texture2d = sapien.render.RenderTexture2D(texturepath)
            material = sapien.render.RenderMaterial()
            material.set_base_color_texture(texture2d)
            material.base_color = [1, 1, 1, 1]
            material.metallic = 0.05
            material.roughness = 0.55
        else:
            material = sapien.render.RenderMaterial(
                base_color=[*(color[:3] if color is not None else (0.95, 0.95, 0.95)), 1]
            )
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeSphere(radius=radius, material=material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create cylinder
def create_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=scene.default_physical_material,
        ))

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=sapien.render.RenderMaterial(base_color=[*color[:3], 1]),
        ))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create box
def create_visual_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])))

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_table(
        scene,
        pose: sapien.Pose,
        length: float,
        width: float,
        height: float,
        thickness=0.1,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    # Tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])  # Center the tabletop at z=0
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(
        pose=tabletop_pose,
        half_size=tabletop_half_size,
        material=scene.default_physical_material,
    )

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        tex = resolve_texture_file(texture_id)
        texturepath = str(tex) if tex is not None else f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
    else:
        builder.add_box_visual(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=color,
        )

    # Table legs (x4)
    leg_spacing = 0.1
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_spacing / 2)
            y = j * (width / 2 - leg_spacing / 2)
            table_leg_pose = sapien.Pose([x, y, -height / 2 - 0.002])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2 - 0.002]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    return table


# create obj model
def create_obj(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
        no_collision=False,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = resolve_model_dir(modelname)
    if model_id is None:
        file_name = modeldir / "textured.obj"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"textured{model_id}.obj"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if not no_collision:
        if convex == True:
            builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
        else:
            builder.add_nonconvex_collision_from_file(filename=str(file_name), scale=scale)

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


# create glb model
def create_glb(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = resolve_model_dir(modelname)
    if model_id is None:
        file_name = modeldir / "base.glb"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"base{model_id}.glb"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(file_name),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


def get_glb_or_obj_file(modeldir, model_id):
    modeldir = Path(modeldir)
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        if model_id is None:
            file = modeldir / "textured.obj"
        else:
            file = modeldir / f"textured{model_id}.obj"
    return file


def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=0,
        scale_mult=1.0,
) -> Actor:
    # scale_mult: optional per-spawn size multiplier applied on top of the asset's authored
    # model_data["scale"]. Lets a task request a smaller/larger instance of a SHARED asset without
    # editing it (e.g. a small apple from the regular 035_apple). It scales the mesh AND the stored
    # scale so contact/functional points (which multiply by model_data["scale"]) stay consistent.
    scene, pose = preprocess(scene, pose)
    modeldir = resolve_model_dir(modelname)

    if model_id is None:
        json_file_path = modeldir / "model_data.json"
    else:
        json_file_path = modeldir / f"model_data{model_id}.json"

    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)

    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)

    if not collision_file.exists() or not visual_file.exists():
        print(modelname, "is not exist model file!")
        return None

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
        # Some assets (e.g. 038_milk-box) ship ``scale: null`` — treat as unit.
        if scale is None:
            scale = [1.0, 1.0, 1.0]
            model_data = dict(model_data)
            model_data["scale"] = list(scale)
    except:
        model_data = None
        scale = [1.0, 1.0, 1.0]

    # scale_mult: float (uniform) OR a 3-sequence (per-axis, e.g. (1,1.6,1) to thicken one axis).
    try:
        _mult = [float(m) for m in scale_mult]            # per-axis sequence
    except TypeError:
        _mult = [float(scale_mult)] * 3                   # uniform scalar
    if any(m != 1.0 for m in _mult):
        scale = [float(s) * m for s, m in zip(scale, _mult)]
        if model_data is not None:
            model_data = dict(model_data)
            model_data["scale"] = scale   # keep points consistent with the scaled mesh

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_name(modelname)
    mesh.set_pose(pose)
    return Actor(mesh, model_data)


# create urdf model
def create_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = resolve_model_dir(modelname)
    json_file_path = modeldir / "model_data.json"
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        loader.scale = model_data["scale"][0]
    except:
        model_data = None

    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object: sapien.Articulation = loader.load(str(modeldir / "mobility.urdf"))

    object.set_root_pose(pose)
    object.set_name(modelname)
    return ArticulationActor(object, model_data)


def create_sapien_urdf_obj(
    scene,
    pose: sapien.Pose,
    modelname: str,
    scale=1.0,
    modelid: int = None,
    fix_root_link=False,
) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = resolve_model_dir(modelname)
    if modelid is not None:
        model_list = [model for model in modeldir.iterdir() if model.is_dir() and model.name != "visual"]

        def extract_number(filename):
            match = re.search(r"\d+", filename.name)
            return int(match.group()) if match else 0

        model_list = sorted(model_list, key=extract_number)

        if modelid >= len(model_list):
            is_find = False
            for model in model_list:
                if modelid == int(model.name):
                    modeldir = model
                    is_find = True
                    break
            if not is_find:
                raise ValueError(f"modelid {modelid} is out of range for {modelname}.")
        else:
            modeldir = model_list[modelid]
    json_file = modeldir / "model_data.json"

    if json_file.exists():
        with open(json_file, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
        trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
    else:
        model_data = None
        trans_mat = np.eye(4)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object = loader.load_multiple(str(modeldir / "mobility.urdf"))[0][0]

    pose_mat = pose.to_transformation_matrix()
    pose = sapien.Pose(
        p=pose_mat[:3, 3] + trans_mat[:3, 3],
        q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
    )
    object.set_pose(pose)

    if model_data is not None:
        if "init_qpos" in model_data and len(model_data["init_qpos"]) > 0:
            object.set_qpos(np.array(model_data["init_qpos"]))
        if "mass" in model_data and len(model_data["mass"]) > 0:
            for link in object.get_links():
                link.set_mass(model_data["mass"].get(link.get_name(), 0.1))

        bounding_box_file = modeldir / "bounding_box.json"
        if bounding_box_file.exists():
            bounding_box = json.load(open(bounding_box_file, "r", encoding="utf-8"))
            model_data["extents"] = (np.array(bounding_box["max"]) - np.array(bounding_box["min"])).tolist()
    object.set_name(modelname)
    return ArticulationActor(object, model_data)
