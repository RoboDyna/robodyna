import numpy as np
import os
from pathlib import Path
import yaml

import sapien
from transforms3d.euler import euler2quat
from envs.utils.create_actor import create_actor
from envs.utils.rand_create_actor import rand_pose

_BACKGROUND_TEXTURE_ROOT = Path(__file__).resolve().parents[2] / "bench_assets" / "backgrounds"
_BACKGROUND_TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg")

def get_quat_euler(euler):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw) in degrees.
    """
    return euler2quat(*[np.deg2rad(d) for d in euler])
  


def _resolve_texture_path(texture_id: str | Path | None, object_type: str | None = None) -> Path:
    """
    Resolve a texture path for SAPIEN render materials.

    ``texture_id`` is translated to a file stem, e.g. ``0`` resolves to
    ``0.png``. When ``object_type`` is provided, it is used as the subdirectory
    name under the background texture root. If ``texture_id`` is ``None``, a
    random texture is chosen from that subdirectory. Explicit existing file
    paths are accepted directly.
    """
    if texture_id is None:
        if not object_type:
            raise ValueError("texture_id cannot be None when object_type is not provided.")

        texture_dir = _BACKGROUND_TEXTURE_ROOT / object_type
        candidates = []
        for extension in _BACKGROUND_TEXTURE_EXTENSIONS:
            candidates.extend(sorted(texture_dir.glob(f"*{extension}")))
        if not candidates:
            raise FileNotFoundError(f"No textures found in background directory {texture_dir}.")
        return Path(np.random.choice(candidates))

    texture_path = Path(texture_id)
    if texture_path.exists():
        return texture_path

    texture_stem = texture_path.stem if texture_path.suffix else str(texture_id)

    if object_type:
        for extension in _BACKGROUND_TEXTURE_EXTENSIONS:
            background_texture_path = _BACKGROUND_TEXTURE_ROOT / object_type / f"{texture_stem}{extension}"
            if background_texture_path.exists():
                return background_texture_path

    for extension in _BACKGROUND_TEXTURE_EXTENSIONS:
        background_texture_path = _BACKGROUND_TEXTURE_ROOT / f"{texture_stem}{extension}"
        if background_texture_path.exists():
            return background_texture_path

    search_root = _BACKGROUND_TEXTURE_ROOT / object_type if object_type else _BACKGROUND_TEXTURE_ROOT
    raise FileNotFoundError(
        f"Could not resolve texture_id={texture_id!r} in {search_root} "
        f"with extensions {', '.join(_BACKGROUND_TEXTURE_EXTENSIONS)}."
    )


def _make_texture_material(
    texture_id: str | Path | None,
    object_type: str | None = None,
    base_color=(1, 1, 1, 1),
    metallic: float = 0.1,
    roughness: float = 0.3,
):
    material = sapien.render.RenderMaterial()
    texture2d = sapien.render.RenderTexture2D(str(_resolve_texture_path(texture_id, object_type=object_type)))
    material.set_base_color_texture(texture2d)
    try:
        material.set_diffuse_texture(texture2d)
    except Exception:
        pass
    material.base_color = list(base_color)
    material.metallic = float(metallic)
    material.roughness = float(roughness)
    return material

def _get_scene(scene_or_task):
    return scene_or_task.scene if hasattr(scene_or_task, "scene") else scene_or_task

def _iter_render_shape_holders(scene_or_task, obj):
    """Yield entities/components that may own SAPIEN render shapes."""
    scene = _get_scene(scene_or_task)
    if isinstance(obj, str):
        found = None
        for get_all in ("get_all_actors", "get_all_articulations"):
            try:
                candidates = getattr(scene, get_all)()
            except Exception:
                continue
            for candidate in candidates:
                try:
                    if candidate.get_name() == obj:
                        found = candidate
                        break
                except Exception:
                    continue
            if found is not None:
                break
        if found is None:
            raise ValueError(f"Object named {obj!r} was not found in the SAPIEN scene.")
        obj = found

    if hasattr(obj, "actor"):
        obj = obj.actor

    yield obj

    try:
        links = obj.get_links()
    except Exception:
        links = []

    for link in links:
        yield link
        for attr_name in ("entity", "owner", "parent", "get_entity"):
            try:
                owner = getattr(link, attr_name)
                owner = owner() if callable(owner) else owner
            except Exception:
                continue
            if owner is not None:
                yield owner

def _iter_render_shapes(scene_or_task, obj):
    """Yield render shapes for plain actors and articulation links."""
    for holder in _iter_render_shape_holders(scene_or_task, obj):
        components = []

        try:
            components.extend(holder.get_components())
        except Exception:
            pass

        try:
            holder_components = getattr(holder, "components")
            holder_components = holder_components() if callable(holder_components) else holder_components
            if holder_components is not None:
                components.extend(holder_components)
        except Exception:
            pass

        try:
            render_component = holder.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_component is not None:
                components.append(render_component)
        except Exception:
            pass

        if isinstance(holder, sapien.render.RenderBodyComponent):
            components.append(holder)

        for component in components:
            if any(hasattr(component, attr_name) for attr_name in ("set_material", "material", "set_render_material", "render_material", "set_texture")):
                yield component

            for attr_name in ("render_shapes", "visual_shapes", "shapes"):
                try:
                    shapes = getattr(component, attr_name)
                    shapes = shapes() if callable(shapes) else shapes
                except Exception:
                    continue
                if shapes is None:
                    continue
                for shape in shapes:
                    yield shape

            for method_name in ("get_render_shapes", "get_visual_shapes", "get_shapes"):
                try:
                    shapes = getattr(component, method_name)()
                except Exception:
                    continue
                if shapes is None:
                    continue
                for shape in shapes:
                    yield shape

def _get_render_material_texture(material):
    try:
        return material.get_base_color_texture()
    except Exception:
        pass

    try:
        return material.base_color_texture
    except Exception:
        pass

    try:
        return material.get_diffuse_texture()
    except Exception:
        pass

    try:
        return material.diffuse_texture
    except Exception:
        return None

def _set_render_item_texture(item, material) -> bool:
    if not hasattr(item, "set_texture"):
        return False

    texture2d = _get_render_material_texture(material)
    if texture2d is None:
        return False

    updated = False
    texture_names = (
        "base_color_texture",
        "diffuse_texture",
        "baseColorTexture",
        "diffuse",
        "base_color",
        "BaseColor",
    )
    for texture_name in texture_names:
        try:
            item.set_texture(texture_name, texture2d)
            updated = True
        except Exception:
            pass

    return updated

def _copy_render_material_properties(target_material, source_material) -> bool:
    texture2d = _get_render_material_texture(source_material)
    if texture2d is None:
        return False

    copied_texture = False
    for method_name in ("set_base_color_texture", "set_diffuse_texture"):
        try:
            getattr(target_material, method_name)(texture2d)
            copied_texture = True
        except Exception:
            pass

    for attr_name in ("base_color_texture", "diffuse_texture"):
        try:
            setattr(target_material, attr_name, texture2d)
            copied_texture = True
        except Exception:
            pass

    if not copied_texture:
        return False

    for attr_name in ("base_color", "metallic", "roughness"):
        try:
            setattr(target_material, attr_name, getattr(source_material, attr_name))
        except Exception:
            pass

    return True

def _mutate_existing_render_material(item, material) -> bool:
    for attr_name in ("get_material", "material", "render_material"):
        try:
            existing_material = getattr(item, attr_name)
            existing_material = existing_material() if callable(existing_material) else existing_material
        except Exception:
            continue
        if existing_material is None:
            continue
        if _copy_render_material_properties(existing_material, material):
            return True

    return False

def _set_render_item_material(item, material) -> bool:
    if _set_render_item_texture(item, material):
        return True

    for method_name in ("set_material", "set_render_material"):
        try:
            getattr(item, method_name)(material)
            return True
        except Exception:
            pass

    for attr_name in ("material", "render_material"):
        try:
            setattr(item, attr_name, material)
            return True
        except Exception:
            pass

    if _mutate_existing_render_material(item, material):
        return True

    return False

def _set_render_shape_material(shape, material) -> int:
    updated = 0
    if _set_render_item_material(shape, material):
        updated += 1

    # GLB/mesh visuals may keep the visible materials on triangle-mesh
    # parts instead of only on the parent RenderShape.
    visited_parts = []
    for attr_name in ("parts", "get_parts"):
        try:
            parts = getattr(shape, attr_name)
            parts = parts() if callable(parts) else parts
        except Exception:
            continue
        if parts is None:
            continue
        for part in parts:
            if any(part is visited_part for visited_part in visited_parts):
                continue
            visited_parts.append(part)
            if _set_render_item_material(part, material):
                updated += 1

    return updated

def change_object_texture(
    scene_or_task,
    obj,
    texture_id: str | Path | None,
    object_type: str | None = None,
    base_color=(1, 1, 1, 1),
    metallic: float = 0.1,
    roughness: float = 0.3,
    refresh_render: bool = False,
) -> int:
    """
    Change an existing object's visual texture in a SAPIEN scene.

    ``scene_or_task`` can be a raw SAPIEN scene or a task object with a
    ``scene`` attribute. ``obj`` can be a raw SAPIEN entity/articulation, this
    repo's Actor or ArticulationActor wrapper, or a scene object name. Returns
    the number of render materials updated. When ``texture_id`` is ``None``,
    a random texture is selected from ``object_type``.
    """
    scene = _get_scene(scene_or_task)
    material = _make_texture_material(
        texture_id=texture_id,
        object_type=object_type,
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
    )

    updated = 0
    visited_shapes = []
    for shape in _iter_render_shapes(scene_or_task, obj):
        if any(shape is visited_shape for visited_shape in visited_shapes):
            continue
        visited_shapes.append(shape)
        updated += _set_render_shape_material(shape, material)

    if updated == 0:
        raise RuntimeError(f"No render materials were updated for object {obj!r}.")

    if refresh_render:
        scene.update_render()

    return updated




def get_actor_boundingbox(entity):
    all_points = []
    try:
        actor = entity.actor
    except:
        actor = entity
    
    entity_mat = actor.pose.to_transformation_matrix()
    for comp in actor.get_components():
        # Check if component has collision shapes (RigidBodyComponent)
        if hasattr(comp, 'get_collision_shapes'):
            for shape in comp.get_collision_shapes():
                # 1. Get local vertices [N, 3]
                try:
                    local_v = shape.get_vertices() 
                except:
                    hs = shape.half_size  # e.g., [0.1, 0.1, 0.1]

                    # 2. Define the 8 corners in the shape's local frame
                    local_v = np.array([[x, y, z] for x in [-hs[0], hs[0]] 
                                for y in [-hs[1], hs[1]] 
                                for z in [-hs[2], hs[2]]])
                try:
                    local_v *= shape.scale
                except:
                    pass
                # 2. Get shape's offset relative to the entity center
                shape_mat = shape.get_local_pose().to_transformation_matrix()
                
                # 3. Combine: Entity World * Shape Local
                world_mat = entity_mat @ shape_mat
                
                # 4. Transform points to world space
                homo_v = np.pad(local_v, ((0, 0), (0, 1)), constant_values=1)
                world_v = (world_mat @ homo_v.T).T[:, :3]
                all_points.append(world_v)

    if not all_points:
        return None, None

    points_cloud = np.vstack(all_points)
    return points_cloud.min(axis=0), points_cloud.max(axis=0)

def get_actor_boundingbox_urdf(entity):
    all_vertices = []
    for link in entity.actor.get_links():
        for shape in link.get_collision_shapes():
            # Get geometry and its local transformation
            world_pose = link.get_pose() * shape.get_local_pose()
            mat = world_pose.to_transformation_matrix()
            # 1. For Convex Meshes (the error you hit)
            v = shape.get_vertices()  # Access vertices directly
            v = v * shape.get_scale()
            v_world = (mat[:3, :3] @ v.T).T + mat[:3, 3]
            all_vertices.append(v_world)

    # Note: For Spheres/Capsules, you may need to approximate 
    # using their radius/half_length properties.
    merged = np.vstack(all_vertices)
    return np.min(merged, axis=0), np.max(merged, axis=0)

def get_position_limits(surface_obj, boundary_thr = 0.15, 
                       robot_reach_thr = 0.6,
                       arm_x_pose = 0.15, side=None):
    # Get generation limits
    # Assumption is that the robot is centered with respect to the the surface area
    if isinstance(boundary_thr, float):
        x_thr = y_thr = boundary_thr
    else:
        x_thr = boundary_thr[0]
        y_thr = boundary_thr[1]

    try:
        actor = surface_obj.actor
    except:
        actor = surface_obj
    table_bb = get_actor_boundingbox(actor)

    side_to_place = side or np.random.choice(["left", "right"])

    if side_to_place == "left":
        xmin = max((table_bb[0][0] + x_thr),  (-arm_x_pose - robot_reach_thr))
        xmax = min((-arm_x_pose + robot_reach_thr), arm_x_pose)
        xlim=[xmin, xmax ] # 0.9 is the range of reach for the robot arm
    else:
        xmax = min((table_bb[1][0] - x_thr),  (arm_x_pose + robot_reach_thr))
        xmin = max((arm_x_pose - robot_reach_thr), -arm_x_pose)
        xlim=[xmin, xmax] # 0.9 is the range of reach for the robot arm
    ylim= [table_bb[0][1] + y_thr, table_bb[1][1]- y_thr]

    return xlim, ylim, side_to_place

def get_random_valid_placement(table_bounds, object_bounds, new_w, new_h):
    tx1, tx2 = table_bounds[0]
    ty1,ty2 = table_bounds[1]
    
    # 1. Generate candidate X and Y coordinates from all edges
    # We also add a small buffer (0.01) if you want objects not to touch exactly
    x_coords = sorted(list(set([tx1, tx2 - new_w] + [o for o in object_bounds if o <= tx2 - new_w] + [o for o in object_bounds if o <= tx2 - new_w])))
    y_coords = sorted(list(set([ty1, ty2 - new_h] + [o for o in object_bounds if o <= ty2 - new_h] + [o for o in object_bounds if o <= ty2 - new_h])))

    valid_spots = []

    # 2. Collect ALL valid (x, y) coordinates
    for x in x_coords:
        for y in y_coords:
            nx1, ny1, nx2, ny2 = x, y, x + new_w, y + new_h
            
            # Boundary check
            if nx2 > tx2 or ny2 > ty2 or nx1 < tx1 or ny1 < ty1:
                continue
                
            # Collision check against all existing objects
            collision = False
            for ox1, oy1, ox2, oy2 in object_bounds:
                if not (nx2 <= ox1 or nx1 >= ox2 or ny2 <= oy1 or ny1 >= oy2):
                    collision = True
                    break
            
            if not collision:
                valid_spots.append((nx1, ny1, nx2, ny2))

    # 3. Pick one at random
    if not valid_spots:
        return None
    
    return np.random.choice(valid_spots)

def get_collison_with_objs(object_bounds, obj_pose, x_thr, y_thr = None):

    y_thr = y_thr or x_thr
    for ob in object_bounds:
        if  (ob[0][0] - x_thr <= obj_pose.p[0] <= ob[1][0] + x_thr) and \
            (ob[0][1] - y_thr <= obj_pose.p[1] <= ob[1][1] + y_thr):
            return True
    return False

def get_random_place_pose(xlim, ylim, zlim=None, object_bounds=None, col_thr=0.15, 
                          qpos=(0,0,0), euler=True,
                          rotation=False, rotate_lim = (0,0,0)):
    max_attempts = 100

    if euler:
        qpos = euler2quat(*[np.deg2rad(d) for d in qpos])
    else:
        qpos = qpos
    while True:
        max_attempts -= 1
        if max_attempts <= 0:
                raise RuntimeError("Failed to find a valid placement for the object.")
        obj_pose = rand_pose(
            xlim=xlim,
            ylim=ylim,
            zlim = zlim or [0.741],
            qpos=qpos, 
            rotate_rand=rotation,
            rotate_lim=rotate_lim,
        )
        if not get_collison_with_objs(object_bounds, obj_pose, col_thr):
            return obj_pose
            
def place_actor(obj_name, scene, task_objs, col_thr=0.15, object_bounds=None, 
                obj_id = None, mass = 0.1,  xlim=None, ylim=None, obj_pose=None, 
                qpos=(0,0,0), rotation=False, rotate_lim = (0,0,0),
                is_static=False, scale = None, scene_name='study'):
    
    max_attempts = 100
    if obj_pose is None:
        # Threshold between the objects
        while True:
            max_attempts -= 1
            if max_attempts <= 0:
                raise RuntimeError("Failed to find a valid placement for the object.")
            obj_pose = rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=euler2quat(*[np.deg2rad(d) for d in qpos]),
                rotate_rand=rotation,
                rotate_lim=rotate_lim
            )
            if not get_collison_with_objs(object_bounds, obj_pose, col_thr):
                break
    if isinstance(obj_pose, list):
        obj_pose = sapien.Pose(obj_pose[0],
                    euler2quat(*[np.deg2rad(d) for d in obj_pose[1]]))
    obj_id = obj_id if obj_id is not None else np.random.choice(task_objs['objects'][scene_name]['targets'][obj_name])
    
    print_c(f"Generating {obj_name} with id {obj_id} at position {obj_pose}", "BLUE")
    obj = create_actor(
            scene=scene,
            pose=obj_pose,
            modelname=obj_name,
            convex=True,
            model_id= obj_id,
            is_static=is_static,
            scale= scale or None if task_objs['scales'].get(obj_name) is None else \
                task_objs['scales'][obj_name].get(str(obj_id)) 
    )
    obj.set_mass(mass)
    
    # To compensate for the pose offset
    bbox = get_actor_boundingbox(obj.actor)
    if bbox[0][-1] < obj_pose.p[-1]:
        new_p = [obj_pose.p[0], obj_pose.p[1], 
                    2*obj_pose.p[2]- bbox[0][2]]
        obj.actor.set_pose(
            sapien.Pose(
                p = new_p, 
                q = obj_pose.q)
        )
        print_c(f"Pose adjusted to {new_p}", "YELLOW")
    return obj, obj_id, obj.get_pose()


TEXT_COLOR = {"RED": '\033[31m',
            "GREEN": '\033[32m',
            "BLUE": '\033[34m',
            "RESET": '\033[0m',
            "YELLOW": '\033[33m',
            "MAGENTA": '\033[35m',
            "CYON": '\033[36m',
             "WHITE": '\033[37m'   }
def print_c(text, color="WHITE"):
    print(f"{TEXT_COLOR[color.upper()]}{text}{TEXT_COLOR['RESET']}")

def get_task_objects_config(task_cfg_path=None):
    """
    Load and return the complete benchmark/bench_task_config/task_objects.yml as a dict.

    Args:
        task_cfg_path (str | Path | None):
            Optional explicit path to task_objects.yml.
            If None, defaults to:
            ${BENCH_ROOT}/bench_task_config/task_objects.yml

    Returns:
        dict: Full parsed YAML dictionary.
    """
    if task_cfg_path is None:
        bench_root = os.environ.get("BENCH_ROOT")
        if not bench_root:
            raise RuntimeError("BENCH_ROOT is not set.")
        task_cfg_path = Path(bench_root) / "bench_task_config" / "task_objects.yml"
    else:
        task_cfg_path = Path(task_cfg_path)

    if not task_cfg_path.exists():
        raise FileNotFoundError(f"task_objects.yml not found: {task_cfg_path}")

    with open(task_cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {task_cfg_path}, got {type(data).__name__}")

    return data

def point_to_box_distance(point, b_min, b_max):
    # Calculate distance for each axis (0 if inside the box range)
    dx = max(0, b_min[0] - point[0], point[0] - b_max[0])
    dy = max(0, b_min[1] - point[1], point[1] - b_max[1])
    dz = max(0, b_min[2] - point[2], point[2] - b_max[2])
    
    return np.sqrt(dx**2 + dy**2 + dz**2)
def get_obj_new_pose(obj, ele=0):
    obj_pose = obj.get_pose()
    bbox = get_actor_boundingbox(obj.actor)
    change = False
    z = obj_pose.p[2]
    if bbox[0][-1] < obj_pose.p[-1]:
        z += obj_pose.p[2]- bbox[0][2]
        change = True
    if ele > 0:
        z += ele
        change = True
    if not change:
        return
    new_p = [obj_pose.p[0], obj_pose.p[1], 
                z]
    
    obj.actor.set_pose(
        sapien.Pose(
            p = new_p, 
            q = obj_pose.q)
    )
    print_c(f"Pose adjusted to {new_p}", "YELLOW")