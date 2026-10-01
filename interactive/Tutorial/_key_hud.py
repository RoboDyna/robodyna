"""Top-right SAPIEN viewer HUD that shows keycap figures via an off-stage camera."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sapien
from PIL import Image
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin

from _keycaps import (
    cluster_size,
    draw_action_stage,
    draw_advanced_stage,
    draw_arm_keys,
    draw_control_stage,
    draw_kb_stage,
    draw_part2_play_keys,
    draw_play_keys,
    draw_view_key,
)

_DISPLAY_WIDTH = 400
_MAX_DISPLAY_H = 380
_HALF_H = 0.12
_STAGE_ORIGIN_STEP = 4.0
# ImGui title bar + window padding; keep this larger than the picture so
# the instruction line is visible without a vertical scrollbar.
_WINDOW_PAD_W = 36
_WINDOW_PAD_H = 112
_FLASH_SECONDS = 0.22


def _texture_from_image(img: Image.Image):
    arr = np.ascontiguousarray(np.array(img.convert("RGBA"), dtype=np.uint8))
    try:
        return sapien.render.RenderTexture2D(arr, "r8g8b8a8unorm", srgb=True)
    except Exception:
        path = Path("/tmp/robodyna_tutorial_hud.png")
        img.convert("RGBA").save(path)
        return sapien.render.RenderTexture2D(str(path), srgb=True)


def _textured_quad(scene, tex, half_w: float, half_h: float, name: str):
    mat = sapien.render.RenderMaterial()
    mat.set_base_color_texture(tex)
    mat.set_emission_texture(tex)
    mat.base_color = [1, 1, 1, 1]
    mat.emission = [1, 1, 1, 1]
    vertices = np.array(
        [[0, half_w, -half_h], [0, -half_w, -half_h], [0, -half_w, half_h], [0, half_w, half_h]],
        dtype=np.float32,
    )
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    normals = np.array([[-1, 0, 0]] * 4, dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = sapien.render.RenderShapeTriangleMesh(vertices, triangles, normals, uvs, mat)
    body = sapien.render.RenderBodyComponent()
    body.attach(mesh)
    ent = sapien.Entity()
    ent.add_component(body)
    ent.set_name(name)
    scene.add_entity(ent)
    return ent, mat


def _add_hud_camera(scene, img: Image.Image, origin, name: str):
    aspect = img.width / float(img.height)
    half_h = _HALF_H
    half_w = _HALF_H * aspect
    tex = _texture_from_image(img)
    ent, mat = _textured_quad(scene, tex, half_w, half_h, f"tutorial_hud_{name}")
    ent.set_pose(sapien.Pose(p=list(origin)))
    cam = scene.add_camera(f"tutorial_hud_cam_{name}", img.width, img.height, 1.0, 0.05, 5.0)
    cam.set_orthographic_parameters(0.05, 5.0, -half_w, half_w, -half_h, half_h)
    cam.entity.set_pose(sapien.Pose(p=[origin[0] - 0.8, origin[1], origin[2]]))
    return cam, tex, mat, ent


class TutorialKeyHud(Plugin):
    """ImGui window in the top-left showing the current tutorial key overlay."""

    def __init__(
        self,
        cameras,
        textures,
        materials,
        sizes=None,
        display_width=None,
        window_pad_h=None,
        entities=None,
    ):
        self.cameras = cameras
        self.textures = textures
        self.materials = materials
        self.entities = dict(entities or {})
        self.sizes = dict(sizes or {})
        self.stage = next(iter(cameras), "arms")
        self.pressed_arms: set[str] = set()
        self.v_pressed = False
        self.caption = ""
        self.on_frame = None
        self.ui_window = None
        self.ui_picture = None
        self.ui_caption = None
        self._laid_out_stage = None
        self._held: set[str] = set()
        self._flash_until: dict[str, float] = {}
        self._play_drawn: set[str] | None = None
        self.play_drawer = draw_play_keys
        self.flash_drawer = None
        self.flash_enabled = False
        self.flash_kwargs: dict = {}
        self.lesson_drawer = None
        self.lesson_pressed: set[str] = set()
        self._lesson_key = None
        self._lesson_img = None
        self._rebuild_picture = False
        self._nudge_sign = 1.0
        canvas_w, canvas_h = cluster_size()
        if self.sizes:
            canvas_w, canvas_h = next(iter(self.sizes.values()))
        self.pic_w = int(display_width or _DISPLAY_WIDTH)
        self.pic_h = max(1, int(round(self.pic_w * canvas_h / canvas_w)))
        self.window_pad_w = _WINDOW_PAD_W
        self.window_pad_h = int(window_pad_h if window_pad_h is not None else _WINDOW_PAD_H)

    def before_render(self):
        if self.on_frame is not None:
            viewer = getattr(self, "viewer", None)
            window = getattr(viewer, "window", None) if viewer is not None else None
            self.on_frame(window)
        if self.flash_enabled:
            self._tick_flash()
        else:
            self._sync_lesson()

    def _display_wh(self) -> tuple[int, int]:
        size = self.sizes.get(self.stage)
        if size is None:
            for alt in ("view", "arms", "play", "arrows"):
                size = self.sizes.get(alt)
                if size is not None:
                    break
        if size is None:
            return self.pic_w, self.pic_h
        width, height = size
        pic_w = self.pic_w
        pic_h = max(1, int(round(self.pic_w * height / float(width))))
        if pic_h > _MAX_DISPLAY_H:
            scale = _MAX_DISPLAY_H / float(pic_h)
            pic_w = max(1, int(round(pic_w * scale)))
            pic_h = int(_MAX_DISPLAY_H)
        return pic_w, pic_h

    def update_texture(self, name: str, img: Image.Image) -> None:
        tex = self.textures.get(name)
        if tex is None or img is None:
            return
        try:
            target = (int(tex.width), int(tex.height))
        except Exception:
            target = self.sizes.get(name) or img.size
        if img.size != target:
            img = img.resize(target, Image.BILINEAR)
        arr = np.ascontiguousarray(np.array(img.convert("RGBA"), dtype=np.uint8))
        try:
            tex.upload(arr)
        except Exception:
            return
        self._nudge_quad(name)
        cam = self.cameras.get(name)
        if cam is None:
            return
        try:
            cam.take_picture()
        except Exception:
            pass

    def _nudge_quad(self, name: str) -> None:
        """Dirty the HUD quad so the camera re-samples the uploaded texture."""
        ent = self.entities.get(name)
        if ent is None:
            return
        try:
            pose = ent.get_pose()
            delta = 1e-5 * float(self._nudge_sign)
            self._nudge_sign = -self._nudge_sign
            p = list(pose.p)
            p[0] += delta
            ent.set_pose(sapien.Pose(p, list(pose.q)))
        except Exception:
            pass

    def mark_arm_pressed(self, key: str) -> None:
        self.pressed_arms.add(key)

    def mark_v_pressed(self) -> None:
        self.v_pressed = True

    def flash(self, label: str) -> None:
        self._flash_until[label] = time.perf_counter() + _FLASH_SECONDS

    def set_held(self, held: set[str]) -> None:
        self._held = set(held)

    def _lesson_image(self, key, factory) -> Image.Image:
        if key != self._lesson_key or self._lesson_img is None:
            self._lesson_key = key
            self._lesson_img = factory()
        return self._lesson_img

    def _sync_lesson(self) -> None:
        """Upload persist-green keys every frame, immediately before take_picture.

        A one-shot upload is easy to miss: viewer.render() calls
        window.update_render() *before* before_render, and skipping later
        frames leaves the HUD camera on the unpressed bake.
        """
        if self.stage == "arms":
            key = ("arms", frozenset(self.pressed_arms))
            img = self._lesson_image(key, lambda: draw_arm_keys(self.pressed_arms))
            self.update_texture("arms", img)
            return
        if self.stage == "view":
            key = ("view", bool(self.v_pressed))
            img = self._lesson_image(
                key, lambda: draw_view_key(pressed=self.v_pressed)
            )
            self.update_texture("view", img)
            return
        if self.lesson_drawer is None or self.stage not in self.textures:
            return
        key = (self.stage, frozenset(self.lesson_pressed))
        drawer = self.lesson_drawer
        img = self._lesson_image(key, lambda: drawer(self.lesson_pressed))
        self.update_texture(self.stage, img)

    def _flash_image_drawer(self):
        if self.stage == "play":
            return self.play_drawer
        return self.flash_drawer

    def _tick_flash(self) -> None:
        drawer = self._flash_image_drawer()
        if drawer is None or self.stage not in self.textures:
            return
        now = time.perf_counter()
        active = set(self._held)
        expired = [key for key, until in self._flash_until.items() if now >= until]
        for key in expired:
            self._flash_until.pop(key, None)
        for key, until in self._flash_until.items():
            if now < until:
                active.add(key)
        key = (self.stage, frozenset(active))
        if key != self._lesson_key or self._lesson_img is None:
            self._lesson_key = key
            self._lesson_img = drawer(active)
            self._play_drawn = set(active)
        self.update_texture(self.stage, self._lesson_img)

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self._held = set()
        self._flash_until = {}
        self._play_drawn = None
        self.lesson_pressed = set()
        self._lesson_key = None
        self._lesson_img = None
        self._rebuild_picture = True
        drawer = self._flash_image_drawer() if self.flash_enabled else None
        if drawer is not None and stage in self.textures:
            self.update_texture(stage, drawer())
            self._play_drawn = set()
        elif stage == "done":
            self.v_pressed = True
            if "view" in self.textures:
                self.update_texture("view", draw_view_key(pressed=True))

    def get_ui_windows(self):
        cam = self.cameras.get(self.stage)
        if cam is None:
            cam = self.cameras.get("view") or next(iter(self.cameras.values()))
        try:
            cam.take_picture()
        except Exception:
            pass
        pw, ph = self._display_wh()
        if self.ui_window is None or self._rebuild_picture:
            self._rebuild_picture = False
            self.ui_picture = R.UIPicture().Size(pw, ph)
            self.ui_window = (
                R.UIWindow()
                .Label("Tutorial keys")
                .Id("tutorial_keys_tl")
                .Pos(16, 16)
                .append(self.ui_picture)
            )
            self._laid_out_stage = None
        self.ui_picture.Size(pw, ph)
        self.ui_window.Size(pw + self.window_pad_w, ph + self.window_pad_h)
        # Force top-left every frame — imgui.ini otherwise restores Keys## on the right.
        self.ui_window.Pos(16, 16)
        self._laid_out_stage = self.stage
        self.ui_picture.Picture(cam._internal_renderer, "Color")
        return [self.ui_window]


def build_staged_hud(
    scene,
    images: dict[str, Image.Image],
    start_stage: str,
    *,
    display_width: int | None = None,
    window_pad_h: int | None = None,
) -> TutorialKeyHud:
    """Off-stage UV quads + ortho cameras, one per named overlay stage."""
    cameras = {}
    textures = {}
    materials = {}
    entities = {}
    sizes = {}
    for index, (name, img) in enumerate(images.items()):
        origin = (80.0, (index + 1) * _STAGE_ORIGIN_STEP, 80.0)
        cam, tex, mat, ent = _add_hud_camera(scene, img, origin, name)
        cameras[name] = cam
        textures[name] = tex
        materials[name] = mat
        entities[name] = ent
        sizes[name] = img.size
        try:
            cam.take_picture()
        except Exception:
            pass
    hud = TutorialKeyHud(
        cameras,
        textures,
        materials,
        sizes=sizes,
        display_width=display_width,
        window_pad_h=window_pad_h,
        entities=entities,
    )
    hud.stage = start_stage
    return hud


def build_key_hud(scene) -> TutorialKeyHud:
    """Off-stage UV quads + ortho cameras for the 1/2/3 cluster and the V key."""
    hud = build_staged_hud(
        scene,
        {
            "arms": draw_arm_keys(),
            "view": draw_view_key(),
            "play": draw_play_keys(),
        },
        start_stage="arms",
    )
    hud.cameras["done"] = hud.cameras["play"]
    return hud


def build_part2_key_hud(scene) -> TutorialKeyHud:
    """Overlays for arrows → E/Q → R/T → F/G → Space, then a compact play strip."""
    stages = ("arrows", "height", "yaw", "tilt", "space")
    images = {name: draw_control_stage(name) for name in stages}
    images["play"] = draw_part2_play_keys()
    hud = build_staged_hud(scene, images, start_stage="arrows")
    hud.play_drawer = draw_part2_play_keys
    return hud


def build_part3_key_hud(scene) -> TutorialKeyHud:
    """Overlays for grasp → hold button → on/off switch → push box."""
    stages = ("grasp", "hold", "switch", "push")
    images = {name: draw_action_stage(name) for name in stages}
    hud = build_staged_hud(scene, images, start_stage="grasp")
    hud.flash_drawer = lambda pressed=None, **kwargs: draw_action_stage(hud.stage, pressed)
    hud.flash_enabled = True
    return hud


def build_part4_key_hud(scene, stages=None) -> TutorialKeyHud:
    """Overlays for Part 4 advanced stages (suite-specific sequence)."""
    stage_names = tuple(stages or ("ball", "mallet"))
    images = {name: draw_advanced_stage(name) for name in stage_names}
    hud = build_staged_hud(scene, images, start_stage=stage_names[0])
    hud.flash_drawer = lambda pressed=None, **kwargs: draw_advanced_stage(
        hud.stage, pressed, **hud.flash_kwargs
    )
    hud.flash_enabled = True
    return hud


def build_keyboard_tutorial_hud(scene, stages) -> TutorialKeyHud:
    """Overlays for keyboard tutorial stages (keys vs mouse)."""
    images = {name: draw_kb_stage(name) for name in stages}
    hud = build_staged_hud(scene, images, start_stage=stages[0], window_pad_h=120)
    hud.flash_drawer = lambda pressed=None, **kwargs: draw_kb_stage(hud.stage, pressed)
    hud.flash_enabled = True
    return hud
