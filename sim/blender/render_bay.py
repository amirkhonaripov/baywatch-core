"""Render one 2-bay episode (run inside Blender 5.0.1).

    blender -b -P sim/blender/render_bay.py -- --episode EP/episode.json --assets ASSETS [--every 1] [--limit N]
                                               [--engine CYCLES|EEVEE] [--frames 0,100,200] [--reverse 1]

Engine: Cycles + OptiX by default (honours CUDA_VISIBLE_DEVICES, so it stays on the evaluation GPU; Eevee renders
through EGL and ignored the GPU pin on the box).

Scene (metres): concrete floor, three walls, two bays at x = -2.75 / +2.75 (cars along y, front towards -y),
a two-post lift per bay (posts, carriages and arms move with the lift height), a workbench, wheel stacks,
CC0 Poly Haven HDRI lighting plus two area lights, and one CCTV-like high camera (pose cam_A/B/C).
Cars: Kenney Car Kit (CC0) with separate wheel meshes; the bonnet is a proxy panel sized from the body mesh.
People: Kenney Mini Characters (CC0, rigged) used as proxy mechanics with idle / interact / holding poses.

Per frame, every bay's state comes from the scenario pieces in episode.json (exact ground truth): car drives in /
out, lift height, wheels removed (placed on the floor beside the car), bonnet open, mechanic at the car / under the
car / away, working or not. Outputs EP/frames/000000.jpg ... (1280x720 JPEG q80), EP/labels.jsonl (projected 2D
boxes: car, person, removed wheels per bay) and EP/rois.json (per-bay image region). Existing frames are skipped,
so a killed job can be restarted.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector

BAY_X = {"bay1": -2.75, "bay2": 2.75}
LIFT_H = 1.75
CAR_SCALE = 1.6
PERSON_H = 1.72
ENTRY_Y = 8.0
CAMERAS = {  # location, look-at, horizontal FOV (deg)
    "cam_A": ((-5.2, -8.4, 4.6), (0.8, 1.2, 0.9), 78.0),
    "cam_B": ((5.4, -8.2, 4.4), (-0.8, 1.2, 0.9), 76.0),
    "cam_C": ((0.3, -9.2, 5.0), (0.0, 1.0, 0.9), 72.0),
}


def args() -> dict:
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"every": 1, "limit": None, "engine": "CYCLES", "frames": None, "reverse": "0", "samples": "32"}
    i = 0
    while i < len(a):
        k = a[i].lstrip("-")
        out[k] = a[i + 1]
        i += 2
    out["every"] = int(out["every"])
    out["limit"] = int(out["limit"]) if out["limit"] else None
    if out["frames"]:
        out["frames"] = [int(x) for x in str(out["frames"]).split(",")]
    return out


def h01(*parts) -> float:
    """Deterministic pseudo-random number in [0, 1) from strings (stable choices within a scenario piece)."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 2 ** 32


# ------------------------------------------------------------------------------------------------ materials
def mat_color(name, rgba, rough=0.6, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def mat_pbr(name, folder: Path, tex: str, scale: float):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])

    def img(suffix, non_color=False):
        p = folder / tex / f"{tex}_{suffix}_1k.jpg"
        if not p.exists():
            return None
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(p))
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    d = img("diff")
    if d:
        nt.links.new(d.outputs["Color"], b.inputs["Base Color"])
    r = img("rough", True)
    if r:
        nt.links.new(r.outputs["Color"], b.inputs["Roughness"])
    n = img("nor", True)
    if n:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return m


def box(name, size, loc, mat, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    o.data.materials.append(mat)
    return o


def plane(name, size, loc, mat):
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0], size[1], 1)
    bpy.ops.object.transform_apply(scale=True)
    o.data.materials.append(mat)
    return o


# -------------------------------------------------------------------------------------------------- scene
def build_world(ep, assets: Path):
    w = bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(assets / "polyhaven" / "hdri" / ep["hdri"] / f"{ep['hdri']}_1k.hdr"))
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(ep["hdri_rotation_deg"]))
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = ep["hdri_strength"]


def build_room(ep, assets: Path):
    tex = assets / "polyhaven" / "tex"
    floor = mat_pbr("floor", tex, ep["floor"], 6.0)
    wall = mat_pbr("wall", tex, "painted_plaster_wall", 3.0)
    plane("floor", (20, 20), (0, 0, 0), floor)
    for name, size, loc in (("wall_left", (0.2, 18, 4.2), (-7.2, 0.5, 2.1)), ("wall_right", (0.2, 18, 4.2), (7.2, 0.5, 2.1)),
                            ("wall_back_mid", (2.6, 0.2, 4.2), (0, 9.4, 2.1)), ("wall_back_l", (1.8, 0.2, 4.2), (-6.3, 9.4, 2.1)),
                            ("wall_back_r", (1.8, 0.2, 4.2), (6.3, 9.4, 2.1)), ("lintel", (14.4, 0.2, 1.0), (0, 9.4, 3.7))):
        box(name, size, loc, wall)
    yellow = mat_color("bay_line", (0.85, 0.65, 0.05, 1), 0.5)
    for x in (-5.0, -0.5, 0.5, 5.0):
        plane(f"line_{x}", (0.12, 9.0), (x, 0.5, 0.004), yellow)
    bench_m = mat_color("bench", (0.25, 0.27, 0.3, 1), 0.4, 0.6)
    red = mat_color("toolbox", (0.6, 0.05, 0.04, 1), 0.35, 0.3)
    box("bench_l", (0.8, 2.2, 0.9), (-6.6, 3.0, 0.45), bench_m)
    box("bench_r", (0.8, 2.2, 0.9), (6.6, 3.0, 0.45), bench_m)
    box("toolbox_l", (0.6, 0.9, 1.1), (-6.6, -1.5, 0.55), red)
    box("toolbox_r", (0.6, 0.9, 1.1), (6.6, -1.4, 0.55), red)
    for i, (x, y) in enumerate(((-0.0, 6.5), (0.0, -2.0))):
        bpy.ops.object.light_add(type="AREA", location=(x, y + 1.5, 4.0))
        L = bpy.context.active_object
        L.data.shape = "RECTANGLE"
        L.data.size, L.data.size_y = 9.0, 2.0
        L.data.energy = 300 * (0.7 + 0.6 * h01(ep["episode_id"], "light", i))
    for x in BAY_X.values():
        bpy.ops.object.light_add(type="AREA", location=(x, 0.3, 4.1))
        L = bpy.context.active_object
        L.data.shape = "RECTANGLE"
        L.data.size, L.data.size_y = 1.2, 5.0
        L.data.energy = 220


class Lift:
    def __init__(self, bay: str):
        c = BAY_X[bay]
        steel = mat_color(f"lift_{bay}", (0.12, 0.25, 0.55, 1), 0.4, 0.4)
        arm_m = mat_color(f"arm_{bay}", (0.08, 0.08, 0.08, 1), 0.5, 0.5)
        for s in (-1, 1):
            box(f"post_{bay}_{s}", (0.32, 0.42, 3.1), (c + s * 1.75, 0.15, 1.55), steel)
        self.carriages = [box(f"carr_{bay}_{s}", (0.2, 0.5, 0.45), (c + s * 1.52, 0.15, 0.25), arm_m) for s in (-1, 1)]
        self.arms = []
        for s in (-1, 1):
            for dy in (-1, 1):
                self.arms.append((box(f"arm_{bay}_{s}_{dy}", (0.95, 0.12, 0.08), (0, 0, 0), arm_m), c, s, dy))
        self.set(0.0)

    def set(self, h: float):
        for k, o in enumerate(self.carriages):
            o.location.z = 0.25 + h
        for o, c, s, dy in self.arms:
            o.location = (c + s * 1.05, 0.15 + dy * 0.75, 0.06 + h)
            o.rotation_euler = (0, 0, math.radians(s * dy * 28))


def import_glb(path: Path) -> list:
    before = set(bpy.data.objects)
    acts_before = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.data.objects if o not in before]
    new_acts = [a for a in bpy.data.actions if a not in acts_before]
    return new, new_acts


def world_bbox(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in objs:
        if o.type != "MESH":
            continue
        oe = o.evaluated_get(dg)
        pts += [oe.matrix_world @ Vector(c) for c in oe.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


class Car:
    def __init__(self, bay: str, model: str, assets: Path, key: str):
        objs, _ = import_glb(assets / "kenney_car_kit" / "x" / "Models" / "GLB format" / f"{model}.glb")
        self.root = bpy.data.objects.new(f"car_{key}", None)
        bpy.context.scene.collection.objects.link(self.root)
        for o in objs:
            if o.parent is None:
                o.parent = self.root
        self.root.scale = (CAR_SCALE,) * 3
        self.body = next(o for o in objs if o.name.startswith("body"))
        self.wheels = {o.name.split(".")[0]: o for o in objs if o.name.startswith("wheel-") and o.parent is self.root}
        self.wheel_home = {k: (w.location.copy(), w.rotation_euler.copy()) for k, w in self.wheels.items()}
        self.objs = objs
        bpy.context.view_layer.update()
        lo, hi = world_bbox([self.body])
        self.L, self.W, self.H = hi.y - lo.y, hi.x - lo.x, hi.z - lo.z
        self.hood = self._make_hood(lo, hi)
        self.bay = bay
        self.show(False)

    def _make_hood(self, lo, hi):
        """Proxy bonnet: a body-coloured panel on the front 30 % of the car, hinged at its rear edge."""
        dg = bpy.context.evaluated_depsgraph_get()
        be = self.body.evaluated_get(dg)
        mw = be.matrix_world
        front = lo.y
        cx = (lo.x + hi.x) / 2
        region = [v.index for v in be.data.vertices if front + 0.04 * self.L < (mw @ v.co).y < front + 0.2 * self.L
                  and abs((mw @ v.co).x - cx) < 0.3 * self.W]
        zs = sorted((mw @ be.data.vertices[i].co).z for i in region)
        hz = zs[int(0.6 * (len(zs) - 1))] if zs else lo.z + 0.5 * self.H
        hinge_y = front + 0.24 * self.L
        self._hood_verts = set(region)
        piv = bpy.data.objects.new(f"hoodpivot_{self.root.name}", None)
        bpy.context.scene.collection.objects.link(piv)
        piv.parent = self.root  # identity parent inverse: location is in car-root units (world / CAR_SCALE)
        piv.location = (0, hinge_y / CAR_SCALE, (hz + 0.015) / CAR_SCALE)
        col = self.body.data.materials[0] if self.body.data.materials else mat_color("hood", (0.5, 0.5, 0.5, 1))
        panel = box(f"hood_{self.root.name}", (0.78 * self.W / CAR_SCALE, 0.22 * self.L / CAR_SCALE, 0.03 / CAR_SCALE),
                    (0, 0, 0), col)
        panel.data.materials.clear()
        hood_m = mat_color(f"hoodm_{self.root.name}", tuple(self._body_tint()) + (1,), 0.35, 0.2)
        panel.data.materials.append(hood_m)
        panel.parent = piv
        panel.location = (0, -0.11 * self.L / CAR_SCALE, 0)
        bay_m = mat_color(f"enginebay_{self.root.name}", (0.03, 0.03, 0.035, 1), 0.7, 0.3)
        plate = box(f"enginebay_{self.root.name}", (0.7 * self.W / CAR_SCALE, 0.2 * self.L / CAR_SCALE, 0.02 / CAR_SCALE),
                    (0, 0, 0), bay_m)
        plate.parent = self.root
        plate.location = (0, (hinge_y - 0.11 * self.L) / CAR_SCALE, (hz + 0.012) / CAR_SCALE)
        self.hood_pivot, self.hood_panel, self.engine_bay = piv, panel, plate
        return panel

    def _body_tint(self):
        """Approximate bonnet colour: the Kenney colour-atlas texel used most by the bonnet-region faces."""
        try:
            img = self.body.data.materials[0].node_tree.nodes["Image Texture"].image
            me = self.body.data
            uv = me.uv_layers.active.data
            counts = {}
            loops = [li for poly in me.polygons if all(vi in self._hood_verts for vi in poly.vertices)
                     for li in poly.loop_indices] or range(min(len(uv), 4000))
            for li in loops:
                d = uv[li]
                k = (round(d.uv[0], 2), round(d.uv[1], 2))
                counts[k] = counts.get(k, 0) + 1
            u, v = max(counts, key=counts.get)
            w, h = img.size
            a = (int(v * (h - 1)) * w + int(u * (w - 1))) * 4
            return list(img.pixels[a:a + 3])
        except Exception:
            return [0.5, 0.5, 0.5]

    def show(self, on: bool):
        for o in [self.root] + self.objs + [self.hood_pivot, self.hood_panel, self.engine_bay]:
            o.hide_render = not on
            o.hide_viewport = not on

    def set(self, y: float, h: float, wheels_off: int, hood: bool, stack_side: int):
        c = BAY_X[self.bay]
        self.root.location = (c, y, h)
        self.hood_pivot.rotation_euler = (math.radians(-55) if hood else 0.0, 0, 0)
        for o in (self.hood_panel, self.engine_bay):  # closed bonnet = the car's own body; open = proxy panel
            o.hide_render = o.hide_viewport = not hood
        order = ["wheel-front-left", "wheel-front-right", "wheel-back-left", "wheel-back-right"]
        for k, name in enumerate(order):
            w = self.wheels.get(name)
            if w is None:
                continue
            loc, rot = self.wheel_home[name]
            if k < wheels_off:  # removed: lying flat on the floor beside the bay (world coords via parent inverse)
                wx = c + stack_side * (self.W / 2 + 0.75) - self.root.location.x
                wy = -1.4 + 0.75 * k - self.root.location.y
                w.location = (wx / CAR_SCALE, wy / CAR_SCALE, (0.2 - h) / CAR_SCALE)
                w.rotation_euler = (Euler((0, math.radians(90), 0)).to_matrix() @ rot.to_matrix()).to_euler()
            else:
                w.location, w.rotation_euler = loc, rot


class Person:
    def __init__(self, bay: str, character: str, assets: Path):
        objs, acts = import_glb(assets / "kenney_mini_characters" / "x" / "Models" / "GLB format" / f"{character}.glb")
        self.arm = next(o for o in objs if o.type == "ARMATURE")
        for o in objs:
            if o.type == "MESH" and o.parent is None:  # stray helper meshes (e.g. an icosphere) are not rendered
                o.hide_render = o.hide_viewport = True
        self.meshes = [o for o in objs if o.type == "MESH" and o.parent is not None]
        self.root = bpy.data.objects.new(f"person_{bay}", None)
        bpy.context.scene.collection.objects.link(self.root)
        self.arm.parent = self.root
        bpy.context.view_layer.update()
        lo, hi = world_bbox(self.meshes)
        s = PERSON_H / max(1e-3, hi.z - lo.z)
        self.root.scale = (s, s, s)
        self.acts = {a.name.split(".")[0]: a for a in acts}
        if self.arm.animation_data is None:
            self.arm.animation_data_create()
        self.bay = bay

    def show(self, on: bool):
        for o in [self.root, self.arm] + self.meshes:
            o.hide_render = not on
            o.hide_viewport = not on

    def pose(self, name: str):
        act = self.acts.get(name) or self.acts.get("idle")
        ad = self.arm.animation_data
        if ad.action is not act:
            ad.action = act
            try:
                if getattr(ad, "action_slot", None) is None and len(act.slots):
                    ad.action_slot = act.slots[0]
            except Exception:
                pass

    def place(self, x, y, yaw_deg):
        self.root.location = (x, y, 0)
        self.root.rotation_euler = Euler((0, 0, math.radians(yaw_deg)))


# ------------------------------------------------------------------------------------------------ state
def piece_at(bay, ts):
    for v in bay["visits"]:
        u = ts - v["offset_s"]
        if 0 <= u < v["duration_s"]:
            for p in v["pieces"]:
                if p["start_s"] <= u < p["start_s"] + p["dur_s"]:
                    return v, p, u - p["start_s"]
    return None, None, 0.0


def smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def bay_state(bay, ts):
    v, p, e = piece_at(bay, ts)
    if v is None:
        return None
    dur = max(1.0, p["dur_s"])
    u = e / dur
    y = 0.0
    if p["step"] == "drive_in":
        y = ENTRY_Y * (1 - smooth(u))
    elif p["step"] == "drive_out":
        y = ENTRY_Y * smooth(u)
    if p["step"] == "raise_lift":
        h = LIFT_H * smooth(e / min(dur, 25.0))
    elif p["step"] == "lower_lift":
        h = LIFT_H * (1 - smooth(e / min(dur, 25.0)))
    else:
        h = LIFT_H if p["raised"] else 0.0
    n_wh = 2 if v["job_type"] == "brake_pads" else 4
    if p["step"] == "wheels_off":
        wo = 1 + int(u * (n_wh - 0.001))
    elif p["step"] == "wheels_on":
        wo = max(1, n_wh - int(u * n_wh))
    else:
        wo = n_wh if p["wheel"] else 0
    return {"visit": v, "piece": p, "y": y, "h": h, "wheels_off": wo, "hood": bool(p["hood"]),
            "person": p["person"], "work": p["work"]}


def person_pose(st, car: Car, bay: str, ts: float):
    p, v = st["piece"], st["visit"]
    c = BAY_X[bay]
    r = h01(v["visit_id"], p["start_s"], p["step"])
    if st["person"] == 2:
        return (c + (r - 0.5) * 0.6, -0.9 + 1.8 * h01(v["visit_id"], p["start_s"], "y"), 180 if r < 0.5 else 0,
                "holding-both" if st["work"] else "idle")
    if st["person"] == 1:
        if st["hood"] and p["step"] not in ("replace_brake_pads", "bleed_brakes"):
            return (c + (r - 0.5) * 0.8, st["y"] - car.L / 2 - 0.45, 180, "interact-right" if st["work"] else "idle")
        side = -1 if r < 0.5 else 1
        if st["wheels_off"]:
            return (c + side * (car.W / 2 + 0.4), st["y"] - car.L * 0.28, -90 * side, "crouch" if st["work"] else "idle")
        return (c + side * (car.W / 2 + 0.45), st["y"] - 0.2 + (r - 0.5), -90 * side, "interact-left" if st["work"] else "idle")
    if r < 0.5:  # technician away from the car but visible at the bench
        return (5.8 if c > 0 else -5.8, 3.0 + r, 90 if c > 0 else -90, "idle")
    return None


def bbox2d(scene, cam, objs):
    dg = bpy.context.evaluated_depsgraph_get()
    xs, ys = [], []
    for o in objs:
        if o.type != "MESH" or o.hide_render:
            continue
        oe = o.evaluated_get(dg)
        for c in oe.bound_box:
            p = world_to_camera_view(scene, cam, oe.matrix_world @ Vector(c))
            if p.z > 0:
                xs.append(p.x)
                ys.append(1 - p.y)
    if not xs:
        return None
    b = [max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))]
    return [round(v, 4) for v in b] if b[2] > b[0] and b[3] > b[1] else None


def main():
    A = args()
    ep_path = Path(A["episode"])
    ep = json.loads(ep_path.read_text())
    assets = Path(A["assets"])
    out = ep_path.parent
    (out / "frames").mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = ep["resolution"]
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 80
    if A["engine"] == "CYCLES":
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "OPTIX"
        sc.cycles.device = "GPU"
        sc.cycles.samples = int(A["samples"])
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPTIX"
        sc.cycles.max_bounces = 4
        sc.cycles.diffuse_bounces = 2
        sc.cycles.glossy_bounces = 2
        sc.cycles.transmission_bounces = 2
        sc.render.use_persistent_data = True  # keep the BVH between frames
    else:
        sc.render.engine = "BLENDER_EEVEE"
        try:
            sc.eevee.taa_render_samples = 24
        except AttributeError:
            pass
    build_world(ep, assets)
    build_room(ep, assets)
    lifts = {b["bay_id"]: Lift(b["bay_id"]) for b in ep["bays"]}
    cars, people = {}, {}
    for b in ep["bays"]:
        for v in b["visits"]:
            key = f"{b['bay_id']}_{v['car_model']}"
            if key not in cars:
                cars[key] = Car(b["bay_id"], v["car_model"], assets, key)
        people[b["bay_id"]] = Person(b["bay_id"], b["visits"][0]["character"], assets)
    loc, look, fov = CAMERAS[ep["camera"]]
    cam_data = bpy.data.cameras.new("cctv")
    cam_data.angle = math.radians(fov)
    cam_data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("cctv", cam_data)
    sc.collection.objects.link(cam)
    jit = [(h01(ep["episode_id"], "cam", i) - 0.5) * 0.3 for i in range(3)]
    cam.location = Vector(loc) + Vector(jit)
    cam.rotation_euler = (Vector(look) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    bpy.context.view_layer.update()

    rois = {}
    for bay_id, c in BAY_X.items():
        pts = [Vector((c + dx, y, z)) for dx in (-2.3, 2.3) for y in (-3.6, 3.8) for z in (0.0, 3.3)]
        pr = [world_to_camera_view(sc, cam, p) for p in pts]
        rois[bay_id] = [round(max(0, min(p.x for p in pr)), 4), round(max(0, min(1 - p.y for p in pr)), 4),
                        round(min(1, max(p.x for p in pr)), 4), round(min(1, max(1 - p.y for p in pr)), 4)]
    (out / "rois.json").write_text(json.dumps({"camera": ep["camera"], "rois": rois}, indent=1))

    frames = A["frames"] or list(range(0, ep["n_frames"], A["every"]))
    if A["limit"]:
        frames = frames[: A["limit"]]
    if str(A["reverse"]) == "1":  # a second worker can render the same episode from the end; existing frames are skipped
        frames = frames[::-1]
    lab_f = open(out / "labels.jsonl", "a")
    t_start, n_done = time.time(), 0
    for i in frames:
        fp = out / "frames" / f"{i:06d}.jpg"
        if fp.exists():
            continue
        ts = i * ep["frame_dt_s"]
        rec = {"frame": i, "t_s": ts, "bays": {}}
        for car in cars.values():
            car.show(False)
        for person in people.values():
            person.show(False)
        for b in ep["bays"]:
            bay = b["bay_id"]
            st = bay_state(b, ts)
            lifts[bay].set(st["h"] if st else 0.0)
            rec["bays"][bay] = {"car": None, "person": None, "wheels": []}
            if st is None:
                continue
            car = cars[f"{bay}_{st['visit']['car_model']}"]
            car.show(True)
            side = 1 if BAY_X[bay] > 0 else -1
            car.set(st["y"], st["h"], st["wheels_off"], st["hood"], side)
            pp = person_pose(st, car, bay, ts)
            person = people[bay]
            if pp is not None:
                person.show(True)
                person.place(pp[0], pp[1], pp[2])
                person.pose(pp[3])
        sc.frame_set(1 + (i * 7) % 24)
        bpy.context.view_layer.update()
        for b in ep["bays"]:
            bay = b["bay_id"]
            st = bay_state(b, ts)
            if st is None:
                continue
            car = cars[f"{bay}_{st['visit']['car_model']}"]
            mounted = [car.body, car.hood_panel] + [w for k, w in car.wheels.items()
                                                    if k not in ["wheel-front-left", "wheel-front-right", "wheel-back-left", "wheel-back-right"][: st["wheels_off"]]]
            rec["bays"][bay]["car"] = bbox2d(sc, cam, mounted)
            rec["bays"][bay]["person"] = bbox2d(sc, cam, people[bay].meshes) if not people[bay].root.hide_render else None
            rec["bays"][bay]["wheels"] = [bbox2d(sc, cam, [car.wheels[k]]) for k in
                                          ["wheel-front-left", "wheel-front-right", "wheel-back-left", "wheel-back-right"][: st["wheels_off"]]
                                          if k in car.wheels]
        sc.render.filepath = str(fp)
        bpy.ops.render.render(write_still=True)
        lab_f.write(json.dumps(rec) + "\n")
        lab_f.flush()
        n_done += 1
        if n_done % 50 == 0:
            el = time.time() - t_start
            print(f"[render] {ep['episode_id']} {n_done}/{len(frames)} frames, {el / n_done:.2f} s/frame", flush=True)
    lab_f.close()
    print(f"[render] done {ep['episode_id']}: {n_done} new frames in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
