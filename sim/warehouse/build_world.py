"""Build the warehouse aisle on a USD stage (x along the aisle from the entrance, y left, z up, metres)."""
import math
import os
import random
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "stock_drone", "stock_drone"))
import aisle_spec as spec  # noqa: E402

TEX = os.path.join(HERE, "textures")
TEMPLATES = os.path.join(HERE, "templates")
PROPS = "Isaac/Environments/Simple_Warehouse/Props/"

X0, X1 = -15.0, 95.0  # building extent along the aisle, m
WALL_Y = 2.75  # walls just behind the racks on both sides, m
ROOF_Z = 9.5  # m, 1.5 m above the rack tops
BLUE, ORANGE, YELLOW = (0.05, 0.22, 0.62), (0.85, 0.33, 0.05), (0.95, 0.75, 0.05)


def material(stage, path, color=(0.5, 0.5, 0.5), texture=None, roughness=0.6, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, path + "/tex")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def quad(stage, path, p0, u, v, mat, uv_scale=(1.0, 1.0)):
    """Rectangle with corner p0 and edge vectors u, v; normal is u x v; UVs span uv_scale."""
    m = UsdGeom.Mesh.Define(stage, path)
    p0, u, v = Gf.Vec3f(*p0), Gf.Vec3f(*u), Gf.Vec3f(*v)
    m.CreatePointsAttr([p0, p0 + u, p0 + u + v, p0 + v])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    n = Gf.Cross(u, v).GetNormalized()
    m.CreateNormalsAttr([n] * 4)
    m.SetNormalsInterpolation("vertex")
    su, sv = uv_scale
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    st.Set([(0, 0), (su, 0), (su, sv), (0, sv)])
    m.CreateDoubleSidedAttr(False)
    bind(m.GetPrim(), mat)
    return m


def collide(prim):
    """Make a prim a static collider (no rigid body)."""
    UsdPhysics.CollisionAPI.Apply(prim)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")


def box(stage, path, center, size, mat, yaw_deg=0.0, roll_deg=0.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg or roll_deg:
        xf.AddRotateXYZOp().Set(Gf.Vec3f(roll_deg, 0.0, yaw_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(c.GetPrim(), mat)
    return c


def _ref(stage, path, asset, pos, yaw_deg=0.0):
    p = UsdGeom.Xform.Define(stage, path)
    p.GetPrim().GetReferences().AddReference(asset)
    xf = UsdGeom.Xformable(p)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if yaw_deg:
        xf.AddRotateZOp().Set(yaw_deg)
    return p


def _template_items(kind):
    """(prop, x, y, z, yaw) list for one load; origin at the pallet front-centre bottom, y into the rack."""
    pal = [("SM_PaletteA_02", 0.0, 0.6085, 0.0, 90.0)]
    z0 = 0.209
    items = []
    if kind == "drums":  # 0.6 x 0.708 m barrels turned so the 0.6 m side faces the aisle
        items = [("SM_BarelPlastic_A_01", 0.0, 0.305, z0, 90.0), ("SM_BarelPlastic_A_01", 0.0, 0.905, z0, 270.0)]
    elif kind == "drums_low":
        items = [("SM_BarelPlastic_B_01", 0.0, 0.305, z0, 90.0), ("SM_BarelPlastic_B_01", 0.0, 0.905, z0, 270.0),
                 ("SM_BarelPlastic_B_01", 0.0, 0.905, z0 + 0.588, 90.0)]
    elif kind == "boxes_A":  # 0.7 x 0.5 and 0.5 x 0.5 boxes, 0.25 m deep each side
        for layer in range(2):
            z = z0 + 0.5 * layer
            items += [("SM_CardBoxA_01", 0.0, 0.255, z, 0.0),
                      ("SM_CardBoxB_01" if layer else "SM_CardBoxA_01", 0.0, 0.855, z, 0.0)]
    elif kind == "boxes_D":  # 0.38 x 0.25 cartons turned 90 deg: 0.25 wide, 0.19 deep each side
        for layer in range(6):
            z = z0 + 0.149 * layer
            for ix in (-0.26, 0.0, 0.26):
                for iy in (0.195, 0.575, 0.955):
                    if iy > 0.9 and layer > 3:
                        continue
                    items.append(("SM_CardBoxD_01" if (layer + int(iy * 8) + int(ix * 10)) % 3 else "SM_CardBoxD_05",
                                  ix, iy, z, 90.0))
    elif kind == "crates":  # 0.616 x 0.499 crates turned 90 deg: 0.308 m deep each side
        for layer in range(3):
            z = z0 + 0.435 * layer
            items += [("SM_CratePlastic_A_01", 0.0, 0.31, z, 90.0), ("SM_CratePlastic_A_01", 0.0, 0.92, z, 90.0)]
    elif kind == "bags":  # flat 0.5 x 0.5 cartons
        for layer in range(4):
            z = z0 + 0.25 * layer
            items += [("SM_CardBoxC_01", 0.0, 0.255, z, 0.0), ("SM_CardBoxC_01", 0.0, 0.855, z, 0.0)]
    return pal + items


TEMPLATE_KINDS = {"drums": ["drums", "drums_low"], "boxes": ["boxes_A", "boxes_D"], "crates": ["crates"],
                  "bags": ["bags"]}


def write_templates(assets_root):
    os.makedirs(TEMPLATES, exist_ok=True)
    paths = {}
    for kinds in TEMPLATE_KINDS.values():
        for kind in kinds:
            path = os.path.join(TEMPLATES, f"load_{kind}.usda")
            st = Usd.Stage.CreateNew(path) if not os.path.exists(path) else Usd.Stage.Open(path)
            st.GetRootLayer().Clear()
            UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(st, 1.0)
            root = UsdGeom.Xform.Define(st, "/load")
            st.SetDefaultPrim(root.GetPrim())
            for i, (prop, x, y, z, yaw) in enumerate(_template_items(kind)):
                _ref(st, f"/load/p{i:02d}", f"{assets_root}/{PROPS}{prop}.usd", (x, y, z), yaw)
            st.GetRootLayer().Save()
            paths[kind] = path
    return paths


def build_shell(stage, root):
    m_floor = material(stage, f"{root}/Looks/floor", texture=os.path.join(TEX, "floor.png"), roughness=0.8)
    m_wall = material(stage, f"{root}/Looks/wall", texture=os.path.join(TEX, "wall.png"), roughness=0.5,
                      metallic=0.3)
    m_roof = material(stage, f"{root}/Looks/roof", texture=os.path.join(TEX, "ceiling.png"), roughness=0.5,
                      metallic=0.3)
    m_line = material(stage, f"{root}/Looks/line", color=YELLOW, roughness=0.5)
    L, W = X1 - X0, 2 * WALL_Y
    floor = quad(stage, f"{root}/Shell/floor", (X0, -WALL_Y, 0.0), (L, 0, 0), (0, W, 0), m_floor,
                 (L / 6.0, W / 6.0))
    collide(floor.GetPrim())
    quad(stage, f"{root}/Shell/roof", (X0, WALL_Y, ROOF_Z), (L, 0, 0), (0, -W, 0), m_roof, (L / 6.0, W / 6.0))
    for wall in (quad(stage, f"{root}/Shell/wall_left", (X0, WALL_Y, 0.0), (0, 0, ROOF_Z), (L, 0, 0), m_wall,
                      (ROOF_Z / 4.0, L / 4.0)),
                 quad(stage, f"{root}/Shell/wall_right", (X0, -WALL_Y, 0.0), (L, 0, 0), (0, 0, ROOF_Z), m_wall,
                      (L / 4.0, ROOF_Z / 4.0))):
        collide(wall.GetPrim())
    quad(stage, f"{root}/Shell/wall_start", (X0, -WALL_Y, 0.0), (0, 0, ROOF_Z), (0, W, 0), m_wall, (ROOF_Z / 4, W / 4))
    quad(stage, f"{root}/Shell/wall_end", (X1, -WALL_Y, 0.0), (0, W, 0), (0, 0, ROOF_Z), m_wall, (W / 4, ROOF_Z / 4))
    for s in (1, -1):  # painted lines along the pallet faces, and the take-off pad marks
        quad(stage, f"{root}/Shell/line_{'L' if s > 0 else 'R'}", (0.0, s * 1.22 - 0.05, 0.002), (80.0, 0, 0),
             (0, 0.1, 0), m_line)
    for k, (x, y, dx, dy) in enumerate([(-4.0, -1.0, 2.0, 0.08), (-4.0, 0.92, 2.0, 0.08), (-4.0, -1.0, 0.08, 2.0),
                                        (-2.08, -1.0, 0.08, 2.0)]):
        quad(stage, f"{root}/Shell/pad_{k}", (x, y, 0.002), (dx, 0, 0), (0, dy, 0), m_line)
    # roof purlins every 6 m give the fisheyes features overhead
    m_steel = material(stage, f"{root}/Looks/steel", color=(0.35, 0.36, 0.40), roughness=0.4, metallic=0.6)
    for i, x in enumerate(range(int(X0), int(X1) + 1, 6)):
        box(stage, f"{root}/Shell/purlin_{i:02d}", (x, 0.0, ROOF_Z - 0.2), (0.25, W, 0.4), m_steel)


def build_lights(stage, root, intensity=150000.0, fill=2000.0):
    for i, x in enumerate(range(int(X0) + 3, int(X1) - 2, 5)):
        light = UsdLux.RectLight.Define(stage, f"{root}/Lights/bay_{i:02d}")
        light.CreateWidthAttr(2.0)
        light.CreateHeightAttr(0.35)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, ROOF_Z - 0.5))
    dome = UsdLux.DomeLight.Define(stage, f"{root}/Lights/fill")
    dome.CreateIntensityAttr(fill)


def build_racks(stage, root):
    m_up = material(stage, f"{root}/Looks/upright", color=BLUE, roughness=0.45, metallic=0.4)
    m_beam = material(stage, f"{root}/Looks/beam", color=ORANGE, roughness=0.45, metallic=0.4)
    m_brace = material(stage, f"{root}/Looks/brace", color=(0.55, 0.58, 0.62), roughness=0.4, metallic=0.6)
    h = spec.HEIGHT
    for side, s in (("L", 1), ("R", -1)):
        y_front = s * (spec.WIDTH / 2 + spec.FACE_GAP + spec.UPRIGHT_D / 2)
        y_back = s * (spec.WIDTH / 2 + spec.FACE_GAP + spec.RACK_DEPTH - spec.UPRIGHT_D / 2)
        depth = abs(y_back - y_front)
        for i in range(spec.N_BAYS + 1):
            x = spec.RACK_X0 + i * spec.BAY_PITCH
            base = f"{root}/Racks/{side}/frame_{i:02d}"
            collide(box(stage, base + "/front", (x, y_front, h / 2), (spec.UPRIGHT_W, spec.UPRIGHT_D, h),
                        m_up).GetPrim())
            box(stage, base + "/back", (x, y_back, h / 2), (spec.UPRIGHT_W, spec.UPRIGHT_D, h), m_up)
            ymid = (y_front + y_back) / 2
            for j, z in enumerate([0.15 + 1.2 * k for k in range(7)]):
                box(stage, f"{base}/h{j}", (x, ymid, z), (0.03, depth, 0.03), m_brace)
                if z + 1.2 < h:
                    ang = math.degrees(math.atan2(1.2, depth)) * (1 if j % 2 else -1) * s
                    box(stage, f"{base}/d{j}", (x, ymid, z + 0.6), (0.025, math.hypot(depth, 1.2), 0.025), m_brace,
                        roll_deg=ang)
        for b in range(spec.N_BAYS):
            xc = spec.RACK_X0 + (b + 0.5) * spec.BAY_PITCH
            span = spec.BAY_PITCH - spec.UPRIGHT_W
            for k in range(1, spec.LEVELS + 1):
                z = k * spec.LEVEL_H if k < spec.LEVELS else h - spec.BEAM_H / 2
                for yy, tag in ((y_front - s * 0.01, "f"), (y_back + s * 0.01, "b")):
                    beam = box(stage, f"{root}/Racks/{side}/bay_{b:02d}/beam{k}{tag}", (xc, yy, z),
                               (span, spec.BEAM_D, spec.BEAM_H), m_beam)
                    if tag == "f":
                        collide(beam.GetPrim())


def build_pallets(stage, root, pallets, templates, seed=spec.SEED):
    rng = random.Random(seed + 1)
    for p in pallets:
        kind = rng.choice(TEMPLATE_KINDS[p.load])
        path = f"{root}/Pallets/{p.side}_{p.bay:02d}_{p.level}_{p.slot}"
        prim = UsdGeom.Xform.Define(stage, path)
        prim.GetPrim().GetReferences().AddReference(templates[kind])
        prim.GetPrim().SetInstanceable(True)  # instanced so ~750 loads render cheaply
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(p.x, p.face_y, spec.level_base_z(p.level)))
        xf.AddRotateZOp().Set(0.0 if p.side == "L" else 180.0)


def build_labels(stage, root, pallets):
    m_face = material(stage, f"{root}/Looks/label_face", texture=os.path.join(TEX, "label_face.png"), roughness=0.7)
    strip_h = 0.025  # bar height, m
    strip_w = 176 * spec.X_DIM_MM * 1e-3  # 176 modules with quiet zones
    for p in pallets:
        x, y, z = p.label_pose
        s = 1 if p.side == "L" else -1  # left labels face -y, right labels face +y
        yl = y - s * 0.004
        # u runs left to right as seen from the aisle and v up, so u x v faces the aisle (not mirrored)
        u = (s * spec.LABEL_W, 0.0, 0.0)
        v = (0.0, 0.0, spec.LABEL_H)
        p0 = (x - s * spec.LABEL_W / 2, yl, z - spec.LABEL_H / 2)
        name = f"{root}/Labels/{p.side}_{p.bay:02d}_{p.level}_{p.slot}"
        quad(stage, name + "/face", p0, u, v, m_face)
        m_strip = material(stage, f"{root}/Looks/strips/s{p.sscc}",
                           texture=os.path.join(TEX, "strips", f"{p.sscc}.png"), roughness=0.7)
        ys = yl - s * 0.001
        ps = (x - s * strip_w / 2, ys, z - spec.LABEL_H / 2 + 0.006)
        quad(stage, name + "/strip", ps, (s * strip_w, 0.0, 0.0), (0.0, 0.0, strip_h), m_strip)


def build(stage, assets_root, root="/World/Warehouse", pallets=None, labels=True, lights=True,
          x_range=None):
    """Populate the stage. x_range=(x0, x1) keeps only pallets in that span (for quick tests)."""
    UsdGeom.Xform.Define(stage, root)
    pallets = spec.layout() if pallets is None else pallets
    if x_range is not None:
        pallets = [p for p in pallets if x_range[0] <= p.x <= x_range[1]]
    templates = write_templates(assets_root)
    build_shell(stage, root)
    if lights:
        build_lights(stage, root)
    build_racks(stage, root)
    build_pallets(stage, root, pallets, templates)
    if labels:
        build_labels(stage, root, pallets)
    return pallets
