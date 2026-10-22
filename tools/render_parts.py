"""Render part STLs coloured by part, assembled and exploded.
Usage: python3 render_parts.py parts_dir out_prefix [--explode 1.0] [--angles 0,180] [--res 900] [--only a,b]"""
import sys, os, math, bpy, mathutils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parts import COLOR
a = sys.argv[1:]; pdir, out = a[0], a[1]
opt = lambda k, d: a[a.index(k) + 1] if k in a else d
explode = float(opt('--explode', '0')); angles = [float(v) for v in opt('--angles', '0,180').split(',')]
res = int(opt('--res', '900')); only = opt('--only', ''); elev = float(opt('--elev', '14'))
only = set(only.split(',')) if only else None
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
# explode directions: away from the figure axis + up for head parts
DIR = {'base': (0, 0, -1.2), 'croc_R': (-0.6, 0, -0.5), 'croc_L': (0.6, 0, -0.5), 'pants': (0, 0, -0.2), 'shirt': (0, 0, 0.3),
       'tshirt': (0, -1.2, 0.6), 'neck': (0, 0, 0.9), 'arm_R': (-1.4, 0, 0.3), 'arm_L': (1.4, 0, 0.3), 'glove_R': (-1.8, 0, -0.2),
       'glove_L': (1.8, 0, -0.2), 'head': (0, 0, 1.4), 'ear_R': (-1.6, 0, 1.4), 'ear_L': (1.6, 0, 1.4), 'hair': (0, 0.3, 2.1),
       'cap': (0, 0.4, 2.9), 'mask': (0, -1.5, 1.2)}
for f in sorted(os.listdir(pdir)):
    if not f.endswith('.stl'): continue
    name = f[:-4]
    if only and name not in only: continue
    bpy.ops.wm.stl_import(filepath=os.path.join(pdir, f))
    o = bpy.context.selected_objects[0]; o.name = name
    m = bpy.data.materials.new(name); m.use_nodes = True
    c = COLOR.get(name, (200, 200, 200)); bsdf = m.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*[(v / 255) ** 2.2 for v in c], 1); bsdf.inputs['Roughness'].default_value = 0.45
    o.data.materials.append(m)
    dx = DIR.get(name, (0, 0, 0)); o.location = (dx[0] * 10 * explode, dx[1] * 10 * explode, dx[2] * 10 * explode)
    for p in o.data.polygons: p.use_smooth = False
bpy.context.view_layer.update()
obs = [o for o in sc.objects if o.type == 'MESH']
pts = [o.matrix_world @ mathutils.Vector(c) for o in obs for c in o.bound_box]
mn = mathutils.Vector([min(p[k] for p in pts) for k in range(3)]); mx = mathutils.Vector([max(p[k] for p in pts) for k in range(3)])
ctr = (mn + mx) / 2; size = max(mx - mn)
sc.render.engine = 'CYCLES'; sc.cycles.samples = 32; sc.cycles.device = 'CPU'
sc.render.resolution_x = res; sc.render.resolution_y = int(res * float(opt('--aspect', '1.2')))
sc.world = bpy.data.worlds.new('w'); sc.world.use_nodes = True
bg = sc.world.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.16, 0.16, 0.18, 1); bg.inputs[1].default_value = 0.6
sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = size * 1.1
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sun.data.energy = 3.5; sc.collection.objects.link(sun)
for ang in angles:
    r = math.radians(ang); d = size * 3
    er = math.radians(elev)
    cam.location = ctr + mathutils.Vector((math.sin(r) * d * math.cos(er), -math.cos(r) * d * math.cos(er), d * math.sin(er)))
    cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sun.rotation_euler = cam.rotation_euler.copy(); sun.rotation_euler.z += 0.6; sun.rotation_euler.x -= 0.3
    sc.render.filepath = f'{out}_{int(ang):03d}.png'; bpy.ops.render.render(write_still=True)
