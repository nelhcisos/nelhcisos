"""Render preview images of mesh files (GLB/STL/OBJ) from several angles with Blender Cycles (CPU).
Usage: python3 render.py out_prefix file1 [file2 ...] [--angles 0,90,180,270] [--res 600]
"""
import sys, math, bpy, mathutils

args = sys.argv[1:]
angles = [0, 90, 180, 270]; res = 600; files = []; clay = False; rotz = 0.0
i = 0
while i < len(args):
    if args[i] == '--angles': angles = [float(a) for a in args[i+1].split(',')]; i += 2
    elif args[i] == '--res': res = int(args[i+1]); i += 2
    elif args[i] == '--clay': clay = True; i += 1
    elif args[i] == '--rotz': rotz = float(args[i+1]); i += 2
    else: files.append(args[i]); i += 1
out = files.pop(0)

bpy.ops.wm.read_factory_settings(use_empty=True)
for f in files:
    if f.endswith('.glb') or f.endswith('.gltf'): bpy.ops.import_scene.gltf(filepath=f)
    elif f.endswith('.stl'): bpy.ops.wm.stl_import(filepath=f)
    elif f.endswith('.obj'): bpy.ops.wm.obj_import(filepath=f)
    elif f.endswith('.ply'): bpy.ops.wm.ply_import(filepath=f)

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if rotz:
    for o in bpy.context.scene.objects:
        if o.parent is None: o.rotation_mode='XYZ'; o.rotation_euler.z += math.radians(rotz)
    bpy.context.view_layer.update()
if clay:
    m = bpy.data.materials.new('clay'); m.use_nodes = True
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.75, 0.75, 0.78, 1)
    m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.6
    for o in meshes:
        o.data.materials.clear(); o.data.materials.append(m)
pts = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
mn = mathutils.Vector([min(p[k] for p in pts) for k in range(3)])
mx = mathutils.Vector([max(p[k] for p in pts) for k in range(3)])
ctr = (mn + mx) / 2; size = max(mx - mn)

sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.samples = 24; sc.cycles.device = 'CPU'
sc.render.resolution_x = res; sc.render.resolution_y = int(res * 1.25)
sc.world = bpy.data.worlds.new('w'); sc.world.use_nodes = True
sc.world.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
sc.world.node_tree.nodes['Background'].inputs[1].default_value = (0.35 if clay else 0.9)
sc.view_settings.view_transform = 'Standard'

cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.ortho_scale = size * 1.15
light = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); light.data.energy = (4 if clay else 3)
sc.collection.objects.link(light)

# Blender up axis = Z. Figure front assumed facing -Y.
for a in angles:
    r = math.radians(a); d = size * 3
    cam.location = ctr + mathutils.Vector((math.sin(r) * d, -math.cos(r) * d, size * 0.15))
    cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
    light.rotation_euler = cam.rotation_euler.copy(); light.rotation_euler.z += 0.5; light.rotation_euler.x -= 0.4
    sc.render.filepath = f'{out}_{int(a):03d}.png'
    bpy.ops.render.render(write_still=True)
print('done')
