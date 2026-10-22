"""Make the generated GLB a clean closed solid via Blender voxel remesh.
Usage: python3 remesh.py in.glb out.ply height_mm voxel_mm"""
import sys, bpy, bmesh, math
src, out, H, vox = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
bpy.ops.object.select_all(action='DESELECT')
for o in obs: o.select_set(True)
bpy.context.view_layer.objects.active = obs[0]
if len(obs) > 1: bpy.ops.object.join()
o = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
zs = [v.co.z for v in o.data.vertices]; s = H / (max(zs) - min(zs))
o.scale = (s, s, s); bpy.ops.object.transform_apply(scale=True)
xs=[v.co.x for v in o.data.vertices]; ys=[v.co.y for v in o.data.vertices]; zs=[v.co.z for v in o.data.vertices]
o.location = (-(min(xs)+max(xs))/2, -(min(ys)+max(ys))/2, -min(zs)); bpy.ops.object.transform_apply(location=True)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles(threshold=0.001); bpy.ops.object.mode_set(mode='OBJECT')
o.data.remesh_voxel_size = vox; o.data.use_remesh_fix_poles = False
bpy.ops.object.voxel_remesh()
print('faces after remesh', len(o.data.polygons))
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.ply_export(filepath=out, export_selected_objects=True, export_normals=False, export_uv=False, export_colors='NONE', ascii_format=False)
