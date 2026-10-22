"""Decimate a mesh with Blender. Usage: python3 decimate.py in.ply out.ply ratio"""
import sys, bpy
src, out, r = sys.argv[1], sys.argv[2], float(sys.argv[3])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.ply_import(filepath=src)
o = bpy.context.scene.objects[0]; bpy.context.view_layer.objects.active = o; o.select_set(True)
md = o.modifiers.new('d', 'DECIMATE'); md.ratio = r; md.use_collapse_triangulate = True
bpy.ops.object.modifier_apply(modifier='d')
print('faces', len(o.data.polygons))
bpy.ops.wm.ply_export(filepath=out, export_selected_objects=True, export_normals=False, export_uv=False, export_colors='NONE')
