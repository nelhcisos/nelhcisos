"""Clean voxel remesh + decimate of an STL/PLY solid in Blender (no scaling).
Usage: python3 remesh_solid.py in.stl out.ply voxel_mm decimate_ratio"""
import sys, bpy
src, out, vox, ratio = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
bpy.ops.wm.read_factory_settings(use_empty=True)
(bpy.ops.wm.stl_import if src.endswith('.stl') else bpy.ops.wm.ply_import)(filepath=src)
o = bpy.context.scene.objects[0]; bpy.context.view_layer.objects.active = o; o.select_set(True)
o.data.remesh_voxel_size = vox; bpy.ops.object.voxel_remesh()
md = o.modifiers.new('d', 'DECIMATE'); md.ratio = ratio; md.use_collapse_triangulate = True
bpy.ops.object.modifier_apply(modifier='d')
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.object.mode_set(mode='OBJECT')
print('faces', len(o.data.polygons))
bpy.ops.wm.ply_export(filepath=out, export_selected_objects=True, export_normals=False, export_uv=False, export_colors='NONE')
