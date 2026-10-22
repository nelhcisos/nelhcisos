"""Decimate heavy part STLs (Blender) down to a face budget, keep them closed.
Usage: python3 slim_parts.py in_dir out_dir max_faces"""
import sys, os, bpy
src, dst, budget = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    if not f.endswith('.stl'): continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=os.path.join(src, f))
    o = bpy.context.selected_objects[0]; bpy.context.view_layer.objects.active = o
    n = len(o.data.polygons)
    if n > budget:
        md = o.modifiers.new('d', 'DECIMATE'); md.ratio = budget / n; md.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier='d')
    print(f, n, '->', len(o.data.polygons))
    bpy.ops.wm.stl_export(filepath=os.path.join(dst, f), export_selected_objects=True, ascii_format=False)
