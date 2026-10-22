"""Transfer texture colour from the generated GLB to the clean solid; rotate so the figure faces -Y.
Usage: python3 colorize.py tripo.glb solid.ply height_mm rotz_deg out.npz"""
import sys, numpy as np, trimesh
from scipy.spatial import cKDTree
src, solid, H, rotz, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
s = trimesh.load(src); g = list(s.geometry.values())[0]
tex = np.asarray(g.visual.material.baseColorTexture.convert('RGB')); th, tw = tex.shape[:2]
uv = g.visual.uv
vc = tex[np.clip(((1 - uv[:, 1] % 1) * (th - 1)).astype(int), 0, th - 1), np.clip((uv[:, 0] % 1 * (tw - 1)).astype(int), 0, tw - 1)]
# same transform as remesh.py: glTF (x,y,z) -> Blender (x,-z,y), scale, centre
v = g.vertices[:, [0, 2, 1]] * [1, -1, 1]
# sample extra points on faces so colour lookup is dense
pts, fi = trimesh.sample.sample_surface_even(trimesh.Trimesh(v, g.faces, process=False), 3_000_000)
bc = trimesh.triangles.points_to_barycentric(v[g.faces[fi]], pts)
uvs = (uv[g.faces[fi]] * bc[:, :, None]).sum(1)
pc = tex[np.clip(((1 - uvs[:, 1] % 1) * (th - 1)).astype(int), 0, th - 1), np.clip((uvs[:, 0] % 1 * (tw - 1)).astype(int), 0, tw - 1)]
P = np.vstack([v, pts]); C = np.vstack([vc, pc])
sc = H / (P[:, 2].max() - P[:, 2].min())
P = P * sc; P -= [(P[:, 0].min() + P[:, 0].max()) / 2, (P[:, 1].min() + P[:, 1].max()) / 2, P[:, 2].min()]
m = trimesh.load(solid)
d, idx = cKDTree(P).query(m.vertices, k=4)
w = 1 / (d + 1e-4); col = (C[idx] * w[:, :, None]).sum(1) / w.sum(1)[:, None]
print('max lookup dist', d[:, 0].max().round(3))
a = np.radians(rotz); Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
V = m.vertices @ Rz.T
np.savez_compressed(out, v=V, f=m.faces, c=col.astype(np.uint8))
trimesh.Trimesh(V, m.faces, vertex_colors=col.astype(np.uint8), process=False).export(out.replace('.npz', '.ply'))
