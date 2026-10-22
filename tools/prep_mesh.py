"""Load the generated GLB, repair it into a closed solid, scale to target height (mm), orient Z-up.
Keeps per-face colour sampled from the texture (used later to label parts).
Usage: python3 prep_mesh.py in.glb out.npz height_mm"""
import sys, numpy as np, trimesh, manifold3d as mf

src, out, H = sys.argv[1], sys.argv[2], float(sys.argv[3])
s = trimesh.load(src); g = list(s.geometry.values())[0]
uv = g.visual.uv; tex = np.asarray(g.visual.material.baseColorTexture.convert('RGB'))
# per-face colour = texture colour at the face's UV centroid (before vertices are merged)
fuv = uv[g.faces].mean(1)
th, tw = tex.shape[:2]
px = np.clip((fuv[:, 0] % 1) * (tw - 1), 0, tw - 1).astype(int)
py = np.clip((1 - fuv[:, 1] % 1) * (th - 1), 0, th - 1).astype(int)
fcol = tex[py, px]

m = trimesh.Trimesh(g.vertices, g.faces, process=False)
m.merge_vertices()
keep = np.ones(len(m.faces), bool)
# drop faces on non-manifold edges (shared by >2 faces), then let hole filling close the gaps
e = m.edges_sorted
u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
bad_edge = cnt[inv] > 2
keep[m.edges_face[bad_edge]] = False
m.update_faces(keep); fcol = fcol[keep]
m.remove_unreferenced_vertices()
parts = m.split(only_watertight=False)
print('components', [len(p.faces) for p in parts][:5])
trimesh.repair.fix_winding(m)
trimesh.repair.fill_holes(m)
n_new = len(m.faces) - len(fcol)
if n_new > 0: fcol = np.vstack([fcol, np.repeat(fcol[:1], n_new, 0)])
print('watertight', m.is_watertight, 'winding', m.is_winding_consistent)

# orient: glTF is Y-up -> make Z up
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
v = m.vertices @ R.T
v -= [(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].min()]
v *= H / (v[:, 2].max())
if np.linalg.det(R) < 0: pass
mm = mf.Manifold(mf.Mesh(vert_properties=v.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
print('manifold status', mm.status(), 'volume mm3', round(mm.volume(), 1))
np.savez_compressed(out, v=v, f=m.faces, c=fcol)
print('bounds', v.min(0).round(2), v.max(0).round(2))
