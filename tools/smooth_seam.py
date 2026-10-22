"""Taubin-smooth the band around the removed-bow patch on a clean mesh, re-attach colours.
Usage: python3 smooth_seam.py remeshed.ply fig2.npz out.npz"""
import sys, numpy as np, trimesh, scipy.sparse as sp, manifold3d as mf
from scipy.spatial import cKDTree
m = trimesh.load(sys.argv[1], process=False); ref = np.load(sys.argv[2])
V, F = np.asarray(m.vertices, float).copy(), np.asarray(m.faces)
CEN, KC, KR = ref['cen'], ref['knot_c'], ref['knot_r']
rel = V - CEN; rr = np.linalg.norm(rel, axis=1)
az = np.degrees(np.arctan2(rel[:, 0], rel[:, 1])); el = np.degrees(np.arcsin(np.clip(rel[:, 2] / np.maximum(rr, 1e-9), -1, 1)))
ev = np.sqrt(((az - KC[0]) / KR[0]) ** 2 + ((el - KC[1]) / KR[1]) ** 2)
band = np.clip(1 - np.abs(ev - 1) / 0.14, 0, 1) * (rr < 30) * (rel[:, 1] > 0)   # weight 1 on the seam, 0 away
n = len(V); e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr(); A.data[:] = 1
deg = np.asarray(A.sum(1)).ravel()
for it in range(25):
    for lam in (0.55, -0.58):
        V += (lam * band)[:, None] * ((A @ V) / deg[:, None] - V)
print('band verts', int((band > 0).sum()))
M = mf.Manifold(mf.Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
print('manifold', M.status(), 'pieces', len(M.decompose()), 'vol', round(M.volume(), 1))
_, j = cKDTree(ref['v']).query(V)
np.savez_compressed(sys.argv[3], v=V, f=F, c=ref['c'][j])
