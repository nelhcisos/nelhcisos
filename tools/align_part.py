"""Fit a separately generated part (GLB) onto its place on the master figure.

Similarity transform (scale + rotation + shift) found by trimmed ICP that goes FROM the master's visible
surface points TO the part: every visible master point looks for the nearest point of the part, so the
part's hidden sides (inside of the cap, back of the mask...) do not disturb the fit.
Many start orientations are tried; the best trimmed error wins.

Usage: python3 align_part.py part.glb fig.npz labels.npy out.npz --labels 16 [--zmin 6.9 --zmax 12 --xsign -1]
       [--mirror]   (mirror the part in X before fitting, e.g. to make a right ear from a left one)
"""
import sys, argparse, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument('part'); ap.add_argument('fig'); ap.add_argument('labels'); ap.add_argument('out')
ap.add_argument('--labels', dest='labs', default='')
ap.add_argument('--zmin', type=float, default=-1e9); ap.add_argument('--zmax', type=float, default=1e9)
ap.add_argument('--xsign', type=float, default=0); ap.add_argument('--mirror', action='store_true')
ap.add_argument('--trim', type=float, default=0.8)
a = ap.parse_args()

s = trimesh.load(a.part)
g = s.dump(concatenate=True) if isinstance(s, trimesh.Scene) else s
m = trimesh.Trimesh(g.vertices[:, [0, 2, 1]] * [1, -1, 1], g.faces, process=True)   # glTF Y-up -> Z-up
if a.mirror:
    m.vertices[:, 0] *= -1; m.invert()
src_pts = m.sample(80000)

d = np.load(a.fig); V = d['v']; L = np.load(a.labels)
sel = np.ones(len(V), bool)
if a.labs: sel &= np.isin(L, [int(v) for v in a.labs.split(',')])
sel &= (V[:, 2] >= a.zmin) & (V[:, 2] <= a.zmax)
if a.xsign: sel &= np.sign(V[:, 0]) == np.sign(a.xsign)
T = V[sel]
rng = np.random.default_rng(0)
T = T[rng.choice(len(T), min(len(T), 15000), replace=False)]
print('target points', len(T), 'extent', np.ptp(T, 0).round(1))

def umeyama(P, Q):
    """similarity that maps P onto Q"""
    mp, mq = P.mean(0), Q.mean(0); Pc, Qc = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(Qc.T @ Pc / len(P))
    Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ Dm @ Vt; sc = (S * np.diag(Dm)).sum() / (Pc ** 2).sum(1).mean()
    return sc, R, mq - sc * R @ mp

def icp(sc, R, t, iters=60):
    for it in range(iters):
        X = sc * src_pts @ R.T + t
        tree = cKDTree(X)
        dist, j = tree.query(T)
        keep = dist <= np.quantile(dist, a.trim)
        sc, R, t = umeyama(src_pts[j[keep]], T[keep])
    X = sc * src_pts @ R.T + t
    dist, _ = cKDTree(X).query(T)
    return np.sqrt((np.sort(dist)[:int(len(dist) * a.trim)] ** 2).mean()), sc, R, t

# start guesses: target size vs part size, many orientations
ext_t = np.sqrt(np.linalg.eigvalsh(np.cov(T.T))).max()
ext_s = np.sqrt(np.linalg.eigvalsh(np.cov(src_pts.T))).max()
starts = [Rot.from_euler('zyx', [z, y, x], degrees=True).as_matrix()
          for z in range(0, 360, 45) for y in (0, 90, 180, 270) for x in (0, 90)]
best = None
for R0 in starts:
    for sf in (0.9, 1.15):
        sc0 = ext_t / ext_s * sf
        t0 = T.mean(0) - sc0 * R0 @ src_pts.mean(0)
        err, sc, R, t = icp(sc0, R0, t0, iters=25)
        if best is None or err < best[0]: best = (err, sc, R, t)
err, sc, R, t = icp(*best[1:], iters=80)
print(f'fit error {err:.3f} mm, scale {sc:.4f}')
M = np.eye(4); M[:3, :3] = sc * R; M[:3, 3] = t
m.apply_transform(M)
np.savez_compressed(a.out, v=m.vertices, f=m.faces, err=err, scale=sc)
m.export(a.out.replace('.npz', '.stl'))
