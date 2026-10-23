"""Cut a part wherever the master figure shows another surface that must stay visible.
Example: remove generated fringe hair that hangs over the eyes, using the master's visible face skin.

The zone = master surface points of the given labels (optionally limited by z/y), extended along a
direction (toward the viewer) and a little behind the surface, grown by a margin.

Usage: python3 trim_by_master.py part.stl fig.npz labels.npy out.stl --labels 12 --dir=-y
       [--front 8] [--back 0.6] [--grow 0.25] [--zmin 58 --zmax 72 --ymax 0] [--voxel 0.06]
"""
import argparse, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from skimage import measure

ap = argparse.ArgumentParser()
ap.add_argument('part'); ap.add_argument('fig'); ap.add_argument('labels'); ap.add_argument('out')
ap.add_argument('--labels', dest='labs', required=True); ap.add_argument('--dir', default='-y')
ap.add_argument('--front', type=float, default=8.0); ap.add_argument('--back', type=float, default=0.6)
ap.add_argument('--grow', type=float, default=0.25); ap.add_argument('--voxel', type=float, default=0.06)
ap.add_argument('--zmin', type=float, default=-1e9); ap.add_argument('--zmax', type=float, default=1e9)
ap.add_argument('--ymax', type=float, default=1e9)
a = ap.parse_args(); H = a.voxel
AX = 'xyz'.index(a.dir[1]); SG = 1 if a.dir[0] == '+' else -1

d = np.load(a.fig); V = d['v']; L = np.load(a.labels)
sel = np.isin(L, [int(v) for v in a.labs.split(',')]) & (V[:, 2] >= a.zmin) & (V[:, 2] <= a.zmax) & (V[:, 1] <= a.ymax)
Pm = V[sel]
lo = Pm.min(0) - a.front - a.back - 2; hi = Pm.max(0) + a.front + a.back + 2
shape = np.ceil((hi - lo) / H).astype(int) + 1
seed = np.zeros(shape, bool)
idx = np.round((Pm - lo) / H).astype(int); seed[tuple(idx.T)] = True
# extend along the direction (front) and a bit against it (behind the surface)
zone = seed.copy()
n_f, n_b = int(a.front / H), int(a.back / H)
for i in range(1, n_f + 1): zone |= np.roll(seed, SG * i, axis=AX)
for i in range(1, n_b + 1): zone |= np.roll(seed, -SG * i, axis=AX)
zone = ndi.binary_closing(zone, iterations=2)
zone = ndi.binary_dilation(zone, iterations=max(1, int(round(a.grow / H))))
f = ndi.gaussian_filter(np.pad(zone, 3).astype(np.float32), 0.6)
v, fc, _, _ = measure.marching_cubes(f, 0.5)
Z = mf.Manifold(mf.Mesh(vert_properties=((v - 3) * H + lo).astype(np.float32), tri_verts=fc[:, ::-1].astype(np.uint32)))
m = trimesh.load(a.part)
P = mf.Manifold(mf.Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
R = P - Z
pcs = sorted(R.decompose(), key=lambda q: -q.volume())
keep = [q for q in pcs if q.volume() > 0.02 * pcs[0].volume()]
R = keep[0]
for q in keep[1:]: R = R + q
print('removed mm3', round(P.volume() - R.volume(), 1), 'pieces kept', len(keep), 'dropped', len(pcs) - len(keep))
q = R.to_mesh(); trimesh.Trimesh(q.vert_properties[:, :3], q.tri_verts).export(a.out)
