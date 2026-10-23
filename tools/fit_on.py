"""Make a part fit onto a receiver, put on / taken off along one straight direction, crisp outer surfaces.

  part wins: receiver material inside the part or beyond it (in the take-off direction) is removed
  hollow  = space between the part and the receiver along the direction -> filled, so the part rests on it
            (--ring: for a cap-like part only inside its closed ring in each slice across the direction)
  keepout = receiver swept against the take-off direction, grown by the clearance -> cut from the part
Only hollow/keepout come from voxels; visible surfaces stay the original meshes.

Usage: python3 fit_on.py receiver.stl part.stl out_dir --dir +z [--voxel 0.06] [--clear 0.10] [--ring]
       --dir is the take-off direction of the part: +z = lift up, -y = pull forward, +x / -x sideways.
Writes out_dir/receiver.stl, out_dir/part.stl, report.json
"""
import sys, os, json, argparse, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from skimage import measure
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('receiver'); ap.add_argument('part'); ap.add_argument('out')
ap.add_argument('--dir', default='+z'); ap.add_argument('--voxel', type=float, default=0.06)
ap.add_argument('--clear', type=float, default=0.10); ap.add_argument('--ring', action='store_true')
ap.add_argument('--part-loses', action='store_true', help='receiver stays untouched; the part is cut wherever it would collide')
ap.add_argument('--wins-halfwidth', type=float, default=0, help='part wins over the receiver only where |x| < this (0 = everywhere)')
a = ap.parse_args()
H = a.voxel; os.makedirs(a.out, exist_ok=True)
AX = 'xyz'.index(a.dir[1]); SG = 1 if a.dir[0] == '+' else -1

def man(m):
    M = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    assert M.status() == mf.Error.NoError, M.status()
    return M
def tm(M):
    q = M.to_mesh(); return trimesh.Trimesh(q.vert_properties[:, :3], q.tri_verts, process=False)

recv = man(trimesh.load(a.receiver)); part = man(trimesh.load(a.part))
b1, b2 = tm(recv).bounds, tm(part).bounds
lo = np.minimum(b1[0], b2[0]) - 1; hi = np.maximum(b1[1], b2[1]) + 1
shape = np.ceil((hi - lo) / H).astype(int) + 1
print('grid', shape, round(np.prod(shape) / 1e6, 1), 'M')

def vox(M):
    occ = np.zeros(shape, bool)
    for k in range(shape[2]):
        polys = M.slice(lo[2] + k * H).to_polygons()
        if not polys: continue
        acc = np.zeros((shape[1], shape[0]), bool)
        for p in polys:
            im = Image.new('1', (shape[0], shape[1]), 0)
            ImageDraw.Draw(im).polygon([((q[0] - lo[0]) / H, (q[1] - lo[1]) / H) for q in p], fill=1)
            acc ^= np.asarray(im, bool)
        occ[:, :, k] = acc.T
    return occ

def mesh_of(occ, sigma=0.6):
    f = ndi.gaussian_filter(np.pad(occ, 3).astype(np.float32), sigma)
    v, fc, _, _ = measure.marching_cubes(f, 0.5)
    return man(trimesh.Trimesh((v - 3) * H + lo, fc[:, ::-1], process=True))

def sweep(A, sign):
    """extend A along +axis (sign=+1) or -axis (sign=-1)"""
    if sign > 0: return np.maximum.accumulate(A, axis=AX)
    return np.flip(np.maximum.accumulate(np.flip(A, AX), axis=AX), AX)

P = vox(part)
# part wins: receiver inside the part or beyond it in the take-off direction goes away
wins = P | sweep(P, SG)
if a.wins_halfwidth > 0:
    xs = lo[0] + np.arange(shape[0]) * H
    wins &= (np.abs(xs) < a.wins_halfwidth)[:, None, None]
if not a.part_loses:
    recv = recv - mesh_of(wins, 0.5)
_pcs = [q for q in recv.decompose() if q.volume() > 5.0]
recv = _pcs[0]
for _q in _pcs[1:]: recv = recv + _q
R = vox(recv)
behind = sweep(P, -SG)                      # space the part covers, looking against the take-off direction
if a.ring:
    ring = np.zeros(shape, bool)
    for k in range(shape[AX]):
        sl = [slice(None)] * 3; sl[AX] = k; sl = tuple(sl)
        if P[sl].any(): ring[sl] = ndi.binary_fill_holes(P[sl])
    behind &= ring
hollow = behind & sweep(R, SG) & ~P & ~R     # between part and receiver
keepout = ndi.binary_dilation(sweep(R, -SG), ndi.generate_binary_structure(3, 1), iterations=max(1, int(round(a.clear / H))))
part_f = (part + mesh_of(hollow & ~keepout)) - mesh_of(keepout, 0.5)
pcs = sorted(part_f.decompose(), key=lambda q: -q.volume()); part_f = pcs[0]
Pf, Rf = vox(part_f), vox(recv)
rep = {'part volume mm3': round(part_f.volume(), 1), 'receiver volume mm3': round(recv.volume(), 1),
       'hollow filled mm3': round((hollow & ~keepout).sum() * H ** 3, 1),
       'collisions on take-off (voxels)': int((Pf & sweep(Rf, -SG)).sum()), 'part pieces dropped': len(pcs) - 1}
tm(part_f).export(os.path.join(a.out, 'part.stl')); tm(recv).export(os.path.join(a.out, 'receiver.stl'))
json.dump(rep, open(os.path.join(a.out, 'report.json'), 'w'), indent=1)
for k, v in rep.items(): print(k, v)
