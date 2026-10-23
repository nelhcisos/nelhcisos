"""Make nested head parts (head <- hair <- cap) that really go on, in the voxel domain.

Assembly direction is straight down (-Z): a part can be put on / lifted off only if no material of the
part below it sits under it. So for a part P sitting on receivers R:
  * material of P that has R above it would collide -> it is given to R instead (looks stay the same),
  * the hollow of P is filled down to R, then R swept downward (+ clearance) is cut out of P,
    so P's inside is an exact imprint of R and P rests on it.

Usage: python3 fit_stack.py head.stl hair.stl cap.stl out_dir [voxel_mm] [clearance_mm]
Writes out_dir/head.stl hair.stl cap.stl (closed meshes) and a report.
"""
import sys, os, json, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from skimage import measure
from PIL import Image, ImageDraw

head_f, hair_f, cap_f, out = sys.argv[1:5]
H = float(sys.argv[5]) if len(sys.argv) > 5 else 0.08
CLEAR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.10
os.makedirs(out, exist_ok=True)

def load(f):
    m = trimesh.load(f, process=True)
    return m

meshes = {k: load(f) for k, f in (('head', head_f), ('hair', hair_f), ('cap', cap_f))}
allv = np.vstack([m.vertices for m in meshes.values()])
lo = allv.min(0) - 1.0; hi = allv.max(0) + 1.0
shape = np.ceil((hi - lo) / H).astype(int) + 1
print('grid', shape, round(np.prod(shape) / 1e6, 1), 'M')

def voxelize(m):
    M = mf.Manifold(mf.Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
    if M.status() != mf.Error.NoError:
        raise SystemExit(f'not a closed mesh: {M.status()}')
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

O = {k: voxelize(m) for k, m in meshes.items()}
# priority where generated parts overlap: cap over hair over head (outer layer wins)
O['hair'] &= ~O['cap']; O['head'] &= ~(O['cap'] | O['hair'])

def sweep_down(A):
    """voxels that have some A voxel at or above them (A extended downward)"""
    return np.flip(np.maximum.accumulate(np.flip(A, 2), axis=2), 2)

def sweep_up(A):
    return np.maximum.accumulate(A, axis=2)

clr = max(1, int(round(CLEAR / H)))
ball = ndi.generate_binary_structure(3, 1)
rep = {}

def seat(part, receivers):
    """make `part` removable straight up from `receivers`; returns (new part, material given to receivers)"""
    R = np.zeros(shape, bool)
    for r in receivers: R |= O[r]
    blocked = O[part] & sweep_down(R)          # part material that has receiver material above it
    # the hollow of the part, filled down to the receiver so the part rests on it
    hollow = sweep_down(O[part]) & sweep_up(R) & ~O[part] & ~R
    filled = O[part] | hollow
    keepout = ndi.binary_dilation(sweep_down(R), ball, iterations=clr)
    new = filled & ~keepout
    give = blocked & ~sweep_down(new)          # blocked material goes to the receiver below, if it is visible
    return new, give

# hair onto head
hair_new, hair_give = seat('hair', ['head'])
O['head'] |= hair_give & ~O['head']
rep['hair -> head: moved to head (mm3)'] = round(hair_give.sum() * H ** 3, 1)
O['hair'] = hair_new
# cap onto hair + head
cap_new, cap_give = seat('cap', ['hair', 'head'])
O['hair'] |= cap_give & ~O['head'] & ~O['hair']
rep['cap -> hair: moved to hair (mm3)'] = round(cap_give.sum() * H ** 3, 1)
O['cap'] = cap_new

# keep the biggest piece of each part
for k in O:
    lab, n = ndi.label(O[k])
    if n > 1:
        sizes = ndi.sum(O[k], lab, range(1, n + 1)); O[k] = lab == (np.argmax(sizes) + 1)
    rep[f'{k} volume mm3'] = round(O[k].sum() * H ** 3, 1)
    rep[f'{k} islands dropped'] = int(max(n - 1, 0))

# check: lifting each part straight up never hits what is under it
for part, recv in (('hair', ['head']), ('cap', ['hair', 'head'])):
    R = np.zeros(shape, bool)
    for r in recv: R |= O[r]
    rep[f'{part} collisions when lifted (voxels)'] = int((O[part] & sweep_down(R)).sum())

for k, occ in O.items():
    f = ndi.gaussian_filter(np.pad(occ, 2).astype(np.float32), 0.6)
    v, fc, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 2) * H + lo
    m = trimesh.Trimesh(v, fc[:, ::-1], process=True)
    m.export(os.path.join(out, f'{k}.stl'))
    rep[f'{k} watertight'] = bool(m.is_watertight)
json.dump(rep, open(os.path.join(out, 'report.json'), 'w'), indent=1, ensure_ascii=False)
for k, v in rep.items(): print(k, v)
