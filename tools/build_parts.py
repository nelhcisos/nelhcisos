"""Split the figure into printable parts with pegs and sockets.

Steps
1. Voxelise the solid (slice by slice).
2. Each voxel inside (and in a 1 mm band outside) takes the label of the nearest labelled surface point.
3. Rules turn surface labels into volumes: cap / hair / mask are shells of fixed thickness over the
   head core; body parts are cut by flat planes; a neck cylinder joins shirt and head.
4. Each part's region is meshed (marching cubes, small clearance at the seams) and intersected with the
   original solid, so the outside surface keeps the full detail.
5. Pegs (male) and sockets (female, with clearance) are added at each joint.

Usage: python3 build_parts.py fig.npz labels.npy out_dir
"""
import sys, os, json, time
import numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parts import PARTS, ID, Z_BASE, Z_CROC, Z_HEM, Z_HEAD

src, labf, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)

H = 0.12          # voxel size, mm
BAND = 1.0        # labelled band outside the surface, mm
CLEAR_SEAM = 0.65 # iso level for region meshes (0.5 = touching; 0.65 ~ 0.09 mm total gap)
PEG_CLEAR = 0.10  # radial clearance of sockets, mm
PEG_DEPTH_CLEAR = 0.25

d = np.load(src); V, F = d['v'], d['f']
SL = np.load(labf).astype(np.int16)
SOLID = mf.Manifold(mf.Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))

# ---------------------------------------------------------------- 1. voxelise
lo = V.min(0) - BAND - 3 * H
hi = V.max(0) + BAND + 3 * H
shape = np.ceil((hi - lo) / H).astype(int) + 1
log('grid', shape, np.prod(shape) / 1e6, 'M voxels')
occ = np.zeros(shape, bool)
for k in range(shape[2]):
    zc = lo[2] + k * H
    polys = SOLID.slice(zc).to_polygons()
    if not polys: continue
    acc = np.zeros((shape[1], shape[0]), bool)
    for p in polys:
        im = Image.new('1', (shape[0], shape[1]), 0)
        ImageDraw.Draw(im).polygon([((q[0] - lo[0]) / H, (q[1] - lo[1]) / H) for q in p], fill=1)
        acc ^= np.asarray(im, bool)
    occ[:, :, k] = acc.T
log('occupied', occ.sum() / 1e6, 'M  ->', round(occ.sum() * H ** 3, 1), 'mm3 (solid', round(SOLID.volume(), 1), ')')

d_out = ndi.distance_transform_edt(~occ) * H
d_in = ndi.distance_transform_edt(occ) * H
work = occ | (d_out <= BAND)

# ---------------------------------------------------------------- 2. nearest surface label
good = SL > 0
tree = cKDTree(V[good]); labs_good = SL[good]
# fill undecided surface vertices
if (~good).any():
    _, j = tree.query(V[~good]); SL[~good] = labs_good[j]
idx = np.argwhere(work)
P = lo + idx * H
dist, j = tree.query(P, workers=-1)
L0 = labs_good[j]
log('nearest-label done')

# ---------------------------------------------------------------- 3. rules
x, y, z = P.T
inside = occ[idx[:, 0], idx[:, 1], idx[:, 2]]
depth = np.where(inside, dist, 0.0)
L = L0.copy()
c = lambda n: ID[n]
head_zone = z >= Z_HEAD - 0.8
# head shells over a core
cap, hair, mask, head = c('cap'), c('hair'), c('mask'), c('head')
m = head_zone & (L0 == cap) & (depth >= 1.4) & (depth < 2.6); L[m] = hair
m = head_zone & (L0 == cap) & (depth >= 2.6); L[m] = head
m = head_zone & (L0 == hair) & (depth >= 2.0); L[m] = head
m = head_zone & (L0 == mask) & (depth >= 1.0); L[m] = head
for e in ('ear_R', 'ear_L'):
    m = (L0 == c(e)) & (depth >= 3.0); L[m] = head
# body: flat cuts, limbs keep their nearest-surface volume
body_set = np.isin(L0, [c('base'), c('croc_R'), c('croc_L'), c('pants'), c('shirt')]) | (~head_zone & np.isin(L0, [head, hair, mask, cap]))
plane = np.where(z < Z_BASE, c('base'), np.where(z < Z_CROC, np.where(x < 0, c('croc_R'), c('croc_L')),
                 np.where(z < Z_HEM, c('pants'), c('shirt'))))
L[body_set] = plane[body_set]
m = (L0 == c('tshirt')) & (depth >= 1.5); L[m] = c('shirt')
# neck cylinder: from inside the shirt up into the head
nk = np.argwhere(SL == c('neck'))
neck_xy = V[SL == c('neck')][:, :2].mean(0) if len(nk) else np.array([0.0, 0.0])
sl55 = SOLID.slice(55.0).to_polygons()
if sl55:
    big = max(sl55, key=lambda p: len(p)); neck_xy = np.asarray(big).mean(0)
NECK_R, NECK_Z0, NECK_Z1 = 2.8, 51.5, 57.5
m = (np.hypot(x - neck_xy[0], y - neck_xy[1]) < NECK_R) & (z > NECK_Z0) & (z < NECK_Z1) & np.isin(L, [c('shirt'), head, c('neck'), c('tshirt'), hair, mask]) & inside
L[m] = c('neck')
m = (L == c('neck')) & ((z < NECK_Z0) | (z > NECK_Z1)); L[m] = np.where(z[m] < 55, c('shirt'), head)
log('neck axis', neck_xy.round(2))

G = np.zeros(shape, np.int16)
G[idx[:, 0], idx[:, 1], idx[:, 2]] = L
del L0, P, dist, j, x, y, z

# ---------------------------------------------------------------- cleanup: one connected piece per part
for k in PARTS:
    lab, n = ndi.label(G == k)
    if n <= 1: continue
    sizes = ndi.sum(np.ones_like(lab, np.int8), lab, range(1, n + 1))
    keep = np.argmax(sizes) + 1
    G[(lab > 0) & (lab != keep)] = 0
    log(PARTS[k], 'removed', n - 1, 'islands', int(sizes.sum() - sizes.max()), 'vox')
holes = (G == 0) & work
if holes.any():
    _, (ii, jj, kk) = ndi.distance_transform_edt(G == 0, return_indices=True)
    G[holes] = G[ii[holes], jj[holes], kk[holes]]
    del ii, jj, kk
np.save(os.path.join(out, 'voxels.npy'), G)
json.dump({'lo': lo.tolist(), 'h': H, 'shape': shape.tolist()}, open(os.path.join(out, 'voxels.json'), 'w'))

# ---------------------------------------------------------------- 4. part meshes
def region_mesh(k):
    pts = np.argwhere(G == k)
    a = np.maximum(pts.min(0) - 4, 0); b = np.minimum(pts.max(0) + 5, shape)
    sub = (G[a[0]:b[0], a[1]:b[1], a[2]:b[2]] == k).astype(np.float32)
    sub = np.pad(sub, 2)
    f = ndi.gaussian_filter(sub, 1.0)
    vv, ff, _, _ = measure.marching_cubes(f, CLEAR_SEAM)
    vv = (vv - 2 + a) * H + lo
    ff = ff[:, ::-1]  # outward normals
    return mf.Manifold(mf.Mesh(vert_properties=vv.astype(np.float32), tri_verts=ff.astype(np.uint32)))

parts = {}
for k, name in PARTS.items():
    if not (G == k).any(): log('MISSING', name); continue
    R = region_mesh(k)
    if R.status() != mf.Error.NoError: log('region error', name, R.status())
    parts[name] = SOLID ^ R
    log(name, 'vol', round(parts[name].volume(), 1), 'pieces', len(parts[name].decompose()))

# ---------------------------------------------------------------- 5. pegs
idx_of = lambda p: np.clip(np.round((np.asarray(p) - lo) / H).astype(int), 0, shape - 1)
lab_at = lambda p: G[tuple(idx_of(p).T)]

def cylinder(p, n, r, z0, z1, seg=40):
    n = n / np.linalg.norm(n)
    cyl = mf.Manifold.cylinder(z1 - z0, r, r, seg).translate([0, 0, z0])
    # rotate +Z onto n
    zax = np.array([0, 0, 1.0]); v = np.cross(zax, n); s = np.linalg.norm(v); cth = zax @ n
    if s < 1e-8:
        Rm = np.eye(3) if cth > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * ((1 - cth) / s ** 2)
    M = np.hstack([Rm, np.asarray(p).reshape(3, 1)])
    return cyl.transform(M.tolist())

def peg(male, female, dia, length, anchor=None, axis=None, at=None):
    """Peg on `male` going `length` mm into `female`. Socket in female gets clearance."""
    km, kf = ID[male], ID[female]
    a = (G == km); b = (G == kf)
    iface = a & ndi.binary_dilation(b, iterations=1) & occ
    pts = np.argwhere(iface)
    if len(pts) == 0: return f'{male}->{female}: no contact'
    if at is not None:
        pc = np.asarray(at, float)
        sel = pts[np.argmin(np.linalg.norm(lo + pts * H - pc, axis=1))]
    else:
        # deepest interface voxel = farthest from the outer surface
        sel = pts[np.argmax(d_in[tuple(pts.T)])]
    p = lo + sel * H
    if axis is None:
        near = lambda msk: np.argwhere(msk[max(sel[0]-25,0):sel[0]+26, max(sel[1]-25,0):sel[1]+26, max(sel[2]-25,0):sel[2]+26]) + np.maximum(sel - 25, 0)
        nb, na = near(b), near(a)
        n = (lo + nb * H).mean(0) - (lo + na * H).mean(0)
    else:
        n = np.asarray(axis, float)
    n /= np.linalg.norm(n)
    anchor = anchor or length
    # shrink until the peg sits fully inside male (anchor) and female (tip)
    for scale in (1.0, 0.85, 0.7, 0.55, 0.45):
        r = dia * scale / 2; Lf = length * max(scale, 0.6); La = anchor * max(scale, 0.6)
        ok = True
        for t in np.linspace(-La, Lf + PEG_DEPTH_CLEAR, 12):
            for ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(n, u)
                q = p + n * t + (u * np.cos(ang) + w * np.sin(ang)) * (r + PEG_CLEAR + 0.25)
                want = km if t < 0 else kf
                if lab_at(q) != want or not occ[tuple(idx_of(q))]: ok = False; break
            if not ok: break
        if ok: break
    if not ok: return f'{male}->{female}: no room for a peg (glue joint)'
    pin = cylinder(p, n, r, -La, Lf)
    hole = cylinder(p, n, r + PEG_CLEAR, -0.05, Lf + PEG_DEPTH_CLEAR)
    parts[male] = parts[male] + (pin ^ (SOLID))
    parts[female] = parts[female] - hole
    return f'{male}->{female}: d={2*r:.2f} mm, depth {Lf:.1f} mm at {np.round(p,1).tolist()}'

JOINTS = [
    ('croc_R', 'base', 2.0, 2.0), ('croc_L', 'base', 2.0, 2.0),
    ('pants', 'croc_R', 2.4, 2.2), ('pants', 'croc_L', 2.4, 2.2),
    ('pants', 'shirt', 4.0, 4.0),
    ('neck', 'shirt', 2.6, 2.5), ('neck', 'head', 2.6, 3.0),
    ('tshirt', 'shirt', 1.2, 1.2),
    ('arm_R', 'shirt', 1.6, 2.5), ('arm_L', 'shirt', 1.6, 2.5),
    ('glove_R', 'arm_R', 1.4, 2.0), ('glove_L', 'arm_L', 1.4, 2.0),
    ('ear_R', 'head', 1.4, 1.5), ('ear_L', 'head', 1.4, 1.5),
    ('hair', 'head', 2.0, 2.0), ('cap', 'hair', 1.6, 1.0), ('cap', 'head', 2.0, 2.0),
    ('mask', 'head', 1.2, 1.2),
]
report = []
for male, female, dia, ln in JOINTS:
    if male in parts and female in parts:
        msg = peg(male, female, dia, ln); report.append(msg); log(msg)

# ---------------------------------------------------------------- 6. export
summary = {}
for name, M in parts.items():
    mesh = M.to_mesh()
    tm = trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=False)
    tm.export(os.path.join(out, f'{name}.stl'))
    ext = tm.extents
    summary[name] = {'volume_mm3': round(M.volume(), 1), 'pieces': len(M.decompose()), 'size_mm': np.round(ext, 1).tolist(), 'tris': len(tm.faces)}
json.dump({'parts': summary, 'joints': report}, open(os.path.join(out, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
log('done')
