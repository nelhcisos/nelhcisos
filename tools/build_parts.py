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

CACHE = os.path.join(out, 'voxels.npz')
if os.path.exists(CACHE) and '--rebuild' not in sys.argv:
    cz = np.load(CACHE); G, occ, d_in, lo, shape = cz['G'], cz['occ'], cz['d_in'].astype(np.float64), cz['lo'], cz['shape']
    log('loaded voxel cache')
else:
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
    log('distance fields done')
    work = occ | (d_out <= BAND)
    del d_out

    # ---------------------------------------------------------------- 2. nearest surface label (grid EDT)
    good = SL > 0
    if (~good).any():
        _, j = cKDTree(V[good]).query(V[~good]); SL[~good] = SL[good][j]
    seed = np.zeros(shape, np.int16)
    vi = np.clip(np.round((V - lo) / H).astype(int), 0, shape - 1)
    seed[vi[:, 0], vi[:, 1], vi[:, 2]] = SL
    nearest = ndi.distance_transform_edt(seed == 0, return_distances=False, return_indices=True).astype(np.int32)
    L0g = seed[nearest[0], nearest[1], nearest[2]]
    del nearest, seed
    log('nearest-label done')

    # ---------------------------------------------------------------- 3. rules
    c = lambda n: ID[n]
    cap, hair, mask, head = c('cap'), c('hair'), c('mask'), c('head')
    zz = lo[2] + np.arange(shape[2]) * H
    head_zone_g = np.broadcast_to(zz >= Z_HEAD - 0.8, shape)
    # smooth skull core: head-zone solid, shrunk 2.4 mm and smoothed -> hair/cap/mask/ears sit on it
    hz = occ & head_zone_g
    skull = ndi.gaussian_filter((ndi.distance_transform_edt(hz) * H > 2.4).astype(np.float32), 1.2 / H * 0.8) > 0.5
    d_skull = ndi.distance_transform_edt(~skull) * H
    log('skull core', round(skull.sum() * H ** 3, 1), 'mm3')

    idx = np.argwhere(work)
    P = lo + idx * H
    x, y, z = P.T
    t = tuple(idx.T)
    L0 = L0g[t]; inside = occ[t]; depth = np.where(inside, d_in[t], 0.0); in_skull = skull[t]; dsk = d_skull[t]
    del L0g
    L = L0.copy()
    head_zone = z >= Z_HEAD - 0.8
    # inside the skull everything is head
    L[head_zone & in_skull & inside] = head
    out_sk = head_zone & ~in_skull
    # cap: 1.4 mm shell; deeper cap material close to the skull becomes a hair "helmet" under the cap,
    # deeper material far from the skull (knot) stays cap
    m = out_sk & (L0 == cap) & (depth >= 1.4) & (dsk < 2.0); L[m] = hair
    m = out_sk & (L0 == mask) & (depth >= 1.2); L[m] = head
    # body: flat cuts, limbs keep their nearest-surface volume
    body_set = np.isin(L0, [c('base'), c('croc_R'), c('croc_L'), c('pants'), c('shirt')]) | (~head_zone & np.isin(L0, [head, hair, mask, cap]))
    plane = np.where(z < Z_BASE, c('base'), np.where(z < Z_CROC, np.where(x < 0, c('croc_R'), c('croc_L')),
                     np.where(z < Z_HEM, c('pants'), c('shirt'))))
    L[body_set] = plane[body_set]
    # arms / gloves must not grow into the torso: keep them outside |x| = 7 mm
    limb = np.isin(L, [c('arm_R'), c('arm_L'), c('glove_R'), c('glove_L')]) & (np.abs(x) < 7.0)
    L[limb] = plane[limb]
    m = (L0 == c('tshirt')) & (depth >= 1.5); L[m] = c('shirt')
    # neck cylinder: from inside the shirt up into the head
    neck_xy = np.array([0.0, -1.0])
    sl55 = SOLID.slice(55.0).to_polygons()
    if sl55:
        big = max(sl55, key=lambda p: len(p)); neck_xy = np.asarray(big).mean(0)
    NECK_R, NECK_Z0, NECK_Z1 = 2.8, 51.5, 57.5
    m = (np.hypot(x - neck_xy[0], y - neck_xy[1]) < NECK_R) & (z > NECK_Z0) & (z < NECK_Z1) & np.isin(L, [c('shirt'), head, c('neck'), c('tshirt'), hair, mask]) & inside
    L[m] = c('neck')
    m = (L == c('neck')) & ((z < NECK_Z0) | (z > NECK_Z1)); L[m] = np.where(z[m] < 55, c('shirt'), head)
    log('neck axis', neck_xy.round(2))

    G = np.zeros(shape, np.int16)
    G[t] = L
    del L0, P, x, y, z, L, d_skull, skull

    # smooth the head/hair seam: thin skin slivers between fringe spikes go to hair, thin hair dents go to head
    hd = (G == head)
    sm = ndi.gaussian_filter(hd.astype(np.float32), 0.6 / H) > 0.5
    sm &= occ & head_zone_g
    lost = hd & ~sm
    gain = sm & ~hd & (G == hair)
    G[gain] = head
    if lost.any():
        tmp = G.copy(); tmp[lost] = 0; tmp[tmp == head] = 0
        near = ndi.distance_transform_edt(tmp == 0, return_distances=False, return_indices=True)
        G[lost] = tmp[near[0][lost], near[1][lost], near[2][lost]]
        del tmp, near
    log('head seam smoothed: lost', int(lost.sum()), 'gained', int(gain.sum()), 'voxels')

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

    np.savez(CACHE, G=G, occ=occ, d_in=d_in.astype(np.float32), lo=lo, shape=shape)
    log('saved voxel cache')

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

parts = {}; REG = {}
for k, name in PARTS.items():
    if not (G == k).any(): log('MISSING', name); continue
    R = region_mesh(k)
    if R.status() != mf.Error.NoError: log('region error', name, R.status())
    REG[name] = R
    pm = SOLID ^ R
    pcs = sorted(pm.decompose(), key=lambda q: -q.volume())
    keep = [q for q in pcs if q.volume() > max(0.3, 0.02 * pcs[0].volume())]
    pm = keep[0]
    for q in keep[1:]: pm = pm + q
    parts[name] = pm
    log(name, 'vol', round(pm.volume(), 1), 'pieces kept', len(keep), 'slivers dropped', len(pcs) - len(keep))

# ---------------------------------------------------------------- 5. pegs
idx_of = lambda p: np.clip(np.round((np.asarray(p) - lo) / H).astype(int), 0, shape - 1)
lab_at = lambda p: G[tuple(idx_of(p).T)]

def frame(n):
    """Rotation matrix that maps +Z onto n, keeping the key's sides square to the world axes."""
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = ref - n * (ref @ n); u /= np.linalg.norm(u); w = np.cross(n, u)
    return np.column_stack([u, w, n])

def key(p, n, size, z0, z1, sides=4, taper=0.8):
    """Tapered prism key (square by default, like the example kit) from z0 to z1 along axis n.
    `size` = width across flats at z0; tip is `taper` times smaller so it self-centres."""
    rad = size / 2 / np.cos(np.pi / sides)
    ang = np.pi / sides
    poly = [[rad * np.cos(ang + 2 * np.pi * i / sides), rad * np.sin(ang + 2 * np.pi * i / sides)] for i in range(sides)]
    cs = mf.CrossSection([poly])
    body = mf.Manifold.extrude(cs, z1 - z0, 0, 0.0, [taper, taper]).translate([0, 0, z0])
    M = np.hstack([frame(np.asarray(n, float)), np.asarray(p, float).reshape(3, 1)])
    return body.transform(M.tolist())

WALL = 0.4   # minimum wall around a socket, mm

def ring_ok(p, n, r_out, t0, t1, want, nt=10, na=12, near_ok=None, near_t=0.0):
    u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(n, u)
    for t in np.linspace(t0, t1, nt):
        for ang in np.linspace(0, 2 * np.pi, na, endpoint=False):
            q = p + n * t + (u * np.cos(ang) + w * np.sin(ang)) * r_out
            ii = tuple(idx_of(q))
            if not occ[ii]: return False
            if G[ii] != want and not (near_ok is not None and abs(t) < near_t and G[ii] == near_ok): return False
    return True

def peg(male, female, dia, length, anchor=None, axis=None, at=None, sides=4):
    """Tapered key on `male` going `length` mm into `female`; socket in female gets clearance."""
    km, kf = ID[male], ID[female]
    a = (G == km); b = (G == kf)
    iface = a & ndi.binary_dilation(b, iterations=1) & occ
    pts = np.argwhere(iface)
    if len(pts) == 0: return f'{male}->{female}: no contact'
    P = lo + pts * H
    if at is not None:
        order = np.argsort(np.linalg.norm(P - np.asarray(at, float), axis=1))
    else:
        cen = P.mean(0)
        score = d_in[tuple(pts.T)] - 0.25 * np.linalg.norm(P - cen, axis=1)
        order = np.argsort(-score)
    centre_i = int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))
    cand = [centre_i] + list(order[:600:15])
    anchor = anchor or length
    for scale in (1.0, 0.85, 0.7, 0.55):
        r = dia * scale / 2; rc = r / np.cos(np.pi / sides); Lf = length * max(scale, 0.6)
        for ci in cand:
            sel = pts[ci]; p = P[ci]
            if axis is None:
                lo_i = np.maximum(sel - 25, 0)
                win = (slice(lo_i[0], sel[0] + 26), slice(lo_i[1], sel[1] + 26), slice(lo_i[2], sel[2] + 26))
                nb, na_ = np.argwhere(b[win]) + lo_i, np.argwhere(a[win]) + lo_i
                if len(nb) == 0 or len(na_) == 0: continue
                n = (lo + nb * H).mean(0) - (lo + na_ * H).mean(0)
            else:
                n = np.asarray(axis, float)
            if np.linalg.norm(n) < 1e-6: continue
            n = n / np.linalg.norm(n)
            if not ring_ok(p, n, rc + PEG_CLEAR + WALL, 0.15, Lf + PEG_DEPTH_CLEAR, kf, near_ok=km, near_t=0.9): continue
            # root inside the male part: as long as it fits, at least 0.3 mm
            La = 0.0
            for tt in np.arange(0.3, anchor + 1e-6, 0.3):
                if ring_ok(p, n, rc + 0.1, -tt, -0.05, km, nt=4, near_ok=kf, near_t=0.6): La = tt
                else: break
            if La < 0.3: continue
            pin = key(p, n, 2 * r, -La, Lf, sides, taper=0.8)
            hole = key(p, n, 2 * r + 2 * PEG_CLEAR, -La, Lf + PEG_DEPTH_CLEAR, sides, taper=0.8)  # same taper -> even gap
            parts[male] = parts[male] + (pin ^ SOLID)
            parts[female] = parts[female] - hole
            kind = 'square' if sides == 4 else f'{sides}-sided'
            return f'{male}->{female}: {kind} key {2*r:.2f} mm, depth {Lf:.1f} mm, root {La:.1f} mm at {np.round(p,1).tolist()}'
    return f'{male}->{female}: no room for a key (glue joint, parts locate by shape)'

JOINTS = [   # (male, female, width mm, depth mm[, sides])
    ('croc_R', 'base', 2.2, 2.0), ('croc_L', 'base', 2.2, 2.0),
    ('pants', 'croc_R', 2.6, 2.2), ('pants', 'croc_L', 2.6, 2.2),
    ('pants', 'shirt', 7.0, 5.0, 6),          # big faceted plug, like the torso in the example
    ('neck', 'shirt', 3.0, 2.5, 6), ('neck', 'head', 3.0, 3.0, 6),
    ('tshirt', 'shirt', 1.0, 1.0),
    ('arm_R', 'shirt', 1.4, 2.0), ('arm_L', 'shirt', 1.4, 2.0),
    ('glove_R', 'arm_R', 1.2, 1.5), ('glove_L', 'arm_L', 1.2, 1.5),
    ('glove_R', 'shirt', 1.4, 1.5), ('glove_L', 'shirt', 1.4, 1.5),   # hands rest on the hips
    ('ear_R', 'head', 1.4, 1.5), ('ear_L', 'head', 1.4, 1.5),
    ('hair', 'head', 2.0, 2.0), ('cap', 'hair', 1.6, 1.0), ('cap', 'head', 2.0, 2.0),
    ('mask', 'head', 1.2, 1.2),
]
report = []
for jt in JOINTS:
    male, female, dia, ln = jt[:4]; sides = jt[4] if len(jt) > 4 else 4
    if male in parts and female in parts:
        msg = peg(male, female, dia, ln, sides=sides); report.append(msg); log(msg)

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
