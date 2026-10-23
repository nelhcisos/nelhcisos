"""Assemble kit v2: regenerated head stack + crocs/socks/pants, body parts from v1, all keys and sockets.

Usage: python3 assemble_v2.py cfg.json out_dir
cfg keys: fig (master npz), v1 (v1 parts dir), head, cap, mask, ear_R, ear_L, croc_R, croc_L, sock_R, sock_L, pants
"""
import sys, os, json, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from skimage import measure
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as Rot

cfg = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
CLEAR, DEPTH_CLEAR, H = 0.10, 0.25, 0.05
man = lambda m: mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
def tm(M): q = M.to_mesh(); return trimesh.Trimesh(q.vert_properties[:, :3], q.tri_verts, process=False)
load = lambda f: man(trimesh.load(f))
def big(M, frac=0.02):
    pcs = sorted(M.decompose(), key=lambda q: -q.volume()); keep = [q for q in pcs if q.volume() > frac * pcs[0].volume()]
    R = keep[0]
    for q in keep[1:]: R = R + q
    return R

def grown(M, d):
    """M grown outward by d mm (voxel based, only used for sockets/clearances)"""
    b = tm(M).bounds; lo = b[0] - d - 4 * H; shape = np.ceil((b[1] - b[0] + 2 * d + 8 * H) / H).astype(int) + 1
    occ = np.zeros(shape, bool)
    for k in range(shape[2]):
        polys = M.slice(lo[2] + k * H).to_polygons()
        if not polys: continue
        acc = np.zeros((shape[1], shape[0]), bool)
        for p in polys:
            im = Image.new('1', (shape[0], shape[1]), 0)
            ImageDraw.Draw(im).polygon([((q[0] - lo[0]) / H, (q[1] - lo[1]) / H) for q in p], fill=1); acc ^= np.asarray(im, bool)
        occ[:, :, k] = acc.T
    dist = ndi.distance_transform_edt(~occ) * H
    f = np.pad((d - dist).astype(np.float32), 2, constant_values=-1)
    v, fc, _, _ = measure.marching_cubes(f, 0.0)
    return man(trimesh.Trimesh((v - 2) * H + lo, fc[:, ::-1], process=True))

def key(p, n, size, z0, z1, sides=4, taper=0.8):
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = ref - n * (ref @ n); u /= np.linalg.norm(u); w = np.cross(n, u)
    rad = size / 2 / np.cos(np.pi / sides); ang = np.pi / sides
    poly = [[rad * np.cos(ang + 2 * np.pi * i / sides), rad * np.sin(ang + 2 * np.pi * i / sides)] for i in range(sides)]
    body = mf.Manifold.extrude(mf.CrossSection([poly]), z1 - z0, 0, 0.0, [taper, taper]).translate([0, 0, z0])
    return body.transform(np.hstack([np.column_stack([u, w, n]), np.asarray(p, float).reshape(3, 1)]).tolist())

P = {}; log = []
v1 = cfg['v1']
for k in ('shirt', 'tshirt', 'neck', 'arm_R', 'arm_L', 'glove_R', 'glove_L'):
    P[k] = load(os.path.join(v1, k + '.stl'))
for k in ('cap', 'mask', 'croc_R', 'croc_L', 'sock_R', 'sock_L', 'pants'):
    P[k] = load(cfg[k])
fig = np.load(cfg['fig']); SOL = man(trimesh.Trimesh(fig['v'], fig['f'], process=False))
P['base'] = SOL ^ mf.Manifold.cube([100, 100, 6.8]).translate([-50, -50, 0])

# head: sockets for the neck (with its hex key) and nothing of the head inside the shirt
head = load(cfg['head'])
head = big(head - grown(P['neck'], CLEAR) - P['shirt'])
# ears: ear wins over head, socket by the ear's own shape + clearance
for e in ('ear_R', 'ear_L'):
    E = load(cfg[e]); E = big(E - P['cap'] - P['mask'])
    P[e] = E
    head = big(head - grown(E, CLEAR))
P['head'] = head

def add_key(male, female, p, n, size, depth, root, sides=4):
    P[male] = P[male] + key(p, n, size, -root, depth, sides)
    P[female] = P[female] - key(p, n, size + 2 * CLEAR, -root, depth + DEPTH_CLEAR, sides)
    log.append(f'{male}->{female}: {"square" if sides == 4 else "hex"} key {size} mm, depth {depth} mm at {np.round(p, 1).tolist()}')

# crocs -> base: key straight down from the middle of the sole
for c in ('croc_R', 'croc_L'):
    sec = P[c].slice(6.95).to_polygons(); pts = np.vstack(sec)
    ctr = pts.mean(0)
    add_key(c, 'base', [ctr[0], ctr[1], 6.8], [0, 0, -1], 2.2, 2.0, 1.2)
# socks -> pants: key straight up from the top of the sock
for s in ('sock_R', 'sock_L'):
    b = tm(P[s]).bounds; ztop = b[1][2]
    sec = P[s].slice(ztop - 0.3).to_polygons(); pts = np.vstack(sec); ctr = pts.mean(0)
    add_key(s, 'pants', [ctr[0], ctr[1], ztop], [0, 0, 1], 2.0, 2.5, 1.0)
# ears -> head: key from the ear's contact face toward the head centre
hc = tm(P['head']).bounds.mean(0)
for e in ('ear_R', 'ear_L'):
    em = tm(P[e]); c = em.bounds.mean(0); n = hc - c; n[2] = 0; n /= np.linalg.norm(n)
    # contact point: ear point nearest to the head along n
    pts = em.vertices; p = pts[np.argmax(pts @ n)]
    add_key(e, 'head', p - n * 0.1, n, 1.4, 1.5, 0.6)

rep = {}
for k, M in P.items():
    M = big(M); P[k] = M
    m = tm(M); m.export(os.path.join(out, k + '.stl'))
    rep[k] = {'volume_mm3': round(M.volume(), 1), 'size_mm': np.round(m.extents, 1).tolist(), 'closed': bool(m.is_watertight), 'pieces': len(M.decompose())}
json.dump({'parts': rep, 'keys': log}, open(os.path.join(out, 'summary.json'), 'w'), indent=1, ensure_ascii=False)
for k, v in rep.items(): print(k, v)
for l in log: print(l)
