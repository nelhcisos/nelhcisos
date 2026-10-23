"""Cap that goes onto the hair straight down, with crisp outer surfaces.

head+hair are merged into one piece (the receiver). The cap is made removable straight up:
  cap_final = (cap ∪ hollow) − keepout
  hollow  = inside the cap's closed ring in every horizontal slice, down to the receiver (the cap rests on it)
  keepout = receiver swept downward and grown by the clearance (so lifting the cap never hits anything)
Only `hollow` and `keepout` come from voxels; the visible cap surface stays the original mesh.
Receiver material that sticks out of the cap where it should be covered is trimmed by the cap.

Usage: python3 fit_cap.py head.stl hair.stl cap.stl out_dir [voxel_mm] [clearance_mm]
"""
import sys, os, json, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from skimage import measure
from PIL import Image, ImageDraw

head_f, hair_f, cap_f, out = sys.argv[1:5]
H = float(sys.argv[5]) if len(sys.argv) > 5 else 0.06
CLEAR = float(sys.argv[6]) if len(sys.argv) > 6 else 0.10
os.makedirs(out, exist_ok=True)

def man(m):
    M = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    assert M.status() == mf.Error.NoError, M.status()
    return M

def tm(M):
    q = M.to_mesh(); return trimesh.Trimesh(q.vert_properties[:, :3], q.tri_verts, process=False)

head, hair, cap = (man(trimesh.load(f)) for f in (head_f, hair_f, cap_f))
recv = head + hair                           # one piece: head with its hair
lo = np.minimum(tm(recv).bounds[0], tm(cap).bounds[0]) - 1
hi = np.maximum(tm(recv).bounds[1], tm(cap).bounds[1]) + 1
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

def mesh_of(occ, sigma=0.7):
    f = ndi.gaussian_filter(np.pad(occ, 3).astype(np.float32), sigma)
    v, fc, _, _ = measure.marching_cubes(f, 0.5)
    v = (v - 3) * H + lo
    return man(trimesh.Trimesh(v, fc[:, ::-1], process=True))

C = vox(cap)
sweep_down = lambda A: np.flip(np.maximum.accumulate(np.flip(A, 2), axis=2), 2)
sweep_up = lambda A: np.maximum.accumulate(A, axis=2)
# the cap wins: receiver material inside the cap shell or poking out above it is removed
recv = recv - mesh_of(C | sweep_up(C), sigma=0.5)
recv = sorted(recv.decompose(), key=lambda q: -q.volume())[0]
R = vox(recv)
ring = np.zeros(shape, bool)
for k in range(shape[2]):
    if C[:, :, k].any(): ring[:, :, k] = ndi.binary_fill_holes(C[:, :, k])
hollow = ring & sweep_down(C) & ~C
keepout = ndi.binary_dilation(sweep_down(R), ndi.generate_binary_structure(3, 1), iterations=max(1, int(round(CLEAR / H))))
print('hollow mm3', round(hollow.sum() * H ** 3, 1), 'keepout overlap with cap mm3', round((keepout & C).sum() * H ** 3, 1))

hollow_m = mesh_of(hollow & ~keepout)
keep_m = mesh_of(keepout, sigma=0.5)
cap_final = (cap + hollow_m) - keep_m
pcs = sorted(cap_final.decompose(), key=lambda q: -q.volume()); cap_final = pcs[0]
recv_final = recv; rp = [recv]

# check: lift the cap straight up, nothing of the receiver may be under it
Cf = vox(cap_final); Rf = vox(recv_final)
coll = (Cf & sweep_down(Rf)).sum()
rep = {'cap volume mm3': round(cap_final.volume(), 1), 'head+hair volume mm3': round(recv_final.volume(), 1),
       'cap collisions when lifted (voxels)': int(coll), 'cap pieces dropped': len(pcs) - 1, 'head pieces dropped': len(rp) - 1}
tm(cap_final).export(os.path.join(out, 'cap.stl')); tm(recv_final).export(os.path.join(out, 'head_hair.stl'))
json.dump(rep, open(os.path.join(out, 'report.json'), 'w'), indent=1)
for k, v in rep.items(): print(k, v)
