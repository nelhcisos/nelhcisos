"""Remove the extra bow at the back-centre of the cap (an artefact of the generated back view) and
re-centre the figure on its body axis.

Method: look at the head from a centre point inside it. For every direction in an angular window that
covers the bow, find the first surface the ray leaves the head through (so detached bow tails are
ignored). Inside the window that radius map is replaced by a smooth membrane (Laplace fill) that matches
the cap fabric on the window edges. The solid inside the window cone is then rebuilt from that map.

Usage: python3 fix_bow.py fig.npz out.npz [dx]"""
import sys, numpy as np, manifold3d as mf
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

src, out = sys.argv[1], sys.argv[2]; dx = float(sys.argv[3]) if len(sys.argv) > 3 else 1.1
d = np.load(src); V = d['v'].copy(); F = d['f']; C = d['c']
V[:, 0] += dx
S = mf.Manifold(mf.Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))

CEN = np.array([0.5, 0.5, 66.0])
AZ = (-48.0, 48.0)      # azimuth around +Y (back), degrees; + toward +X (figure's left)
EL = (-58.0, 14.0)       # elevation, degrees
STEP = 0.4              # angular grid, degrees

# occupancy of the head region
h = 0.08; lo = np.array([-24.0, -24.0, 44.0]); n = np.ceil(np.array([48, 48, 46]) / h).astype(int)
occ = np.zeros(n, bool)
for k in range(n[2]):
    zc = lo[2] + k * h; acc = np.zeros((n[1], n[0]), bool)
    for p in S.slice(zc).to_polygons():
        im = Image.new('1', (n[0], n[1]), 0)
        ImageDraw.Draw(im).polygon([((q[0] - lo[0]) / h, (q[1] - lo[1]) / h) for q in p], fill=1)
        acc ^= np.asarray(im, bool)
    occ[:, :, k] = acc.T

az = np.radians(np.arange(AZ[0], AZ[1] + 1e-6, STEP)); el = np.radians(np.arange(EL[0], EL[1] + 1e-6, STEP))
A, E = np.meshgrid(az, el, indexing='ij')
dirs = np.stack([np.sin(A) * np.cos(E), np.cos(A) * np.cos(E), np.sin(E)], -1)   # az=0 -> +Y
ts = np.arange(0, 30, 0.04)
pts = CEN + dirs[..., None, :] * ts[:, None]
ii = np.clip(np.round((pts - lo) / h).astype(int), 0, n - 1)
inside = occ[ii[..., 0], ii[..., 1], ii[..., 2]]
inside[..., 0] = True
first_out = np.argmax(~inside, axis=-1)          # first step outside the solid
r_first = ts[first_out]
np.save(out.replace('.npz','_rfirst.npy'), r_first)

# Laplace fill inside the window, fixed values on the border (2 cells thick)
# knot area (ellipse in az/el degrees); everything else keeps the first-exit surface, which already
# drops the detached bow tails and leaves the nape hair as it was
KNOT_C, KNOT_R = np.array([2.5, -1.5]), np.array([23.0, 15.0])
ell = ((np.degrees(A) - KNOT_C[0]) / KNOT_R[0]) ** 2 + ((np.degrees(E) - KNOT_C[1]) / KNOT_R[1]) ** 2
r = r_first.copy(); fixed = ell > 1.0
r[~fixed] = r[fixed].mean()
for it in range(6000):
    rn = 0.25 * (np.roll(r, 1, 0) + np.roll(r, -1, 0) + np.roll(r, 1, 1) + np.roll(r, -1, 1))
    r = np.where(fixed, r_first, rn)
# feather: blend from the real surface at the window edge to the membrane inside (no visible step)
FEATHER = 18  # cells (~7 degrees)
dist_edge = ndi.distance_transform_edt(~fixed)
w = np.clip(dist_edge / FEATHER, 0, 1); w = w * w * (3 - 2 * w)
r = r_first + w * (r - r_first)
print('membrane radius range', r.min().round(2), r.max().round(2), 'max cut', (r_first - r).max().round(2))

from scipy.interpolate import RegularGridInterpolator
RI = RegularGridInterpolator((np.degrees(az), np.degrees(el)), r, bounds_error=False, fill_value=None)

def ellipse_star(rfun, nu=90, nv=360, scale=1.0):
    """Closed star-shaped mesh over the elliptical knot window: apex at CEN, surface at radius rfun(az,el)."""
    u = np.linspace(0, 1, nu + 1)[1:]; v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, Vv = np.meshgrid(u, v, indexing='ij')
    azd = KNOT_C[0] + KNOT_R[0] * scale * U * np.cos(Vv); eld = KNOT_C[1] + KNOT_R[1] * scale * U * np.sin(Vv)
    def pt(a_, e_):
        a_, e_ = np.radians(a_), np.radians(e_)
        dd = np.stack([np.sin(a_) * np.cos(e_), np.cos(a_) * np.cos(e_), np.sin(e_)], -1)
        return CEN + dd * rfun(np.degrees(a_), np.degrees(e_))[..., None]
    P = pt(azd, eld).reshape(-1, 3)
    c0 = pt(np.array([KNOT_C[0]]), np.array([KNOT_C[1]]))
    P = np.vstack([P, c0, CEN]); ic, iapex = len(P) - 2, len(P) - 1
    vid = np.arange(nu * nv).reshape(nu, nv)
    T = [np.c_[np.full(nv, ic), vid[0], np.roll(vid[0], -1)]]
    for i in range(nu - 1):
        a_, b_ = vid[i], np.roll(vid[i], -1); c_, d_ = np.roll(vid[i + 1], -1), vid[i + 1]
        T += [np.c_[a_, d_, c_], np.c_[a_, c_, b_]]
    T.append(np.c_[np.full(nv, iapex), np.roll(vid[-1], -1), vid[-1]])
    T = np.vstack(T)
    M = mf.Manifold(mf.Mesh(vert_properties=P.astype(np.float32), tri_verts=T.astype(np.uint32)))
    if M.status() != mf.Error.NoError or M.volume() <= 0:
        M = mf.Manifold(mf.Mesh(vert_properties=P.astype(np.float32), tri_verts=T[:, ::-1].astype(np.uint32)))
    return M

cone = ellipse_star(lambda a_, e_: np.full(np.shape(a_), 40.0))
patch = ellipse_star(lambda a_, e_: RI(np.stack([a_, e_], -1)))
print('cone', cone.status(), round(cone.volume(), 1), 'patch', patch.status(), round(patch.volume(), 1))
S2 = (S - cone) + patch
# bow tails hang free behind the nape (y > 11 mm, below z 62.5): cut them off
for x0, x1 in ((-8.5, -0.5), (1.0, 10.0)):
    S2 = S2 - mf.Manifold.cube([x1 - x0, 15.0, 9.0]).translate([x0, 11.0, 53.5])
parts = S2.decompose(); S2 = max(parts, key=lambda m: m.volume())
print('volume change mm3', round(S2.volume() - S.volume(), 1), 'pieces dropped', len(parts) - 1)
m = S2.to_mesh(); V2 = np.asarray(m.vert_properties[:, :3], float); F2 = np.asarray(m.tri_verts)
# seam is smoothed later by smooth_seam.py, after a clean voxel remesh
_, j = cKDTree(V).query(V2); C2 = C[j]
np.savez_compressed(out, v=V2, f=F2, c=C2, cen=CEN, knot_c=KNOT_C, knot_r=KNOT_R)
print('verts', len(V2), 'faces', len(F2))
