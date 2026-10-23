"""Hollow big resin parts with an even wall and add drain holes on hidden sides.

The cavity is the part shrunk by `wall` from its whole surface (socket walls included), so every
socket and key keeps at least `wall` of material around it. Drain holes are short cylinders drilled
from a hidden side straight into the cavity; they never reach the opposite (visible) wall.

Usage: python3 hollow.py in.stl out.stl --wall 1.6 --holes "x,y,top;x,y,top" [--d 2.5] [--voxel 0.08]
  each hole: x,y in mm and the side it is drilled from: top (drill down) or bottom (drill up).
"""
import argparse, json, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from skimage import measure
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--wall', type=float, default=1.6); ap.add_argument('--d', type=float, default=2.5)
ap.add_argument('--voxel', type=float, default=0.08); ap.add_argument('--holes', default='')
ap.add_argument('--min-cavity', type=float, default=20.0)
a = ap.parse_args(); H = a.voxel

m = trimesh.load(a.inp)
P = mf.Manifold(mf.Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
assert P.status() == mf.Error.NoError
lo = m.bounds[0] - 4 * H; shape = np.ceil((m.bounds[1] - m.bounds[0]) / H).astype(int) + 9
occ = np.zeros(shape, bool)
for k in range(shape[2]):
    polys = P.slice(lo[2] + k * H).to_polygons()
    if not polys: continue
    acc = np.zeros((shape[1], shape[0]), bool)
    for p in polys:
        im = Image.new('1', (shape[0], shape[1]), 0)
        ImageDraw.Draw(im).polygon([((q[0] - lo[0]) / H, (q[1] - lo[1]) / H) for q in p], fill=1); acc ^= np.asarray(im, bool)
    occ[:, :, k] = acc.T
cav = ndi.distance_transform_edt(occ) * H > a.wall
lab, n = ndi.label(cav)
if n:
    sz = ndi.sum(cav, lab, range(1, n + 1)) * H ** 3
    cav = np.isin(lab, [i + 1 for i, s in enumerate(sz) if s >= a.min_cavity])
f = ndi.gaussian_filter(np.pad(cav, 3).astype(np.float32), 0.7)
rep = {'volume_before': round(P.volume(), 1), 'wall_mm': a.wall}
if f.max() < 0.5:
    rep['note'] = 'too thin to hollow'; out = P
else:
    v, fc, _, _ = measure.marching_cubes(f, 0.5)
    C = mf.Manifold(mf.Mesh(vert_properties=((v - 3) * H + lo).astype(np.float32), tri_verts=fc[:, ::-1].astype(np.uint32)))
    out = P - C
    holes = []
    for spec in [s for s in a.holes.split(';') if s.strip()]:
        x, y, side = spec.split(',')
        x, y = float(x), float(y)
        i, j = int(round((x - lo[0]) / H)), int(round((y - lo[1]) / H))
        # nearest column that actually crosses the cavity
        best = None
        for r in range(0, int(4 / H)):
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < shape[0] and 0 <= jj < shape[1] and cav[ii, jj].any():
                        best = (ii, jj); break
                if best: break
            if best: break
        if not best: holes.append(f'{spec}: no cavity nearby'); continue
        ii, jj = best; zs = np.where(cav[ii, jj])[0]
        hx, hy = lo[0] + ii * H, lo[1] + jj * H
        col = np.where(occ[ii, jj])[0]
        if side.strip() == 'top':
            z_in = lo[2] + zs.max() * H - 0.3; z_out = lo[2] + col.max() * H + 1.0
        else:
            z_in = lo[2] + zs.min() * H + 0.3; z_out = lo[2] + col.min() * H - 1.0
        z0, z1 = min(z_in, z_out), max(z_in, z_out)
        cyl = mf.Manifold.cylinder(z1 - z0, a.d / 2, a.d / 2, 32).translate([hx, hy, z0])
        out = out - cyl
        holes.append(f'{side.strip()} hole d{a.d} at ({hx:.1f}, {hy:.1f}), z {z0:.1f}..{z1:.1f}')
    rep['holes'] = holes
pcs = sorted(out.decompose(), key=lambda q: -q.volume()); out = pcs[0]
rep['volume_after'] = round(out.volume(), 1); rep['saved_pct'] = round(100 * (1 - out.volume() / P.volume()), 1)
q = out.to_mesh(); trimesh.Trimesh(q.vert_properties[:, :3], q.tri_verts).export(a.out)
print(json.dumps(rep, ensure_ascii=False))
