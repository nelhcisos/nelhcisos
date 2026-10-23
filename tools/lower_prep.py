"""Prepare the lower body: crocs cut flat on the base, pants legs cleaned, socks placed in the crocs.
Usage: python3 lower_prep.py croc_R.stl croc_L.stl pants.stl sock.glb out_dir [base_top_mm]"""
import sys, os, numpy as np, trimesh, manifold3d as mf
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
cr, cl, pf, sf, out = sys.argv[1:6]; ZB = float(sys.argv[6]) if len(sys.argv) > 6 else 6.85
os.makedirs(out, exist_ok=True)
man = lambda m: mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
def tm(M): q = M.to_mesh(); return trimesh.Trimesh(q.vert_properties[:, :3], q.tri_verts, process=False)
big = lambda M: max(M.decompose(), key=lambda q: q.volume())
above = lambda z: mf.Manifold.cube([200, 200, 200]).translate([-100, -100, z])

# 1. crocs: drop the generated peg, flat sole on the base top
crocs = {}
for name, f in (('croc_R', cr), ('croc_L', cl)):
    crocs[name] = big(man(trimesh.load(f)) ^ above(ZB))
    tm(crocs[name]).export(os.path.join(out, name + '.stl'))
    print(name, 'vol', round(crocs[name].volume(), 1), 'z', tm(crocs[name]).bounds[:, 2].round(2))

# 2. pants: below 13.5 mm keep only the leg column (the leg section at 14.5 mm extruded down)
P = man(trimesh.load(pf))
legs = P.slice(14.5)
col = mf.Manifold.extrude(legs.offset(0.2, mf.JoinType.Round), 14.0).translate([0, 0, 0.0])
pants = (P ^ above(13.5)) + (P ^ col)
pants = big(pants)
tm(pants).export(os.path.join(out, 'pants.stl'))
print('pants vol', round(P.volume(), 1), '->', round(pants.volume(), 1))

# 3. socks: sole on the croc footbed, sized to the foot opening
s = trimesh.load(sf); g = s.dump(concatenate=True) if isinstance(s, trimesh.Scene) else s
sock0 = trimesh.Trimesh(g.vertices[:, [0, 2, 1]] * [1, -1, 1], g.faces, process=True)
# the generated sock points its toe along +X: turn the toe to -Y (front)
sock0.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1]))
SC_FIX = float(os.environ.get('SOCK_SCALE', '0'))
for name, C in crocs.items():
    cm = tm(C); zt = cm.bounds[1][2]
    # foot opening: holes of the croc section a little below the rim
    best = None
    for z in np.arange(zt - 1.0, ZB + 0.5, -0.25):
        polys = C.slice(z).to_polygons()
        if not polys: continue
        cs = mf.CrossSection(polys)
        # holes = filled outline minus the croc itself
        filled = mf.CrossSection([max(polys, key=lambda p: abs(mf.CrossSection([p]).area()))])
        hole = filled - cs
        if hole.area() > 5: best = (z, hole)
    zfoot, hole = best
    hp = np.vstack(hole.to_polygons())
    hx, hy = hp[:, 0], hp[:, 1]
    print(name, 'footbed z', round(zfoot, 2), 'opening x', hx.min().round(2), hx.max().round(2), 'y', hy.min().round(2), hy.max().round(2))
    sk = sock0.copy()
    if name == 'croc_L': sk.vertices[:, 0] *= -1; sk.invert()
    b = sk.bounds; L = b[1][1] - b[0][1]; W = b[1][0] - b[0][0]
    sc = min((hy.max() - hy.min()) * 0.97 / L, (hx.max() - hx.min()) * 0.97 / W)
    if SC_FIX: sc = SC_FIX
    sk.apply_scale(sc)
    b = sk.bounds
    sk.apply_translation([(hx.min() + hx.max()) / 2 - (b[0][0] + b[1][0]) / 2, (hy.min() + hy.max()) / 2 - (b[0][1] + b[1][1]) / 2, zfoot - b[0][2]])
    S = big(man(sk)) if man(sk).status() == mf.Error.NoError else None
    sk.export(os.path.join(out, name.replace('croc', 'sock') + '_raw.stl'))
    print(name.replace('croc', 'sock'), 'scale', round(sc, 3), 'size', sk.extents.round(2), 'closed', S is not None)
