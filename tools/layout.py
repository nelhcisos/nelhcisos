"""Orient every part for resin printing and pack them on one build plate.

Orientation: the hidden side of a part (its joint faces, which never show after assembly) is turned
toward the plate, so support marks land on surfaces nobody sees. Then the part is tilted to reduce
peel force per layer. The base lies flat on the plate.

Usage: python3 layout.py parts_dir figure.npz out_dir [plate_w plate_d] [tilt_deg] [lift_mm]
Output: out_dir/plate.3mf (one object per part), out_dir/plate.stl, out_dir/oriented/*.stl, layout.json
"""
import sys, os, json, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

pdir, figf, out = sys.argv[1:4]
PW, PD = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (218.0, 123.0)
TILT = float(sys.argv[6]) if len(sys.argv) > 6 else 25.0
LIFT = float(sys.argv[7]) if len(sys.argv) > 7 else 5.0
GAP = 4.0
SIDEWAYS = {'cap'}
os.makedirs(os.path.join(out, 'oriented'), exist_ok=True)

fig = np.load(figf); tree = cKDTree(fig['v'])

def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else Rot.from_rotvec(np.pi * np.array([1.0, 0, 0])).as_matrix()
    return Rot.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1, 1))).as_matrix()

items = []
for f in sorted(os.listdir(pdir)):
    if not f.endswith('.stl'): continue
    name = f[:-4]; m = trimesh.load(os.path.join(pdir, f))
    if name == 'base':
        R = np.eye(3); hidden = np.array([0, 0, -1.0])
    else:
        d, _ = tree.query(m.triangles_center)
        joint = d > 0.15                         # faces that are not on the original outer surface
        w = m.area_faces * joint
        hidden = (m.face_normals * w[:, None]).sum(0)
        if np.linalg.norm(hidden) < 1e-6: hidden = np.array([0, 0, -1.0])
        if name in SIDEWAYS:
            # cup-shaped part: open side faces sideways so it cannot act as a suction cup on the vat film
            R = rot_a_to_b(hidden, np.array([-1.0, 0, 0]))
            R = Rot.from_euler('y', -20, degrees=True).as_matrix() @ R
        else:
            R = rot_a_to_b(hidden, np.array([0, 0, -1.0]))
            R = Rot.from_euler('x', TILT, degrees=True).as_matrix() @ R
    T = np.eye(4); T[:3, :3] = R
    m.apply_transform(T)
    # turn around Z to the smallest footprint
    best = None
    for a in range(0, 180, 10):
        Rz = Rot.from_euler('z', a, degrees=True).as_matrix()
        xy = m.vertices @ Rz.T
        ext = xy.max(0) - xy.min(0)
        area = ext[0] * ext[1]
        if best is None or area < best[0]: best = (area, a)
    Tz = np.eye(4); Tz[:3, :3] = Rot.from_euler('z', best[1], degrees=True).as_matrix(); m.apply_transform(Tz)
    lift = 0.0 if name == 'base' else LIFT
    m.apply_translation([-m.bounds[0][0], -m.bounds[0][1], -m.bounds[0][2] + lift])
    items.append({'name': name, 'mesh': m, 'w': m.extents[0], 'd': m.extents[1], 'h': m.bounds[1][2],
                  'hidden_dir': np.round(hidden / np.linalg.norm(hidden), 2).tolist()})

# shelf packing, biggest first
items.sort(key=lambda it: -it['d'] * it['w'])
x = y = GAP; shelf = 0.0; placed = []
for it in items:
    if x + it['w'] + GAP > PW:
        x = GAP; y += shelf + GAP; shelf = 0.0
    it['mesh'].apply_translation([x, y, 0]); it['pos'] = [round(x, 1), round(y, 1)]
    x += it['w'] + GAP; shelf = max(shelf, it['d']); placed.append(it)
used_d = y + shelf + GAP
print(f'plate {PW}x{PD} mm, used depth {used_d:.1f} mm', 'OK' if used_d <= PD else 'DOES NOT FIT')
# centre the whole layout on the plate
allv = np.vstack([it['mesh'].vertices for it in placed])
off = [PW / 2 - (allv[:, 0].min() + allv[:, 0].max()) / 2, PD / 2 - (allv[:, 1].min() + allv[:, 1].max()) / 2, 0]
scene = trimesh.Scene()
for it in placed:
    it['mesh'].apply_translation(off)
    it['mesh'].export(os.path.join(out, 'oriented', it['name'] + '.stl'))
    scene.add_geometry(it['mesh'], node_name=it['name'], geom_name=it['name'])
scene.export(os.path.join(out, 'plate.3mf'))
trimesh.util.concatenate([it['mesh'] for it in placed]).export(os.path.join(out, 'plate.stl'))
json.dump({'plate_mm': [PW, PD], 'tilt_deg': TILT, 'lift_mm': LIFT, 'max_height_mm': round(max(it['h'] for it in placed), 1),
           'parts': {it['name']: {'footprint_mm': [round(it['w'], 1), round(it['d'], 1)], 'height_mm': round(it['h'], 1),
                                  'hidden_side_was': it['hidden_dir']} for it in placed}},
          open(os.path.join(out, 'layout.json'), 'w'), indent=1)
print('max height', round(max(it['h'] for it in placed), 1), 'mm')
