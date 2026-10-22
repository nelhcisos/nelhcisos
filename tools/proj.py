"""Orthographic colour projections of a coloured point set with a mm grid (for picking cut heights).
Usage: python3 proj.py fig.npz out.png [labels.npy]"""
import sys, numpy as np
from PIL import Image, ImageDraw
d = np.load(sys.argv[1]); V = d['v']; C = d['c'].astype(float)
if len(sys.argv) > 3:
    lab = np.load(sys.argv[3])
    pal = np.array([(0,0,0),(230,230,60),(30,120,220),(30,200,220),(170,140,200),(150,210,120),(255,255,255),(255,120,0),
        (240,60,60),(255,170,170),(0,90,160),(0,160,255),(250,200,150),(200,0,200),(255,100,255),(40,140,60),(150,170,255),(140,255,140)]+[(0,0,0)]*50)
    C = pal[lab].astype(float)
s = 10  # px per mm
def view(ax_u, sgn_u, ax_d, sgn_d):
    u = V[:, ax_u] * sgn_u; dep = V[:, ax_d] * sgn_d; z = V[:, 2]
    W = int(50 * s); Hh = int(92 * s)
    px = ((u + 25) * s).astype(int); py = ((91 - z) * s).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < Hh)
    order = np.argsort(dep[ok])  # far first, near overwrite
    img = np.full((Hh, W, 3), 255.0)
    pxo, pyo, Co = px[ok][order], py[ok][order], C[ok][order]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            img[np.clip(pyo + dy, 0, Hh - 1), np.clip(pxo + dx, 0, W - 1)] = Co
    im = Image.fromarray(img.astype(np.uint8)); dr = ImageDraw.Draw(im)
    for zz in range(0, 92, 5):
        y = int((91 - zz) * s); dr.line([(0, y), (W, y)], fill=(255, 0, 0) if zz % 10 == 0 else (255, 180, 180), width=1); dr.text((2, y - 11), str(zz), fill=(200, 0, 0))
    for xx in range(-25, 26, 5):
        x = int((xx + 25) * s); dr.line([(x, 0), (x, Hh)], fill=(0, 0, 255) if xx == 0 else (190, 190, 255), width=1)
        dr.text((x + 2, Hh - 12), str(xx), fill=(0, 0, 200))
    return im
# front: camera at -Y looking +Y -> u = x, nearer = smaller y
fr = view(0, 1, 1, -1); sd = view(1, 1, 0, 1)   # side: camera at +X looking -X: u = y
bk = view(0, -1, 1, 1)
out = Image.new('RGB', (fr.width * 3, fr.height), 'white')
out.paste(fr, (0, 0)); out.paste(sd, (fr.width, 0)); out.paste(bk, (fr.width * 2, 0)); out.save(sys.argv[2])
