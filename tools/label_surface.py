"""Label every surface vertex with a part id (colour + position rules, then graph smoothing).
Usage: python3 label_surface.py fig.npz labels.npy"""
import sys, numpy as np, scipy.sparse as sp
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from parts import ID, Z_BASE, Z_CROC, Z_HEM, Z_GLOVE_TOP, Z_HEAD
d = np.load(sys.argv[1]); V, F, C = d['v'], d['f'], d['c'].astype(int)
x, y, z = V.T; r, g, b = C.T
mx = C.max(1); mn = C.min(1); sat = (mx - mn) / np.maximum(mx, 1)
blue = (b > r + 35) & (b > 150)
skin = (r > g + 12) & (g > b + 5) & (r > 170)
white = (mn > 195) & (sat < 0.13)
dark = mx < 110
L = np.zeros(len(V), np.int16); U = 0  # 0 = undecided, filled by smoothing
side = lambda R, Lf: np.where(x < 0, ID[R], ID[Lf])
L[z < Z_BASE] = ID['base']
m = (z >= Z_BASE) & (z < Z_CROC); L[m] = side('croc_R', 'croc_L')[m]
m = (z >= Z_CROC) & (z < Z_HEM); L[m] = ID['pants']
m = (z >= 28) & (z < Z_GLOVE_TOP) & (np.abs(x) > 10.3); L[m] = side('glove_R', 'glove_L')[m]
body = (z >= Z_HEM) & (z < Z_HEAD) & ~((z < Z_GLOVE_TOP) & (np.abs(x) > 10.3))
L[body] = ID['shirt']
m = body & skin & (z < 48) & (np.abs(x) > 8); L[m] = side('arm_R', 'arm_L')[m]
m = body & (white | skin) & (z > 46) & (z < 54.3) & (np.abs(x) < 5.5) & (y < 0); L[m] = ID['tshirt']
m = body & skin & (z >= 54.3) & (np.abs(x) < 7); L[m] = ID['neck']
m = body & skin & (z > 50) & (y >= 0) & (np.abs(x) < 7); L[m] = ID['neck']
hd = z >= Z_HEAD
L[hd] = U
m = hd & blue & ((z > 69) | (np.abs(x) > 12.5) | (y > 3)); L[m] = ID['cap']
m = hd & skin & (np.abs(x) <= 11.3); L[m] = ID['head']
m = hd & skin & (np.abs(x) > 11.3) & (z < 68); L[m] = side('ear_R', 'ear_L')[m]
m = hd & white & (y < -3) & (z < 64.5) & (np.abs(x) < 11.5); L[m] = ID['mask']
m = hd & white & ~((y < -3) & (z < 64.5) & (np.abs(x) < 11.5)); L[m] = ID['hair']
m = hd & dark; L[m] = ID['head']
# --- graph smoothing: majority vote among neighbours, a few rounds (fills undecided + removes speckle)
n = len(V); e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
A = A + sp.identity(n, format='csr')
fixed = ~hd  # body labels come from position, keep them
K = 18
for it in range(12):
    onehot = sp.csr_matrix((np.ones(n), (np.arange(n), L)), shape=(n, K))
    votes = (A @ onehot).toarray()
    votes[:, 0] = 0  # never vote for undecided
    newL = votes.argmax(1).astype(np.int16)
    newL[votes.max(1) == 0] = 0
    L = np.where(fixed, L, newL)
print('undecided left', int((L == 0).sum()))
np.save(sys.argv[2], L)
for k in range(1, K):
    c = int((L == k).sum())
    if c: print(k, c)
