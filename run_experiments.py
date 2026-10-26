"""GLP-GNN vs baselines. Requires: torch, torch_geometric.
Usage: python run_experiments.py --datasets Cora Texas --models gcn glp --splits 10
Prints mean±std test accuracy per (dataset, model); paste into paper.tex.
"""
import argparse, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree

def load(name):
    root = "data"
    if name in ("Cora", "CiteSeer", "PubMed"): return Planetoid(root, name)[0], Planetoid(root, name).num_classes
    if name in ("Texas", "Cornell", "Wisconsin"): ds = WebKB(root, name)
    else: ds = WikipediaNetwork(root, name.lower(), geom_gcn_preprocess=True)
    return ds[0], ds.num_classes

def random_masks(n, seed):  # 48/32/20 random split
    g = torch.Generator().manual_seed(seed); p = torch.randperm(n, generator=g)
    a, b = int(.48 * n), int(.80 * n); m = lambda i: torch.zeros(n, dtype=torch.bool).index_fill_(0, i, True)
    return m(p[:a]), m(p[a:b]), m(p[b:])

def norm_adj(ei, n):
    ei, _ = add_self_loops(ei, num_nodes=n); d = degree(ei[0], n)
    w = d[ei[0]].pow(-.5) * d[ei[1]].pow(-.5)
    return torch.sparse_coo_tensor(ei, w, (n, n)).coalesce()

class GLPLayer(nn.Module):
    def __init__(s, i, o, gate=True, low=True, high=True, ego=True):
        super().__init__(); s.WL, s.WH, s.WS = nn.Linear(i, o), nn.Linear(i, o), nn.Linear(i, o, bias=False)
        s.gate_lin = nn.Linear(2 * i, 1); s.flags = (gate, low, high, ego)
    def forward(s, h, A):
        gate, low, high, ego = s.flags; Ah = torch.sparse.mm(A, h)
        g = torch.sigmoid(s.gate_lin(torch.cat([h, Ah], 1))) if gate else torch.full((h.size(0), 1), .5, device=h.device)
        if not low: g = torch.zeros_like(g)
        if not high: g = torch.ones_like(g)
        out = g * s.WL(Ah) + (1 - g) * s.WH(h - Ah)
        return out + s.WS(h) if ego else out

class Net(nn.Module):
    def __init__(s, kind, i, hid, o, layers=2, p=.5, **kw):
        super().__init__(); s.kind, s.p = kind, p; dims = [i] + [hid] * (layers - 1) + [o]
        mk = {"gcn": lambda a, b: GCNConv(a, b), "gat": lambda a, b: GATConv(a, b, heads=1),
              "sage": lambda a, b: SAGEConv(a, b), "glp": lambda a, b: GLPLayer(a, b, **kw)}.get(kind)
        s.mlp = kind == "mlp"
        s.layers = nn.ModuleList([nn.Linear(a, b) if s.mlp else mk(a, b) for a, b in zip(dims, dims[1:])])
    def forward(s, x, ei, A):
        for k, l in enumerate(s.layers):
            x = F.dropout(x, s.p, s.training)
            x = l(x) if s.mlp else (l(x, A) if s.kind == "glp" else l(x, ei))
            if k < len(s.layers) - 1: x = F.relu(x)
        return x

def run(data, C, kind, seed, grid, **kw):
    n = data.num_nodes; tr, va, te = random_masks(n, seed); A = norm_adj(data.edge_index, n); best = (-1, 0)
    for hid, lr, wd, p in grid:
        torch.manual_seed(seed); m = Net(kind, data.num_features, hid, C, p=p, **kw); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        bv, bt, wait = 0, 0, 0
        for ep in range(500):
            m.train(); opt.zero_grad(); F.cross_entropy(m(data.x, data.edge_index, A)[tr], data.y[tr]).backward(); opt.step()
            m.eval()
            with torch.no_grad(): pred = m(data.x, data.edge_index, A).argmax(1)
            v, t = (pred[va] == data.y[va]).float().mean().item(), (pred[te] == data.y[te]).float().mean().item()
            if v > bv: bv, bt, wait = v, t, 0
            else:
                wait += 1
                if wait >= 100: break
        if bv > best[0]: best = (bv, bt)
    return best[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["Cora", "Texas"])
    ap.add_argument("--models", nargs="+", default=["mlp", "gcn", "gat", "sage", "glp"]); ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--ablate", action="store_true"); a = ap.parse_args()
    grid = list(itertools.product([64], [.01, .005], [5e-4, 5e-3], [.5, .8]))  # same grid for every model
    abl = {"glp_full": {}, "glp_no_gate": dict(gate=False), "glp_no_high": dict(high=False),
           "glp_no_low": dict(low=False), "glp_no_ego": dict(ego=False)}
    for d in a.datasets:
        data, C = load(d)
        runs = {m: ("glp" if m.startswith("glp") else m, abl.get(m, {})) for m in (list(abl) if a.ablate else a.models)}
        for name, (kind, kw) in runs.items():
            accs = [100 * run(data, C, kind, s, grid, **kw) for s in range(a.splits)]
            print(f"{d:10s} {name:12s} {np.mean(accs):.2f} ± {np.std(accs):.2f}")
