# %% [markdown]
# # Graph Neural Field — Synthetic Proof-of-Concept (Colab A100)
# End-to-end, riproducibile. Dipendenze "blindate" per evitare conflitti del resolver.
# - Dynamic multiplex graph (griglia celle + agenti mobili)
# - Campo vettoriale rischio r*
# - MoE (3 expert + gating) in PyTorch per predire r
# - Operatori differenziali su grafo (∇_G, ∇_G·)
# - Intervento: follow -r̂ vs baseline scalare vs random
# - Context-switching + fine-tuning mirato
# - Export: figure (PNG/PDF), CSV, LaTeX tables, environment.txt

# %% [markdown]
# ## 0) Setup (forzato, senza risolutore) & Reproducibility

# %%
# ⚙️ Install "no-deps" per evitare che il resolver tocchi pacchetti preinstallati in Colab
%pip -q install --upgrade pip
%pip -q install --no-deps "numpy==2.1.2"
%pip -q install --no-deps "scipy==1.14.1"
%pip -q install --no-deps "pandas==2.2.2" "matplotlib==3.8.4" "networkx==3.2.1"
%pip -q install "torch==2.4.0" "torchvision==0.19.0" "torchaudio==2.4.0" --index-url https://download.pytorch.org/whl/cu121

# %%
import os, sys, json, random, hashlib, platform, time, math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from scipy import stats
    SCIPY_OK = True
except Exception as e:
    SCIPY_OK = False
import matplotlib.pyplot as plt

# ✅ fallback minimo se SciPy non è importabile (raro con cella sopra)
def paired_ttest_rel_fallback(a, b):
    # Normal approximation fallback (two-sided), ritorna p-value approssimato
    # Nota: usato solo se SciPy manca; per il paper useremo SciPy.
    import math
    d = np.asarray(a) - np.asarray(b)
    n = d.size
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if sd == 0 or n < 2:
        return 1.0
    t = md / (sd / math.sqrt(n))
    # approx con normale standard (non Student), conservativa
    p = 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return p

# Riproducibilità
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = "gnf_synthetic_outputs"
os.makedirs(OUTDIR, exist_ok=True)

env = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "numpy": np.__version__,
    "scipy": (stats.__version__ if SCIPY_OK else "unavailable"),
    "pandas": pd.__version__,
    "networkx": nx.__version__,
    "matplotlib": plt.matplotlib.__version__,
    "seed": SEED,
}
with open(os.path.join(OUTDIR, "environment.txt"), "w") as f:
    for k,v in env.items():
        f.write(f"{k}: {v}\n")
print(env)

# %% [markdown]
# ## 1) Dynamic Multiplex Graph (celle + agenti)

# %%
@dataclass
class SimConfig:
    grid_w: int = 24
    grid_h: int = 24
    n_agents: int = 500
    T: int = 120
    social_radius: int = 0
    plume_speed: float = 0.3
    plume_width: float = 6.0
    pm25_peak: float = 1.0
    temp_peak: float = 1.0
    base_temp: float = 0.0
    inst_nodes: int = 1
    d_phys: int = 3
    d_behav: int = 2
    d_ctx: int = 2
    d_env: int = 3
    m_axes: int = 3

CFG = SimConfig()

def build_grid_graph(w, h):
    G = nx.grid_2d_graph(h, w)
    for u, v in G.edges():
        G.edges[u, v]["w"] = 1.0
    return G

G_cells = build_grid_graph(CFG.grid_w, CFG.grid_h)
cells = list(G_cells.nodes())
cell_to_idx = {c:i for i,c in enumerate(cells)}

def neighbor_cells(cell, R=1):
    r, c = cell
    neigh = []
    for dr in range(-R, R+1):
        for dc in range(-R, R+1):
            if abs(dr)+abs(dc) <= R:
                cand = (r+dr, c+dc)
                if cand in G_cells:
                    neigh.append(cand)
    return neigh

rng = np.random.default_rng(SEED)
agent_pos = rng.choice(len(cells), size=CFG.n_agents, replace=True)
agent_pos = [cells[idx] for idx in agent_pos]
cell_adj = {c:list(G_cells.neighbors(c)) for c in cells}

# %% [markdown]
# ## 2) Campi ambientali sintetici, stati agenti e r* (ground truth)

# %%
def env_fields(t):
    cx = (t * CFG.plume_speed) % CFG.grid_w
    cy = CFG.grid_h / 2.0
    pm = np.zeros((CFG.grid_h, CFG.grid_w), dtype=np.float32)
    temp = np.zeros_like(pm)
    for (r,c) in cells:
        dx = (c - cx)
        dy = (r - cy)
        pm[r, c] = CFG.pm25_peak * math.exp(-(dx*dx+dy*dy)/(2*CFG.plume_width**2))
        temp[r, c] = CFG.base_temp + CFG.temp_peak * (r / CFG.grid_h) + 0.3*pm[r, c]
    return pm, temp

def social_source_vector(cell, t):
    center = (CFG.grid_h//2 + int(6*np.sin(t/20)), CFG.grid_w//2 + int(6*np.cos(t/20)))
    r, c = cell
    dr = r - center[0]; dc = c - center[1]
    dist = math.sqrt(dr*dr+dc*dc) + 1e-6
    return np.array([dr/dist, dc/dist], dtype=np.float32)

def simulate_states_and_truth():
    T = CFG.T; N = CFG.n_agents
    d_total = CFG.d_phys + CFG.d_behav + CFG.d_ctx + CFG.d_env
    X = np.zeros((T, N, d_total), dtype=np.float32)
    R_true = np.zeros((T, N, CFG.m_axes), dtype=np.float32)
    positions = np.zeros((T, N, 2), dtype=np.int32)

    phys_base = rng.normal(0.0, 0.2, size=(N, CFG.d_phys)).astype(np.float32)
    mood = rng.binomial(1, 0.3, size=(N, )).astype(np.int64)
    medication = rng.binomial(1, 0.2, size=(N, )).astype(np.int64)

    pos = list(agent_pos)
    for t in range(T):
        pm, temp = env_fields(t)
        # movimento (bias di repulsione dalla sorgente sociale)
        new_pos = []
        for i, p in enumerate(pos):
            neigh = cell_adj[p] + [p]
            v = social_source_vector(p, t)
            scores = []
            for q in neigh:
                dr = q[0]-p[0]; dc = q[1]-p[1]
                s = -(dr*v[0] + dc*v[1]) + rng.normal(0, 0.1)
                scores.append(s)
            new_p = neigh[int(np.argmax(scores))]
            new_pos.append(new_p)
        pos = new_pos

        for i, p in enumerate(pos):
            positions[t, i] = [p[0], p[1]]
            phys = phys_base[i] + rng.normal(0, 0.05, size=(CFG.d_phys,)).astype(np.float32)
            if t > 0:
                dr = positions[t, i, 0] - positions[t-1, i, 0]
                dc = positions[t, i, 1] - positions[t-1, i, 1]
                speed = math.sqrt(dr*dr+dc*dc)
            else:
                speed = 0.0
            behav = np.array([speed, rng.normal(0,0.2)], dtype=np.float32)
            ctx = np.array([mood[i], medication[i]], dtype=np.float32)
            r, c = p
            x_env = np.array([pm[r,c], temp[r,c], 1.0], dtype=np.float32)
            x = np.concatenate([phys, behav, ctx, x_env], axis=0)
            X[t, i] = x

            # r* in R^3 (resp, therm, social)
            resp = 1.5*pm[r,c] + 0.5*phys[0] - 0.6*medication[i]
            therm = 1.0*temp[r,c] + 0.4*phys[1]
            soc_mag = 0.6 + 0.6*behav[0]
            R_true[t, i] = np.array([resp, therm, soc_mag], dtype=np.float32)

    return X, R_true, positions

X, R_true, positions = simulate_states_and_truth()
print("Shapes: X", X.shape, "R_true", R_true.shape, "positions", positions.shape)

# Salvataggi grezzi (per riproducibilità)
pd.DataFrame(X.reshape(CFG.T*CFG.n_agents, -1)).to_csv(os.path.join(OUTDIR, "X.csv"), index=False)
pd.DataFrame(R_true.reshape(CFG.T*CFG.n_agents, -1)).to_csv(os.path.join(OUTDIR, "R_true.csv"), index=False)
pd.DataFrame(positions.reshape(CFG.T*CFG.n_agents, -1), columns=["row","col"]).to_csv(os.path.join(OUTDIR, "positions.csv"), index=False)

# %% [markdown]
# ## 3) Split (train/val/test) per agenti

# %%
idx = np.arange(CFG.n_agents)
rng.shuffle(idx)
n_tr = int(0.6*CFG.n_agents)
n_va = int(0.2*CFG.n_agents)
train_idx = idx[:n_tr]
val_idx = idx[n_tr:n_tr+n_va]
test_idx = idx[n_tr+n_va:]
json.dump({"train":train_idx.tolist(),"val":val_idx.tolist(),"test":test_idx.tolist()},
          open(os.path.join(OUTDIR,"splits.json"),"w"))

# %% [markdown]
# ## 4) Dataset & Dataloaders (PyTorch)

# %%
class GNFDataset(torch.utils.data.Dataset):
    def __init__(self, X, R, agent_index):
        self.X = torch.tensor(X[:, agent_index], dtype=torch.float32)
        self.R = torch.tensor(R[:, agent_index], dtype=torch.float32)
        self.T = self.X.shape[0]; self.N = self.X.shape[1]
        self.X = self.X.reshape(self.T*self.N, -1)
        self.R = self.R.reshape(self.T*self.N, -1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.R[i]

BATCH = 2048
train_loader = torch.utils.data.DataLoader(GNFDataset(X,R_true,train_idx), batch_size=BATCH, shuffle=True, drop_last=True)
val_loader   = torch.utils.data.DataLoader(GNFDataset(X,R_true,val_idx),   batch_size=BATCH, shuffle=False)
test_loader  = torch.utils.data.DataLoader(GNFDataset(X,R_true,test_idx),  batch_size=BATCH, shuffle=False)

d_in = X.shape[-1]
m_axes = CFG.m_axes

# %% [markdown]
# ## 5) Modelli: MoE (3 expert + gate), Monolith, Scalar baseline

# %%
# NOTA: Per questo proof-of-concept, gli "Expert" sono implementati come semplici MLP
# che operano sullo stato dell'agente x_i(t). Questo è una semplificazione rispetto
# ai modelli GNN (es. TGAT, EvolveGCN) menzionati nel paper, che opererebbero
# sull'intero grafo G(t). L'obiettivo qui è validare il concetto di campo vettoriale
# e l'architettura MoE in un ambiente controllato.

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x): return self.net(x)

class PhysioExpert(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = MLP(d_in, d_out, hidden=128)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.zeros(1))
    def forward(self, x): return self.linear(self.alpha * x + self.beta)

class EnvExpert(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = MLP(d_in, d_out, hidden=128)
    def forward(self, x): return self.linear(x)

class BehavExpert(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = MLP(d_in, d_out, hidden=128)
    def forward(self, x): return self.linear(x)

class GatingNet(nn.Module):
    def __init__(self, d_in, n_exp=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, n_exp)
        )
    def forward(self, x): return F.softmax(self.gate(x), dim=-1)

class GNFMoE(nn.Module):
    def __init__(self, d_in, m_axes, d_phys=3, d_behav=2, d_ctx=2, d_env=3):
        super().__init__()
        self.gate = GatingNet(d_in, n_exp=3)
        self.exp_phys = PhysioExpert(d_in, m_axes)
        self.exp_env  = EnvExpert(d_in, m_axes)
        self.exp_beh  = BehavExpert(d_in, m_axes)
    def forward(self, x):
        w = self.gate(x)
        y = w[:,0:1]*self.exp_phys(x) + w[:,1:2]*self.exp_env(x) + w[:,2:3]*self.exp_beh(x)
        return y, w

class Monolith(nn.Module):
    def __init__(self, d_in, m_axes):
        super().__init__()
        self.net = MLP(d_in, m_axes, hidden=256)
    def forward(self, x): return self.net(x)

class ScalarBaseline(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.scorer = MLP(d_in, 1, hidden=128)
        self.alpha = nn.Parameter(torch.ones(m_axes))
    def forward(self, x):
        s = self.scorer(x)
        return s * self.alpha

# %% [markdown]
# ## 6) Training (early stopping) e fit modelli

# %%
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, l2=1e-6, is_moe=False):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    best = {"val": 1e9, "state": None}
    for ep in range(1, epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)[0] if is_moe else model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)[0] if is_moe else model(xb)
                va_loss += F.mse_loss(pred, yb).item() * xb.size(0)
        va_loss /= len(val_loader.dataset)
        if va_loss < best["val"]:
            best["val"] = va_loss
            best["state"] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"[{model.__class__.__name__}] Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model

moe = train_model(GNFMoE(d_in, m_axes), train_loader, val_loader, epochs=25, lr=1e-3, is_moe=True)
mono = train_model(Monolith(d_in, m_axes), train_loader, val_loader, epochs=20, lr=1e-3, is_moe=False)
scal = train_model(ScalarBaseline(d_in), train_loader, val_loader, epochs=15, lr=1e-3, is_moe=False)

torch.save(moe.state_dict(),  os.path.join(OUTDIR,"moe.pt"))
torch.save(mono.state_dict(), os.path.join(OUTDIR,"monolith.pt"))
torch.save(scal.state_dict(), os.path.join(OUTDIR,"scalar.pt"))

# %% [markdown]
# ## 7) Metriche test (MAE, Cosine, DirAcc)

# %%
def eval_metrics(model, loader, is_moe=False):
    model.eval()
    Y, P = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yp = model(xb)[0] if is_moe else model(xb)
            Y.append(yb.cpu().numpy()); P.append(yp.cpu().numpy())
    Y = np.concatenate(Y, 0); P = np.concatenate(P, 0)
    mae = float(np.mean(np.abs(P - Y)))
    eps = 1e-8
    cos = float(np.mean(np.sum(P*Y, axis=1)/(np.linalg.norm(P, axis=1)*np.linalg.norm(Y, axis=1)+eps)))
    dir_acc = float(np.mean(np.argmax(P, axis=1) == np.argmax(Y, axis=1)))
    return {"MAE": mae, "cosine": cos, "dir_acc": dir_acc}

metrics = {
    "moe": eval_metrics(moe, test_loader, is_moe=True),
    "monolith": eval_metrics(mono, test_loader, is_moe=False),
    "scalar": eval_metrics(scal, test_loader, is_moe=False),
}
print(metrics)
pd.DataFrame(metrics).to_csv(os.path.join(OUTDIR,"test_metrics.csv"))

# %% [markdown]
# ## 8) Operatori su grafo (snapshot): ∇_G R e ∇_G·R

# %%
def build_social_edges_at_time(t, positions, agent_ids, radius=CFG.social_radius):
    pos = positions[t, agent_ids]
    buckets = {}
    for a, (r,c) in zip(agent_ids, pos):
        key = (int(r), int(c))
        buckets.setdefault(key, []).append(a)
        if radius>=1:
            for q in neighbor_cells(key, R=1):
                buckets.setdefault(q, []).append(a)
    edges = set()
    for agents in buckets.values():
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                edges.add((agents[i], agents[j]))
    return list(edges)

def graph_gradient(R_vec, edges, id_to_row):
    return np.array([R_vec[id_to_row[j]] - R_vec[id_to_row[i]] for (i,j) in edges], dtype=np.float32)

def graph_divergence(R_vec, edges, id_to_row, weights=None):
    n, m = R_vec.shape
    div = np.zeros((n, m), dtype=np.float32)
    if weights is None:
        weights = {e:1.0 for e in edges}
    for (i,j) in edges:
        w = weights.get((i,j), 1.0)
        diff = (R_vec[id_to_row[j]] - R_vec[id_to_row[i]]) * w
        div[id_to_row[i]] += diff
        div[id_to_row[j]] -= diff
    return div

# Unit test piccolo
toy_R = np.array([[0,0,0],[1,0,0],[2,0,0]], dtype=np.float32)
toy_edges = [(0,1),(1,2)]
toy_id_to_row = {0:0,1:1,2:2}
assert graph_gradient(toy_R, toy_edges, toy_id_to_row).shape == (2,3)
assert graph_divergence(toy_R, toy_edges, toy_id_to_row).shape == (3,3)

t_snap = CFG.T - 1
test_ids = test_idx
id_to_row = {aid:i for i,aid in enumerate(test_ids)}
R_true_snap = R_true[t_snap, test_ids]
edges = build_social_edges_at_time(t_snap, positions, test_ids, radius=CFG.social_radius)
gradR = graph_gradient(R_true_snap, edges, id_to_row)
divR  = graph_divergence(R_true_snap, edges, id_to_row)
np.save(os.path.join(OUTDIR,"gradR.npy"), gradR)
np.save(os.path.join(OUTDIR,"divR.npy"),  divR)

# %% [markdown]
# ## 9) Intervento: follow -r̂ vs scalare vs random

# %%
def predict_r(model, x_batch, is_moe=None):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(x_batch, dtype=torch.float32, device=DEVICE)
        yp = model(xb)[0] if (is_moe is True or isinstance(model, GNFMoE)) else model(xb)
    return yp.detach().cpu().numpy()

def pick_move_negative_vector(cell, t, x_i, model):
    neigh = cell_adj[cell] + [cell]
    pm, temp = env_fields(t)
    cand = []
    for q in neigh:
        x_mod = x_i.copy()
        x_mod[-3:] = np.array([pm[q[0],q[1]], temp[q[0],q[1]], 1.0], dtype=np.float32)
        rhat = predict_r(model, x_mod[None, :])
        cand.append((np.linalg.norm(rhat[0]), q))
    return min(cand, key=lambda z:z[0])[1]

def pick_move_scalar(cell, t, x_i, model_scalar):
    neigh = cell_adj[cell] + [cell]
    pm, temp = env_fields(t)
    cand = []
    for q in neigh:
        x_mod = x_i.copy()
        x_mod[-3:] = np.array([pm[q[0],q[1]], temp[q[0],q[1]], 1.0], dtype=np.float32)
        svec = predict_r(model_scalar, x_mod[None, :], is_moe=False)
        cand.append((np.linalg.norm(svec[0]), q))
    return min(cand, key=lambda z:z[0])[1]

def pick_move_random(cell):
    return random.choice(cell_adj[cell] + [cell])

def rollout_policy(agent_id, t0, H, policy, model_main=None, model_scalar=None):
    cum = 0.0
    cell = tuple(positions[t0, agent_id])
    for h in range(H):
        x = X[t0+h, agent_id].copy()
        r = R_true[t0+h, agent_id]
        cum += np.linalg.norm(r)
        if policy == "vec":
            cell = pick_move_negative_vector(cell, t0+h, x, model_main)
        elif policy == "scalar":
            cell = pick_move_scalar(cell, t0+h, x, model_scalar)
        else:
            cell = pick_move_random(cell)
    return cum

H = 10
t0 = CFG.T//3
subset_agents = test_idx[:min(120, len(test_idx))]

vec_scores, scal_scores, rand_scores = [], [], []
for aid in subset_agents:
    vec_scores.append(rollout_policy(aid, t0, H, "vec", model_main=moe))
    scal_scores.append(rollout_policy(aid, t0, H, "scalar", model_scalar=scal))
    rand_scores.append(rollout_policy(aid, t0, H, "rand"))

def summarize(arr):
    arr = np.array(arr, dtype=np.float64)
    return np.mean(arr), np.std(arr)/np.sqrt(len(arr))*1.96

res_interv = {
    "vec_mean": summarize(vec_scores)[0], "vec_CI95": summarize(vec_scores)[1],
    "scalar_mean": summarize(scal_scores)[0], "scalar_CI95": summarize(scal_scores)[1],
    "random_mean": summarize(rand_scores)[0], "random_CI95": summarize(rand_scores)[1],
}
print("Intervention summary:", res_interv)
if SCIPY_OK:
    p_vs_scal = stats.ttest_rel(vec_scores, scal_scores, alternative="less").pvalue
    p_vs_rand = stats.ttest_rel(vec_scores, rand_scores, alternative="less").pvalue
else:
    p_vs_scal = paired_ttest_rel_fallback(vec_scores, scal_scores)
    p_vs_rand = paired_ttest_rel_fallback(vec_scores, rand_scores)
print("vec < scalar p:", p_vs_scal, " | vec < random p:", p_vs_rand)

pd.DataFrame({"vec":vec_scores, "scalar":scal_scores, "random":rand_scores}).to_csv(
    os.path.join(OUTDIR,"intervention_scores.csv"), index=False
)

# %% [markdown]
# ## 10) Context switching: plume invertita + FT su expert selezionati

# %%
def context_switch_env_fields(t):
    cx = (CFG.grid_w - (t * CFG.plume_speed) % CFG.grid_w)
    cy = CFG.grid_h / 2.0
    pm = np.zeros((CFG.grid_h, CFG.grid_w), dtype=np.float32)
    temp = np.zeros_like(pm)
    for (r,c) in cells:
        dx = (c - cx); dy = (r - cy)
        pm[r, c] = 1.4*CFG.pm25_peak * math.exp(-(dx*dx+dy*dy)/(2*(0.8*CFG.plume_width)**2))
        temp[r, c] = CFG.base_temp + 1.2*CFG.temp_peak * (r / CFG.grid_h) + 0.4*pm[r, c]
    return pm, temp

def regenerate_after_switch(T_local=40, switch_at=CFG.T-40):
    X2 = np.copy(X); R2 = np.copy(R_true)
    pos = [tuple(positions[switch_at, i]) for i in range(CFG.n_agents)]
    for t in range(T_local):
        tt = switch_at + t
        pm, temp = context_switch_env_fields(tt)
        new_pos = []
        for i, p in enumerate(pos):
            neigh = cell_adj[p] + [p]
            v = social_source_vector(p, tt)
            scores = []
            for q in neigh:
                dr = q[0]-p[0]; dc = q[1]-p[1]
                s = -(dr*v[0] + dc*v[1]) + rng.normal(0, 0.1)
                scores.append(s)
            new_pos.append(neigh[int(np.argmax(scores))])
        pos = new_pos
        for i, p in enumerate(pos):
            phys = X[tt, i][:CFG.d_phys]
            speed = abs(rng.normal(0.6, 0.2)) if t>0 else 0.0
            behav = np.array([speed, rng.normal(0,0.2)], dtype=np.float32)
            ctx = X[tt, i][CFG.d_phys+CFG.d_behav:CFG.d_phys+CFG.d_behav+CFG.d_ctx]
            r, c = p
            x_env = np.array([pm[r,c], temp[r,c], 1.0], dtype=np.float32)
            X2[tt, i] = np.concatenate([phys, behav, ctx, x_env])
            resp = 1.5*pm[r,c] + 0.5*phys[0] - 0.6*ctx[1]
            therm = 1.0*temp[r,c] + 0.4*phys[1]
            soc_mag = 0.6 + 0.6*behav[0]
            R2[tt, i] = np.array([resp, therm, soc_mag], dtype=np.float32)
    return X2, R2

X_sw, R_sw = regenerate_after_switch()
switch_slice = slice(CFG.T-40, CFG.T)
ft_train = GNFDataset(X_sw[switch_slice], R_sw[switch_slice], train_idx)
ft_val   = GNFDataset(X_sw[switch_slice], R_sw[switch_slice], val_idx)

def finetune_selected_experts(moe_model, ft_train, ft_val, epochs=10, lr=5e-4):
    model = GNFMoE(d_in, m_axes)
    model.load_state_dict(moe_model.state_dict()); model.to(DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    # scegli i 2 expert più pesati in media
    xb,_ = next(iter(torch.utils.data.DataLoader(ft_train, batch_size=4096, shuffle=True)))
    xb = xb.to(DEVICE)
    with torch.no_grad(): w = model.gate(xb).mean(dim=0)  # [3]
    topk = torch.topk(w, k=2).indices.tolist()
    # sblocca ultimo layer del gate e gli expert selezionati
    model.gate.gate[-1].weight.requires_grad_(True)
    model.gate.gate[-1].bias.requires_grad_(True)
    exps = [model.exp_phys, model.exp_env, model.exp_beh]
    for i in topk:
        for p in exps[i].parameters():
            p.requires_grad_(True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-6)
    best = 1e9; best_state=None
    ft_loader = torch.utils.data.DataLoader(ft_train, batch_size=2048, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(ft_val, batch_size=2048, shuffle=False)
    for ep in range(1, epochs+1):
        model.train(); tr = 0.0
        for xb, yb in ft_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            yp,_ = model(xb)
            loss = F.mse_loss(yp, yb)
            loss.backward()
            opt.step()
            tr += loss.item()*xb.size(0)
        tr/=len(ft_loader.dataset)
        model.eval(); va=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                yp,_ = model(xb)
                va += F.mse_loss(yp, yb).item()*xb.size(0)
        va/=len(val_loader.dataset)
        print(f"[Context FT] Epoch {ep:02d} | train {tr:.4f} | val {va:.4f}")
        if va<best: best=va; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model

moe_ft = finetune_selected_experts(moe, ft_train, ft_val)

def eval_on_slice(model, Xs, Rs, idx_agents, label, is_moe=False):
    ds = GNFDataset(Xs[switch_slice], Rs[switch_slice], idx_agents)
    dl = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=False)
    m = eval_metrics(model, dl, is_moe=is_moe)
    print(label, m); return m

m_before = eval_on_slice(moe,    X,    R_true, val_idx, "MoE before switch", is_moe=True)
m_after  = eval_on_slice(moe_ft, X_sw, R_sw,   val_idx, "MoE after FT",      is_moe=True)

# Monolith retrain on switch slice
mono_sw = Monolith(d_in, m_axes).to(DEVICE)
ft_train_dl = torch.utils.data.DataLoader(ft_train, batch_size=2048, shuffle=True, drop_last=True)
ft_val_dl   = torch.utils.data.DataLoader(ft_val,   batch_size=2048, shuffle=False)
def train_model_simple(model, train_dl, val_dl, epochs=10, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    best=1e9; best_state=None
    for ep in range(1, epochs+1):
        model.train(); tr=0.0
        for xb,yb in train_dl:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); yp = model(xb); loss = F.mse_loss(yp,yb); loss.backward(); opt.step()
            tr += loss.item()*xb.size(0)
        tr/=len(train_dl.dataset)
        model.eval(); va=0.0
        with torch.no_grad():
            for xb,yb in val_dl:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                va += F.mse_loss(model(xb), yb).item()*xb.size(0)
        va/=len(val_dl.dataset)
        print(f"[Monolith retrain] Epoch {ep:02d} | train {tr:.4f} | val {va:.4f}")
        if va<best: best=va; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model
mono_sw = train_model_simple(mono_sw, ft_train_dl, ft_val_dl, epochs=10, lr=1e-3)
m_mono = eval_on_slice(mono_sw, X_sw, R_sw, val_idx, "Monolith retrain", is_moe=False)

pd.DataFrame([m_before, m_after, m_mono], index=["moe_before","moe_after_ft","monolith_retrain"]).to_csv(
    os.path.join(OUTDIR,"context_switch_metrics.csv")
)

# %% [markdown]
# ## 11) Figure (PNG/PDF) & Tabelle (CSV/LaTeX)

# %%
def quiver_field(t, agent_ids, model=None, title="Risk vectors (true vs pred)", fname="fig_quiver"):
    samp = agent_ids[:500]
    pos = positions[t, samp]
    pm, temp = env_fields(t)
    true_vecs = R_true[t, samp]
    plt.figure(figsize=(8,7))
    plt.imshow(pm, origin="lower", cmap="magma")
    plt.quiver(pos[:,1], pos[:,0], true_vecs[:,1], true_vecs[:,0],
               angles='xy', scale_units='xy', scale=1.5, width=0.002, alpha=0.8, label="True (therm vs resp)")
    if model is not None:
        Xs = X[t, samp]; pred = predict_r(model, Xs, is_moe=isinstance(model, GNFMoE))
        plt.quiver(pos[:,1], pos[:,0], pred[:,1], pred[:,0],
                   angles='xy', scale_units='xy', scale=1.5, width=0.002, color='cyan', alpha=0.6, label="Pred")
    plt.legend(); plt.title(title)
    for ext in ["png","pdf"]:
        plt.savefig(os.path.join(OUTDIR, f"{fname}.{ext}"), dpi=200, bbox_inches="tight")
    plt.show()

quiver_field(CFG.T-1, test_idx, model=moe, title="True vs Predicted vectors (axes 1 vs 0)", fname="fig_quiver_vectors")

div_mean = np.linalg.norm(divR, axis=1)
plt.figure(figsize=(7,5)); plt.hist(div_mean, bins=40)
plt.title("Distribution of node-wise divergence norms (snapshot)")
for ext in ["png","pdf"]:
    plt.savefig(os.path.join(OUTDIR, f"fig_divergence_hist.{ext}"), dpi=200, bbox_inches="tight")
plt.show()

dfi = pd.DataFrame({"vec":vec_scores, "scalar":scal_scores, "random":rand_scores})
plt.figure(figsize=(6,5))
plt.boxplot([dfi["vec"], dfi["scalar"], dfi["random"]], labels=["GNF -grad","Scalar","Random"])
plt.ylabel("Cumulative true risk (H=10)")
plt.title("Intervention outcomes (lower is better)")
for ext in ["png","pdf"]:
    plt.savefig(os.path.join(OUTDIR, f"fig_intervention.{ext}"), dpi=200, bbox_inches="tight")
plt.show()

test_metrics = pd.read_csv(os.path.join(OUTDIR,"test_metrics.csv"), index_col=0)
with open(os.path.join(OUTDIR,"tables.tex"), "w") as f:
    f.write("% Auto-generated tables for manuscript\n")
    f.write("\\begin{table}[h]\\centering\\caption{Test metrics}\\begin{tabular}{lccc}\\hline\n")
    f.write("Model & MAE & Cosine & DirAcc\\\\\\hline\n")
    for k in test_metrics.columns:
        f.write(f"{k} & {test_metrics.loc['MAE',k]:.4f} & {test_metrics.loc['cosine',k]:.4f} & {test_metrics.loc['dir_acc',k]:.4f}\\\\\n")
    f.write("\\hline\\end{tabular}\\end{table}\n\n")
    f.write("\\begin{table}[h]\\centering\\caption{Intervention outcomes (mean $\\pm$ 95\\% CI)}\\begin{tabular}{lcc}\\hline\n")
    f.write("Policy & Mean & CI95\\\\\\hline\n")
    f.write(f"GNF $-\\nabla$ & {res_interv['vec_mean']:.3f} & {res_interv['vec_CI95']:.3f}\\\\\n")
    f.write(f"Scalar & {res_interv['scalar_mean']:.3f} & {res_interv['scalar_CI95']:.3f}\\\\\n")
    f.write(f"Random & {res_interv['random_mean']:.3f} & {res_interv['random_CI95']:.3f}\\\\\n")
    f.write("\\hline\\end{tabular}\\end{table}\n")

# %% [markdown]
# ## 12) Reproducibility block (hash + p-values)

# %%
def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

files_to_hash = [
    "X.csv", "R_true.csv", "positions.csv", "splits.json",
    "moe.pt", "monolith.pt", "scalar.pt",
    "test_metrics.csv", "gradR.npy", "divR.npy",
    "intervention_scores.csv", "context_switch_metrics.csv",
    "tables.tex", "environment.txt",
    "fig_quiver_vectors.png", "fig_divergence_hist.png", "fig_intervention.png"
]

print("--- REPRODUCIBILITY SUMMARY ---")
print(f"Timestamp: {time.ctime()}")
print(f"Seed: {SEED}")
print("\nKey p-values:")
print(f"  Intervention (GNF < Scalar): p = {p_vs_scal:.5f}")
print(f"  Intervention (GNF < Random): p = {p_vs_rand:.5f}")

print("\nOutput file hashes (SHA256):")
repro_data = {}
for fname in files_to_hash:
    fpath = os.path.join(OUTDIR, fname)
    if os.path.exists(fpath):
        h = file_hash(fpath)
        print(f"  {fname:<30} {h}")
        repro_data[fname] = h
    else:
        print(f"  {fname:<30} NOT FOUND")

with open(os.path.join(OUTDIR, "reproducibility.json"), "w") as f:
    json.dump({
        "timestamp": time.ctime(),
        "seed": SEED,
        "p_values": {"gnf_vs_scalar": p_vs_scal, "gnf_vs_random": p_vs_rand},
        "hashes": repro_data
    }, f, indent=2)

print("\n--- END SUMMARY ---")
