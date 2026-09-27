#!/usr/bin/env python3
"""Learned summaries of the raw points: DeepSets and a GNN, against the paper's hand-built features.

    python extensions/learned/learned.py --config C list              # units: (arch, task[, family])
    python extensions/learned/learned.py --config C unit --index I    # train + test one unit (GPU)
    python extensions/learned/learned.py --config C summarize         # summary.md over every unit
    bash extensions/learned/submit.sh                                  # all of it through SLURM

Input: the bank's raw coordinates. Each point sees its k nearest neighbours as offsets scaled by
sqrt(n) (so distances are in mean spacings, the u = r sqrt(n) axis of the classical features), their
distances, and its own distance to the window edge (clipped at 3 spacings) so a network can learn the
boundary; log n enters at the head, as in PHNet. Nothing is hand-summarised beyond that.

    deepsets   phi(one point's neighbourhood) -> mean + max over points -> rho(., log n)
               one hop: a learned function of single k-NN neighbourhoods, averaged
    gnn        EdgeConv on the fixed k-NN graph: m_ij = MLP[h_i, h_j - h_i, offset_ij, d_ij],
               h_i <- h_i + max_j m_ij, `layers` rounds (receptive field ~ layers x k-NN radius),
               then mean + max pooling -> head(., log n)

Tasks and metrics are the paper's (oneshot/compare.py): 8-way classification (accuracy; families are
balanced), and per-family estimation of log(target) with squared error (RMSE / s.d. of the test
targets, mean over targets). `thin` keeps every k-th theta, as oneshot's `thin`.

Every unit also reports, on its own test rows:
    paper:<model>   the paper run's stored predictions (trained on ALL train rows) -- reference only
    hgb_same_data   gradient boosting on the classical features, trained on THIS unit's train rows:
                    the like-for-like baseline when thin > 1

Output  extensions/results/learned/<name>/<arch>/<task>[/<family>]/{report.json, predictions.npz, model.pt}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (FAMILIES, ROOT, TARGETS, load_config, paper_classifier, paper_estimator,  # noqa: E402
                    points, results_dir, rows as bank_rows, write_json)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HGB = {"max_iter": 400, "learning_rate": 0.1, "max_leaf_nodes": 31, "l2_regularization": 1.0,
       "early_stopping": False, "random_state": 0}                       # oneshot/learners.py HGB defaults


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------- units

def units(cfg: dict) -> list[tuple[str, str, str | None]]:
    out = []
    for arch in cfg["archs"]:
        if cfg.get("classify", True):
            out.append((arch, "classify", None))
        out += [(arch, "estimate", f) for f in cfg["estimate"]]
    return out


def unit_dir(cfg: dict, arch: str, task: str, family: str | None) -> Path:
    return results_dir("learned", cfg["name"], arch, task, *([family] if family else []))


# -------------------------------------------------------------------------------------- data

class Clouds(torch.utils.data.Dataset):
    def __init__(self, X: list[np.ndarray], y: np.ndarray):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_collate(k: int):
    def collate(batch):
        feats, rel, dist, nbr, which, logn, ys = [], [], [], [], [], [], []
        start = 0
        for b, (X, y) in enumerate(batch):
            n = len(X)
            s = np.sqrt(n)
            d, j = cKDTree(X).query(X, k + 1)
            d, j = d[:, 1:], j[:, 1:]                                    # drop the point itself
            rel.append((X[j] - X[:, None, :]) * s)
            dist.append(d * s)
            edge = np.minimum(np.minimum(X, 1 - X).min(1) * s, 3.0)
            feats.append(np.stack([edge, d[:, 0] * s, d[:, -1] * s], 1))
            nbr.append(j + start)
            which.append(np.full(n, b))
            logn.append(np.log(n))
            ys.append(y)
            start += n
        cat = lambda a, dt: torch.from_numpy(np.concatenate(a).astype(dt))
        return {"feat": cat(feats, np.float32), "rel": cat(rel, np.float32), "dist": cat(dist, np.float32),
                "nbr": cat(nbr, np.int64), "which": cat(which, np.int64), "B": len(batch),
                "logn": torch.tensor(logn, dtype=torch.float32)[:, None], "y": torch.from_numpy(np.stack(ys))}
    return collate


# ------------------------------------------------------------------------------------- models

def mlp(dims, final_act=True):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(tnn.Linear(a, b))
        if i < len(dims) - 2 or final_act:
            layers.append(tnn.ReLU())
    return tnn.Sequential(*layers)


def pool(h, which, B):
    """mean and max of per-point features over each cloud -> (B, 2 d)."""
    d = h.shape[1]
    cnt = torch.zeros(B, 1, device=h.device).index_add_(0, which, torch.ones_like(h[:, :1]))
    mean = torch.zeros(B, d, device=h.device).index_add_(0, which, h) / cnt
    mx = torch.full((B, d), -1e9, device=h.device).scatter_reduce(0, which[:, None].expand(-1, d), h, "amax")
    return torch.cat([mean, mx], 1)


class DeepSets(tnn.Module):
    def __init__(self, k: int, n_out: int, hidden: int = 128, logn_stats=(0.0, 1.0), **_):
        super().__init__()
        self.phi = mlp([3 * k + 3, hidden, hidden, hidden])
        self.rho = mlp([2 * hidden + 1, hidden, 64, n_out], final_act=False)
        self.register_buffer("logn_stats", torch.tensor(logn_stats))

    def forward(self, b):
        x = torch.cat([b["rel"].flatten(1), b["dist"], b["feat"]], 1)
        g = pool(self.phi(x), b["which"], b["B"])
        z = (b["logn"] - self.logn_stats[0]) / self.logn_stats[1]
        return self.rho(torch.cat([g, z], 1))


class GNN(tnn.Module):
    def __init__(self, k: int, n_out: int, hidden: int = 64, layers: int = 3, logn_stats=(0.0, 1.0), **_):
        super().__init__()
        self.inp = mlp([3, hidden])
        self.msg = tnn.ModuleList([mlp([2 * hidden + 3, hidden, hidden]) for _ in range(layers)])
        self.norm = tnn.ModuleList([tnn.LayerNorm(hidden) for _ in range(layers)])
        self.head = mlp([2 * hidden + 1, 128, 64, n_out], final_act=False)
        self.register_buffer("logn_stats", torch.tensor(logn_stats))

    def forward(self, b):
        h = self.inp(b["feat"])
        nbr, k = b["nbr"], b["nbr"].shape[1]
        geo = torch.cat([b["rel"], b["dist"][..., None]], -1)                    # (N, k, 3)
        for msg, norm in zip(self.msg, self.norm):
            hi = h[:, None, :].expand(-1, k, -1)
            m = msg(torch.cat([hi, h[nbr] - hi, geo], -1)).amax(1)
            h = norm(h + m)
        g = pool(h, b["which"], b["B"])
        z = (b["logn"] - self.logn_stats[0]) / self.logn_stats[1]
        return self.head(torch.cat([g, z], 1))


ARCHS = {"deepsets": DeepSets, "gnn": GNN}


# ----------------------------------------------------------------------------------- training

def to_device(b):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def run_epoch(model, loader, loss_fn, opt=None):
    model.train(opt is not None)
    tot, n, outs = 0.0, 0, []
    with torch.set_grad_enabled(opt is not None):
        for b in loader:
            b = to_device(b)
            out = model(b)
            loss = loss_fn(out, b["y"])
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                tnn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            else:
                outs.append(out.cpu())
            tot += float(loss) * b["B"]
            n += b["B"]
    return tot / n, (torch.cat(outs).numpy() if outs else None)


def fit(cfg, arch, X, y, tr, va, te, n_out, loss_fn):
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    logn = np.log([len(X[i]) for i in tr])
    model = ARCHS[arch](cfg["k"], n_out, logn_stats=(float(logn.mean()), float(logn.std())),
                        **cfg["archs"][arch]).to(DEVICE)
    collate = make_collate(cfg["k"])
    mk = lambda idx, shuf: torch.utils.data.DataLoader(
        Clouds([X[i] for i in idx], y[idx]), batch_size=t["batch_size"], shuffle=shuf, collate_fn=collate,
        num_workers=t["workers"], persistent_workers=t["workers"] > 0, pin_memory=DEVICE == "cuda")
    L_tr, L_va, L_te = mk(tr, True), mk(va, False), mk(te, False)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, t["patience"] // 2))
    best, best_state, since, hist = np.inf, None, 0, []
    for ep in range(1, t["epochs"] + 1):
        t0 = time.time()
        l_tr, _ = run_epoch(model, L_tr, loss_fn, opt)
        l_va, _ = run_epoch(model, L_va, loss_fn)
        sched.step(l_va)
        hist.append({"epoch": ep, "train": l_tr, "val": l_va, "seconds": time.time() - t0})
        star = l_va < best
        if star:
            best, since = l_va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        log(f"  epoch {ep:3d}  train {l_tr:.4f}  val {l_va:.4f}  ({time.time() - t0:.0f}s){'  *' if star else ''}")
        if since >= t["patience"]:
            break
    model.load_state_dict(best_state)
    _, out = run_epoch(model, L_te, loss_fn)
    n_params = sum(p.numel() for p in model.parameters())
    return model, out, {"history": hist, "best_val": best, "n_params": n_params}


# ------------------------------------------------------------------------------------ baselines

def classical(r: pd.DataFrame) -> np.ndarray:
    parts = []
    for f in r.family.unique():
        z = np.load(ROOT / "data" / "cascade" / "features" / f"{f}.npz")
        parts.append(pd.DataFrame(z["X"], index=z["case_id"]))
    return pd.concat(parts).loc[r.index].to_numpy(np.float32)


def rmse_sd(pred_log: np.ndarray, true_log: np.ndarray) -> tuple[float, list[float]]:
    per = np.sqrt(((pred_log - true_log) ** 2).mean(0)) / true_log.std(0)
    return float(per.mean()), [float(v) for v in per]


# ----------------------------------------------------------------------------------------- unit

def run_unit(cfg: dict, index: int) -> None:
    arch, task, family = units(cfg)[index]
    out = unit_dir(cfg, arch, task, family)
    if (out / "report.json").exists():
        log(f"{arch} {task} {family or ''}: done already")
        return
    t0 = time.time()
    r = bank_rows([family] if family else FAMILIES, cfg["thin"])
    pts = points(r)
    X = [pts[c] for c in r.index]
    idx = {s: np.flatnonzero((r.split == s).to_numpy()) for s in ("train", "val", "test")}
    tr, va, te = idx["train"], idx["val"], idx["test"]
    log(f"{arch} {task} {family or ''}: {len(tr)} train / {len(va)} val / {len(te)} test clouds, "
        f"device {DEVICE} ({time.time() - t0:.0f}s)")
    test_ids = r.index[te]
    rep = {"arch": arch, "task": task, "family": family, "thin": cfg["thin"], "k": cfg["k"],
           "n": {"train": len(tr), "val": len(va), "test": len(te)}, "arch_params": cfg["archs"][arch]}
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    F = classical(r)

    if task == "classify":
        y = r.family.map({f: i for i, f in enumerate(FAMILIES)}).to_numpy(np.int64)
        model, logits, info = fit(cfg, arch, X, y, tr, va, te, len(FAMILIES), tnn.CrossEntropyLoss())
        P = torch.softmax(torch.from_numpy(logits), 1).numpy()
        yt = y[te]

        def metrics(P):
            P = np.clip(P, 1e-12, 1)
            return {"accuracy": float((P.argmax(1) == yt).mean()), "nll": float(-np.log(P[np.arange(len(yt)), yt]).mean()),
                    "recall": {f: float((P.argmax(1)[yt == i] == i).mean()) for i, f in enumerate(FAMILIES)}}
        rep["model"] = metrics(P)
        hgb = HistGradientBoostingClassifier(**HGB).fit(F[tr], y[tr])
        rep["hgb_same_data"] = metrics(hgb.predict_proba(F[te]))
        for m in cfg["paper_baselines"]["classify"]:
            rep[f"paper:{m}"] = metrics(paper_classifier(m).loc[test_ids, FAMILIES].to_numpy())
        np.savez(out / "predictions.npz", case_id=test_ids.to_numpy(str), posterior=P, classes=np.array(FAMILIES))
    else:
        tg = TARGETS[family]
        Y = np.log(r[tg].to_numpy(float))
        mu, sd = Y[tr].mean(0), Y[tr].std(0)
        Yz = ((Y - mu) / sd).astype(np.float32)
        model, pred, info = fit(cfg, arch, X, Yz, tr, va, te, len(tg), tnn.MSELoss())
        pred = np.clip(pred * sd + mu, Y[tr].min(0), Y[tr].max(0))          # clip_to_train_range, as oneshot
        m, per = rmse_sd(pred, Y[te])
        rep["model"] = {"rmse_sd": m, "per_target": dict(zip(tg, per))}
        hp = np.stack([HistGradientBoostingRegressor(**HGB).fit(F[tr], Y[tr, j]).predict(F[te]) for j in range(len(tg))], 1)
        m, per = rmse_sd(np.clip(hp, Y[tr].min(0), Y[tr].max(0)), Y[te])
        rep["hgb_same_data"] = {"rmse_sd": m, "per_target": dict(zip(tg, per))}
        for pm in cfg["paper_baselines"]["estimate"]:
            m, per = rmse_sd(np.log(paper_estimator(family, pm).loc[test_ids, tg].to_numpy()), Y[te])
            rep[f"paper:{pm}"] = {"rmse_sd": m, "per_target": dict(zip(tg, per))}
        np.savez(out / "predictions.npz", case_id=test_ids.to_numpy(str), theta_hat=np.exp(pred), targets=np.array(tg))
    rep["fit"] = info
    rep["seconds"] = time.time() - t0
    torch.save({"state_dict": model.state_dict(), "arch": arch, "cfg": cfg}, out / "model.pt")
    write_json(out / "report.json", rep)
    key = "accuracy" if task == "classify" else "rmse_sd"
    log(f"-> {out}  " + "  ".join(f"{k}: {v[key]:.4f}" for k, v in rep.items() if isinstance(v, dict) and key in v))


# ------------------------------------------------------------------------------------- summary

def summarize(cfg: dict) -> None:
    import json
    base = results_dir("learned", cfg["name"])
    reps = {}
    for arch, task, fam in units(cfg):
        p = unit_dir(cfg, arch, task, fam) / "report.json"
        if p.exists():
            reps[(arch, task, fam)] = json.loads(p.read_text())
    L = [f"# Learned summaries: {cfg['name']} (thin {cfg['thin']}, k {cfg['k']})", "",
         "Same test rows in every column. `hgb same data` = gradient boosting on the classical features "
         "trained on the same (thinned) train rows as the networks: the like-for-like comparison. "
         f"`paper:*` = the paper run's models, trained on all train rows ({cfg['thin']}x the networks' data): reference.", ""]
    cls = {a: reps[(a, "classify", None)] for a in cfg["archs"] if (a, "classify", None) in reps}
    if cls:
        any_rep = next(iter(cls.values()))
        cols = ["hgb_same_data", *[k for k in any_rep if k.startswith("paper:")]]
        L += ["## Classification (accuracy / NLL)", "",
              "| | " + " | ".join(cls) + " | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
              "|---|" + "---|" * (len(cls) + len(cols))]
        L.append("| all | " + " | ".join(f"{r['model']['accuracy']:.3f} / {r['model']['nll']:.3f}" for r in cls.values())
                 + " | " + " | ".join(f"{any_rep[c]['accuracy']:.3f} / {any_rep[c]['nll']:.3f}" for c in cols) + " |")
        for f in FAMILIES:
            L.append(f"| recall {f} | " + " | ".join(f"{r['model']['recall'][f]:.2f}" for r in cls.values())
                     + " | " + " | ".join(f"{any_rep[c]['recall'][f]:.2f}" for c in cols) + " |")
        L.append("")
    fams = [f for f in cfg["estimate"] if any((a, "estimate", f) in reps for a in cfg["archs"])]
    if fams:
        cols = ["hgb_same_data", *[f"paper:{m}" for m in cfg["paper_baselines"]["estimate"]]]
        L += ["## Estimation (RMSE(log θ)/s.d., mean over targets; lower is better)", "",
              "| family | " + " | ".join(cfg["archs"]) + " | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
              "|---|" + "---|" * (len(cfg["archs"]) + len(cols))]
        for f in fams:
            some = next(reps[(a, "estimate", f)] for a in cfg["archs"] if (a, "estimate", f) in reps)
            L.append(f"| {f} | " + " | ".join(f"{reps[(a, 'estimate', f)]['model']['rmse_sd']:.3f}"
                                              if (a, "estimate", f) in reps else "—" for a in cfg["archs"])
                     + " | " + " | ".join(f"{some[c]['rmse_sd']:.3f}" for c in cols) + " |")
        L.append("")
    L += ["## Training", "", "| unit | params | epochs | best val | s/epoch | total s |", "|---|---|---|---|---|---|"]
    for (a, t, f), r in reps.items():
        h = r["fit"]["history"]
        L.append(f"| {a} {t} {f or ''} | {r['fit']['n_params']:,} | {len(h)} | {r['fit']['best_val']:.4f} | "
                 f"{np.mean([e['seconds'] for e in h]):.0f} | {r['seconds']:.0f} |")
    missing = [u for u in units(cfg) if u not in reps]
    if missing:
        L += ["", "Missing units: " + ", ".join(" ".join(x for x in u if x) for u in missing)]
    (base / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ----------------------------------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "smoke.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    u = sub.add_parser("unit")
    u.add_argument("--index", type=int, required=True)
    sub.add_parser("summarize")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "list":
        for i, (a, t, f) in enumerate(units(cfg)):
            done = (unit_dir(cfg, a, t, f) / "report.json").exists()
            print(f"{i} {a} {t} {f or ''}{'  (done)' if done else ''}")
    elif args.cmd == "unit":
        run_unit(cfg, args.index)
    else:
        summarize(cfg)


if __name__ == "__main__":
    main()
