"""Limited KT utility evaluation — clean single-copy backbone."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "response_limited_kt_config.json"
PROCESSED = ROOT / "data_processed"
LLM_FEATURES = ROOT / "artifacts" / "scores" / "llm_item_scores.parquet"
MASK_DIR = ROOT / "artifacts" / "response_limited_kt" / "masks"
RUN_DIR = ROOT / "runs" / "response_limited_kt"
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints" / "limited_kt"
REF_TABLE = ROOT / "results" / "AUTHENTIC_DIFFICULTY_REFERENCES.csv"
SURFACE_TABLE = ROOT / "results" / "AUTHENTIC_ITEM_SURFACE_FEATURES.csv"
_IX_CACHE: dict[str, pd.DataFrame] = {}
_SCOREABLE_CACHE: dict[str, set[str]] = {}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CleanKT(nn.Module):
    """Single-copy item GRU; optional scalar concatenated at prediction head."""

    def __init__(self, n_items: int, dim: int = 32, use_scalar: bool = False):
        super().__init__()
        self.use_scalar = use_scalar
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.rnn = nn.GRU(dim + 1, dim, batch_first=True)
        pred_in = dim + (1 if use_scalar else 0)
        self.pred = nn.Linear(pred_in, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, items: torch.Tensor, corrects: torch.Tensor, scalars: torch.Tensor | None = None):
        x = self.item_emb(items)
        prev = torch.cat(
            [torch.zeros(items.size(0), 1, device=items.device), corrects[:, :-1]],
            dim=1,
        ).unsqueeze(-1)
        out, _ = self.rnn(torch.cat([x, prev], dim=-1))
        if self.use_scalar and scalars is not None:
            logits = self.pred(torch.cat([out, scalars.unsqueeze(-1)], dim=-1)).squeeze(-1)
        else:
            logits = self.pred(out).squeeze(-1)
        return logits


class KTSequenceDataset(Dataset):
    def __init__(self, sequences: list[dict]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        s = self.sequences[idx]
        return {
            "items": torch.tensor(s["items"], dtype=torch.long),
            "corrects": torch.tensor(s["corrects"], dtype=torch.float),
            "scalars": torch.tensor(s["scalars"], dtype=torch.float),
            "mask": torch.tensor(s["mask"], dtype=torch.bool),
        }


def collate_sequences(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(b["items"].size(0) for b in batch)
    items, corrects, scalars, masks = [], [], [], []
    for b in batch:
        pad = max_len - b["items"].size(0)
        items.append(torch.nn.functional.pad(b["items"], (0, pad)))
        corrects.append(torch.nn.functional.pad(b["corrects"], (0, pad), value=-1))
        scalars.append(torch.nn.functional.pad(b["scalars"], (0, pad)))
        masks.append(torch.nn.functional.pad(b["mask"], (0, pad)))
    return {
        "items": torch.stack(items),
        "corrects": torch.stack(corrects),
        "scalars": torch.stack(scalars),
        "mask": torch.stack(masks),
    }


@dataclass
class DatasetBundle:
    dataset: str
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]
    train_seqs: list[dict]
    val_seqs: list[dict]
    test_seqs: list[dict]
    scalar_maps: dict[str, dict[str, float]]
    split_hash: str
    universe_hash: str


def load_scoreable_universe(dataset: str) -> pd.DataFrame:
    items = pd.read_parquet(PROCESSED / dataset / "items.parquet")
    return items[items["eligible_for_llm_scoring"]].copy()


def _scoreable_ids(dataset: str) -> set[str]:
    if dataset not in _SCOREABLE_CACHE:
        _SCOREABLE_CACHE[dataset] = set(load_scoreable_universe(dataset)["item_id_hash"])
    return _SCOREABLE_CACHE[dataset]


def _load_interactions_cached(dataset: str) -> pd.DataFrame:
    if dataset not in _IX_CACHE:
        cols = ["student_id_hash", "item_id_hash", "correct", "split_assignment", "sequence_index"]
        ix = pd.read_parquet(PROCESSED / dataset / "interactions.parquet", columns=cols)
        ix = ix[ix["item_id_hash"].isin(_scoreable_ids(dataset))]
        _IX_CACHE[dataset] = ix
    return _IX_CACHE[dataset]


def _deterministic_rank(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    keys = (
        df["student_id_hash"].astype(str)
        + "|"
        + df["sequence_index"].astype(str)
        + "|"
        + str(seed)
    )
    df = df.copy()
    df["_rank"] = keys.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:12], 16))
    return df.sort_values("_rank")


def build_exposure_mask(dataset: str, exposure, cfg: dict) -> tuple[pd.DataFrame, str]:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    path = MASK_DIR / f"{dataset}_exposure_{exposure}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df, sha256_file(path)

    cols = ["student_id_hash", "item_id_hash", "correct", "split_assignment", "sequence_index"]
    ix = _load_interactions_cached(dataset).copy()
    scoreable = _scoreable_ids(dataset)
    train = ix[ix["split_assignment"] == "train"].copy()

    if exposure == "warm":
        kept = train
    else:
        cap = int(exposure)
        parts = []
        for _, g in train.groupby("item_id_hash"):
            g = _deterministic_rank(g, cfg["model"]["mask_seed"])
            parts.append(g.head(cap))
        kept = pd.concat(parts, ignore_index=True) if parts else train.iloc[0:0]

    kept = kept.assign(dataset=dataset, exposure_level=str(exposure), keep=True)
    kept.to_parquet(path, index=False)
    return kept, sha256_file(path)


def build_scalar_maps(dataset: str, bundle_items: set[str], exposure, cfg: dict) -> dict[str, dict[str, float]]:
    llm = pd.read_parquet(LLM_FEATURES)
    llm = llm[llm["dataset"] == dataset]
    mini = llm[llm["model_identifier"] == "gpt-4o-mini"].set_index("item_id_hash")["scalar_difficulty"]
    g54 = llm[llm["model_identifier"] == "gpt-5.4"].set_index("item_id_hash")["scalar_difficulty"]

    refs = pd.read_csv(REF_TABLE)
    train_ref = refs[(refs["dataset"] == dataset) & (refs["reference_scope"] == "deployable_train")]
    test_ref = refs[(refs["dataset"] == dataset) & (refs["reference_scope"] == "held_out_test")]
    oracle_ref = refs[(refs["dataset"] == dataset) & (refs["reference_scope"] == "oracle_diagnostic")]

    mask_df, _ = build_exposure_mask(dataset, exposure, cfg)
    emp_rows = []
    for item in bundle_items:
        sub = mask_df[mask_df["item_id_hash"] == item]
        n = len(sub)
        err = 1 - sub["correct"].mean() if n else np.nan
        emp_rows.append({"item_id_hash": item, "n": n, "raw_err": err})
    emp = pd.DataFrame(emp_rows)
    alpha, beta = cfg["train_emp_prior"]
    global_mean = float(train_ref["smoothed_error_beta_1_1"].mean())
    emp["train_emp"] = emp.apply(
        lambda r: global_mean if r["n"] == 0 else (r["raw_err"] * r["n"] + alpha) / (r["n"] + alpha + beta)
        if np.isfinite(r["raw_err"]) else global_mean,
        axis=1,
    )
    train_map = train_ref.set_index("item_id_hash")["smoothed_error_beta_1_1"]
    oracle_map = oracle_ref.set_index("item_id_hash")["smoothed_error_beta_1_1"]

    surface = pd.read_csv(SURFACE_TABLE)
    surface = surface[(surface["dataset"] == dataset) & (surface["item_id_hash"].isin(bundle_items))]
    char = surface.set_index("item_id_hash")["char_length"].astype(float)
    char_min = float(char.min())
    char_max = float(char.max())
    if np.isfinite(char_min) and np.isfinite(char_max) and char_max > char_min:
        char_norm = ((char - char_min) / (char_max - char_min)).to_dict()
    else:
        char_norm = {item: 0.5 for item in bundle_items}

    rng = np.random.default_rng(cfg["model"]["mask_seed"])
    mini_vals = mini.reindex(list(bundle_items)).dropna().values
    random_draw = {}
    if len(mini_vals):
        shuffled = rng.choice(mini_vals, size=len(bundle_items), replace=True)
        for item, v in zip(bundle_items, shuffled):
            random_draw[item] = float(v)

    return {
        "gpt-4o-mini": mini.to_dict(),
        "gpt-5.4": g54.to_dict(),
        "random_matched": random_draw,
        "char_length_norm": {k: float(v) for k, v in char_norm.items()},
        "train_empirical": emp.set_index("item_id_hash")["train_emp"].to_dict(),
        "oracle_empirical": oracle_map.to_dict(),
        "global_train_mean": global_mean,
    }


def sequences_from_interactions(
    interactions: pd.DataFrame,
    item_to_idx: dict[str, int],
    scalar_value: dict[str, float],
    default_scalar: float,
    max_seq_len: int,
) -> list[dict]:
    seqs = []
    for _, g in interactions.groupby("student_id_hash"):
        g = g.sort_values("sequence_index").head(max_seq_len)
        items = [item_to_idx[i] for i in g["item_id_hash"]]
        corrects = g["correct"].astype(float).tolist()
        scalars = [float(scalar_value.get(i, default_scalar)) for i in g["item_id_hash"]]
        seqs.append({
            "items": items,
            "corrects": corrects,
            "scalars": scalars,
            "mask": [True] * len(items),
        })
    return seqs


def load_dataset_bundle(
    dataset: str,
    exposure,
    cfg: dict,
    *,
    max_train_students: int | None = None,
    max_test_students: int | None = None,
) -> DatasetBundle:
    scoreable = load_scoreable_universe(dataset)
    item_ids = sorted(scoreable["item_id_hash"].tolist())
    item_to_idx = {h: i + 1 for i, h in enumerate(item_ids)}
    idx_to_item = {i: h for h, i in item_to_idx.items()}

    mask_df, mask_hash = build_exposure_mask(dataset, exposure, cfg)
    kept_pairs = set(zip(mask_df["student_id_hash"], mask_df["item_id_hash"]))

    ix = _load_interactions_cached(dataset).copy()

    if exposure != "warm":
        ix_train = ix[ix["split_assignment"] == "train"].copy()
        ix_train["_key"] = list(zip(ix_train["student_id_hash"], ix_train["item_id_hash"]))
        ix_train = ix_train[ix_train["_key"].isin(kept_pairs)]
        ix_train = ix_train.drop(columns=["_key"])
        other = ix[ix["split_assignment"] != "train"]
        ix = pd.concat([ix_train, other], ignore_index=True)

    scalar_maps = build_scalar_maps(dataset, set(item_ids), exposure, cfg)

    def _cap_students(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
        if n is None:
            return df
        students = sorted(df["student_id_hash"].unique())[:n]
        return df[df["student_id_hash"].isin(students)]

    train_ix = _cap_students(ix[ix["split_assignment"] == "train"], max_train_students)
    val_ix = _cap_students(ix[ix["split_assignment"] == "val"], max_test_students)
    test_ix = _cap_students(ix[ix["split_assignment"] == "test"], max_test_students)

    mcfg = cfg["model"]
    def build(ix_sub, smap, default):
        return sequences_from_interactions(ix_sub, item_to_idx, smap, default, mcfg["max_seq_len"])

    train_seqs = build(train_ix, {}, 0.0)
    val_seqs = build(val_ix, {}, 0.0)
    test_seqs = build(test_ix, {}, 0.0)

    universe_hash = sha256_text(",".join(item_ids))
    split_hash = sha256_file(PROCESSED / dataset / "splits.parquet")

    return DatasetBundle(
        dataset=dataset,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        test_seqs=test_seqs,
        scalar_maps=scalar_maps,
        split_hash=split_hash,
        universe_hash=universe_hash,
    )


def train_and_evaluate(
    bundle: DatasetBundle,
    condition: str,
    seed: int,
    cfg: dict,
    *,
    run_id: str,
) -> dict[str, Any]:
    set_seed(seed)
    cond = cfg["conditions"][condition]
    use_scalar = cond["use_scalar"]
    src = cond["scalar_source"]
    default = bundle.scalar_maps.get("global_train_mean", 0.5)
    smap = bundle.scalar_maps.get(src, {}) if src else {}

    def attach_scalars(seqs: list[dict]) -> list[dict]:
        out = []
        for s in seqs:
            items_hash = [bundle.idx_to_item[i] for i in s["items"]]
            out.append({
                **s,
                "scalars": [float(smap.get(h, default)) for h in items_hash],
            })
        return out

    train_seqs = attach_scalars(bundle.train_seqs)
    val_seqs = attach_scalars(bundle.val_seqs)
    test_seqs = attach_scalars(bundle.test_seqs)

    n_items = len(bundle.item_to_idx) + 1
    model = CleanKT(n_items, cfg["model"]["emb_dim"], use_scalar=use_scalar)
    opt = optim.Adam(model.parameters(), lr=cfg["model"]["lr"])

    train_loader = DataLoader(KTSequenceDataset(train_seqs), batch_size=cfg["model"]["batch_size"], shuffle=True, collate_fn=collate_sequences) if train_seqs else None
    val_loader = DataLoader(KTSequenceDataset(val_seqs), batch_size=cfg["model"]["batch_size"], shuffle=False, collate_fn=collate_sequences) if val_seqs else None

    best_val = float("inf")
    best_state = None
    patience = 0
    epoch = -1
    if train_loader and val_loader and len(train_seqs) > 0:
        for epoch in range(cfg["model"]["max_epochs"]):
            model.train()
            for batch in train_loader:
                logits = model(batch["items"], batch["corrects"], batch["scalars"] if use_scalar else None)
                mask = batch["mask"] & (batch["corrects"] >= 0)
                if mask.sum() == 0:
                    continue
                loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], batch["corrects"][mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_loss = _eval_loss(model, val_loader, use_scalar)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience > cfg["model"]["early_stop_patience"]:
                    break

    if best_state:
        model.load_state_dict(best_state)

    test_loader = DataLoader(KTSequenceDataset(test_seqs), batch_size=cfg["model"]["batch_size"], shuffle=False, collate_fn=collate_sequences) if test_seqs else None
    metrics = _eval_metrics(model, test_loader, use_scalar) if test_loader else {
        "log_loss": float("nan"), "brier": float("nan"), "auc": float("nan"),
        "accuracy": float("nan"), "ece": float("nan"), "n_predictions": 0,
    }
    metrics.update({
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "best_val_log_loss": best_val,
        "best_epoch": epoch + 1,
        "n_parameters": model.count_parameters(),
        "deployable": cond["deployable"],
    })
    return metrics


def _eval_loss(model, loader, use_scalar) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["items"], batch["corrects"], batch["scalars"] if use_scalar else None)
            mask = batch["mask"] & (batch["corrects"] >= 0)
            if mask.sum() == 0:
                continue
            losses.append(nn.functional.binary_cross_entropy_with_logits(logits[mask], batch["corrects"][mask]).item())
    return float(np.mean(losses)) if losses else float("inf")


def _eval_metrics(model, loader, use_scalar) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["items"], batch["corrects"], batch["scalars"] if use_scalar else None)
            mask = batch["mask"] & (batch["corrects"] >= 0)
            if mask.sum() == 0:
                continue
            prob = torch.sigmoid(logits[mask]).cpu().numpy()
            ys.extend(batch["corrects"][mask].cpu().numpy().tolist())
            ps.extend(prob.tolist())
    y = np.asarray(ys)
    p = np.asarray(ps)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean((p >= 0.5) == y))
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    ece = _ece(y, p)
    return {"log_loss": log_loss, "brier": brier, "auc": auc, "accuracy": acc, "ece": ece, "n_predictions": int(len(y))}


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)
