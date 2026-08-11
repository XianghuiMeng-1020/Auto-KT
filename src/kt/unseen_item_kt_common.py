"""Genuine unseen-item cold-start and SAKT backbone knowledge tracing.


"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
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
CONFIG_PATH = ROOT / "configs" / "unseen_item_kt_config.json"
PROCESSED = ROOT / "data_processed"
LLM_FEATURES = ROOT / "artifacts" / "scores" / "llm_item_scores.parquet"
MASK_DIR = ROOT / "artifacts" / "response_limited_kt" / "masks"
REF_TABLE = ROOT / "results" / "AUTHENTIC_DIFFICULTY_REFERENCES.csv"
SURFACE_TABLE = ROOT / "results" / "AUTHENTIC_ITEM_SURFACE_FEATURES.csv"
FOLD_DIR = ROOT / "artifacts" / "item_folds"
COLDSTART_RUN_DIR = ROOT / "runs" / "unseen_item_kt"
SAKT_LIMITED_RUN_DIR = ROOT / "runs" / "sakt_response_limited_kt"
GATE_DIR = ROOT / "runs" / "unseen_item_kt" / "checks"

# Reuse scoreable filtering helpers from limited_kt without altering its registry.
from limited_kt_common import (  # noqa: E402
    _load_interactions_cached,
    build_exposure_mask,
    load_scoreable_universe,
    sha256_file,
    sha256_text,
    utc_now,
)

_IX_CACHE = {}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # Prefer CPU for these compact KT models: on Apple MPS, kernel-launch
    # overhead dominates and junyi runs were ~3x slower than the CPU RQ4 baseline.
    return torch.device("cpu")


class CleanGRU(nn.Module):
    """One-layer item GRU; optional scalar concatenated only at prediction head."""

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


class SAKT(nn.Module):
    """Single-block SAKT; scalar concatenated only at prediction head."""

    def __init__(
        self,
        n_items: int,
        d_model: int = 64,
        n_heads: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 80,
        use_scalar: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.use_scalar = use_scalar
        self.d_model = d_model
        self.n_item_slots = n_items + 1  # indices 0..n_items
        self.item_emb = nn.Embedding(self.n_item_slots, d_model, padding_idx=0)
        # interaction ids: item (incorrect/prev0) or item + n_item_slots (correct)
        self.interaction_emb = nn.Embedding(2 * self.n_item_slots, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        pred_in = d_model + (1 if use_scalar else 0)
        self.pred = nn.Linear(pred_in, 1)
        self.max_seq_len = max_seq_len

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, items: torch.Tensor, corrects: torch.Tensor, scalars: torch.Tensor | None = None):
        bsz, seqlen = items.size()
        device = items.device
        positions = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        positions = positions.clamp(max=self.max_seq_len - 1)

        # Interaction embedding uses previous response (causal).
        prev_correct = torch.cat(
            [torch.zeros(bsz, 1, device=device, dtype=corrects.dtype), corrects[:, :-1]],
            dim=1,
        )
        inter_ids = items + (prev_correct >= 0.5).long() * self.n_item_slots
        inter_ids = torch.where(items == 0, torch.zeros_like(inter_ids), inter_ids)

        q = self.item_emb(items) + self.pos_emb(positions)
        kv = self.interaction_emb(inter_ids) + self.pos_emb(positions)
        q = self.dropout(q)
        kv = self.dropout(kv)

        # Causal mask: position i attends to j <= i
        causal = torch.triu(torch.ones(seqlen, seqlen, device=device, dtype=torch.bool), diagonal=1)
        key_padding = items == 0
        attn_out, _ = self.attn(q, kv, kv, attn_mask=causal, key_padding_mask=key_padding, need_weights=False)
        x = self.layer_norm1(q + self.dropout(attn_out))
        x = self.layer_norm2(x + self.dropout(self.ffn(x)))

        if self.use_scalar and scalars is not None:
            logits = self.pred(torch.cat([x, scalars.unsqueeze(-1)], dim=-1)).squeeze(-1)
        else:
            logits = self.pred(x).squeeze(-1)
        return logits

def build_model(backbone: str, n_items: int, use_scalar: bool, cfg: dict) -> nn.Module:
    tcfg = cfg["train"]
    if backbone == "GRU":
        b = cfg["backbones"]["GRU"]
        return CleanGRU(n_items, dim=b["emb_dim"], use_scalar=use_scalar)
    if backbone == "SAKT":
        b = cfg["backbones"]["SAKT"]
        return SAKT(
            n_items,
            d_model=b["d_model"],
            n_heads=b["n_heads"],
            dropout=b["dropout"],
            max_seq_len=tcfg["max_seq_len"],
            use_scalar=use_scalar,
        )
    raise ValueError(f"Unknown backbone: {backbone}")


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
            "eval_mask": torch.tensor(s.get("eval_mask", s["mask"]), dtype=torch.bool),
        }


def collate_sequences(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(b["items"].size(0) for b in batch)
    items, corrects, scalars, masks, eval_masks = [], [], [], [], []
    for b in batch:
        pad = max_len - b["items"].size(0)
        items.append(torch.nn.functional.pad(b["items"], (0, pad)))
        corrects.append(torch.nn.functional.pad(b["corrects"], (0, pad), value=-1))
        scalars.append(torch.nn.functional.pad(b["scalars"], (0, pad)))
        masks.append(torch.nn.functional.pad(b["mask"], (0, pad)))
        eval_masks.append(torch.nn.functional.pad(b["eval_mask"], (0, pad)))
    return {
        "items": torch.stack(items),
        "corrects": torch.stack(corrects),
        "scalars": torch.stack(scalars),
        "mask": torch.stack(masks),
        "eval_mask": torch.stack(eval_masks),
    }


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def metrics_from_arrays(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(y) == 0:
        return {
            "log_loss": float("nan"),
            "brier": float("nan"),
            "auc": float("nan"),
            "accuracy": float("nan"),
            "ece": float("nan"),
            "n_predictions": 0,
        }
    p = np.clip(p, 1e-6, 1 - 1e-6)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean((p >= 0.5) == y))
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {
        "log_loss": log_loss,
        "brier": brier,
        "auc": auc,
        "accuracy": acc,
        "ece": _ece(y, p),
        "n_predictions": int(len(y)),
    }


def build_item_folds(dataset: str, cfg: dict) -> pd.DataFrame:
    """Deterministic 5-fold item holdout balanced by training-response-count bins."""
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = FOLD_DIR / f"{dataset}_item_folds_seed{cfg['item_fold_seed']}.parquet"
    meta_path = FOLD_DIR / f"{dataset}_item_folds_seed{cfg['item_fold_seed']}.meta.json"
    if path.exists() and meta_path.exists():
        return pd.read_parquet(path)

    scoreable = load_scoreable_universe(dataset)
    item_ids = sorted(scoreable["item_id_hash"].tolist())
    ix = _load_interactions_cached(dataset)
    train = ix[ix["split_assignment"] == "train"]
    counts = train.groupby("item_id_hash").size().reindex(item_ids).fillna(0).astype(int)
    count_df = pd.DataFrame({"item_id_hash": item_ids, "train_n": counts.values})

    # Bin by training response count (quantile bins on positive counts; zeros separate).
    nonzero = count_df[count_df["train_n"] > 0].copy()
    zero = count_df[count_df["train_n"] == 0].copy()
    if len(nonzero):
        try:
            nonzero["bin"] = pd.qcut(nonzero["train_n"], q=min(10, len(nonzero)), duplicates="drop")
        except ValueError:
            nonzero["bin"] = 0
    else:
        nonzero["bin"] = pd.Series(dtype=object)
    zero["bin"] = "zero"

    fold_seed = cfg["item_fold_seed"]
    n_folds = cfg["n_item_folds"]
    # Stratify by response-count bin, then assign folds with a global round-robin
    # over the deterministically ordered stratified list for balanced fold sizes.
    pieces = []
    for bin_name, g in pd.concat([nonzero, zero], ignore_index=True).groupby("bin", observed=False):
        g = g.copy()
        keys = g["item_id_hash"].astype(str) + f"|{fold_seed}|{bin_name}"
        ranks = keys.map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:12], 16))
        g = g.assign(_bin=str(bin_name), _rank=ranks).sort_values(["_rank", "item_id_hash"])
        pieces.append(g)
    ordered = pd.concat(pieces, ignore_index=True)
    ordered["item_fold"] = np.arange(len(ordered)) % n_folds
    folds = ordered[["item_id_hash", "train_n", "item_fold"]].sort_values("item_id_hash").reset_index(drop=True)
    folds["dataset"] = dataset
    folds["item_fold_seed"] = fold_seed

    # Assertions: exhaustive, non-overlapping
    assert set(folds["item_id_hash"]) == set(item_ids)
    assert folds["item_id_hash"].duplicated().sum() == 0
    assert set(folds["item_fold"]) == set(range(n_folds))

    folds.to_parquet(path, index=False)
    meta = {
        "dataset": dataset,
        "item_fold_seed": fold_seed,
        "n_folds": n_folds,
        "n_items": len(folds),
        "fold_counts": folds["item_fold"].value_counts().sort_index().to_dict(),
        "file_sha256": sha256_file(path),
        "item_list_sha256": sha256_text(",".join(folds["item_id_hash"].tolist())),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folds


def build_scalar_maps_coldstart(
    dataset: str,
    seen_items: set[str],
    all_items: set[str],
    cfg: dict,
) -> dict[str, dict[str, float]]:
    """Scalars for cold-start: char-length scaling from seen items only; no train-emp for targets."""
    llm = pd.read_parquet(LLM_FEATURES)
    llm = llm[llm["dataset"] == dataset]
    mini = llm[llm["model_identifier"] == "gpt-4o-mini"].set_index("item_id_hash")["scalar_difficulty"]
    g54 = llm[llm["model_identifier"] == "gpt-5.4"].set_index("item_id_hash")["scalar_difficulty"]

    refs = pd.read_csv(REF_TABLE)
    train_ref = refs[(refs["dataset"] == dataset) & (refs["reference_scope"] == "deployable_train")]
    global_mean = float(train_ref["smoothed_error_beta_1_1"].mean())

    surface = pd.read_csv(SURFACE_TABLE)
    surface = surface[(surface["dataset"] == dataset) & (surface["item_id_hash"].isin(all_items))]
    char = surface.set_index("item_id_hash")["char_length"].astype(float)
    seen_char = char.reindex(list(seen_items)).dropna()
    char_min = float(seen_char.min()) if len(seen_char) else 0.0
    char_max = float(seen_char.max()) if len(seen_char) else 1.0
    if np.isfinite(char_min) and np.isfinite(char_max) and char_max > char_min:
        char_norm = ((char - char_min) / (char_max - char_min)).clip(0, 1).to_dict()
    else:
        char_norm = {item: 0.5 for item in all_items}

    # Random: preserve Mini score distribution, destroy item mapping via permutation.
    rng = np.random.default_rng(cfg["train"]["mask_seed"])
    ordered_items = sorted(all_items)
    mini_vals = mini.reindex(ordered_items)
    # Fill rare missing Mini scores with global mean so permutation length matches.
    filled = mini_vals.fillna(float(mini_vals.mean()) if mini_vals.notna().any() else 0.5).values.astype(float)
    shuffled = rng.permutation(filled)
    random_draw = {item: float(v) for item, v in zip(ordered_items, shuffled)}

    return {
        "gpt-4o-mini": mini.to_dict(),
        "gpt-5.4": g54.to_dict(),
        "random_matched": random_draw,
        "char_length_norm": {k: float(v) for k, v in char_norm.items()},
        "global_train_mean": global_mean,
    }


def build_scalar_maps_limited(dataset: str, bundle_items: set[str], exposure, cfg: dict) -> dict[str, dict[str, float]]:
    """Same scientific role as RQ4 scalars; local copy to avoid coupling to limited_kt_common internals."""
    from limited_kt_common import build_scalar_maps

    # Reuse RQ4 scalar builder for limited-response SAKT.
    limited_cfg = {
        "train_emp_prior": cfg["train_emp_prior"],
        "model": {"mask_seed": cfg["train"]["mask_seed"]},
    }
    return build_scalar_maps(dataset, bundle_items, exposure, limited_cfg)


@dataclass
class ColdStartBundle:
    dataset: str
    backbone_ready: bool
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]
    unk_idx: int
    seen_items: set[str]
    target_items: set[str]
    train_seqs: list[dict]
    val_seqs: list[dict]
    test_first_seqs: list[dict]
    test_all_seqs: list[dict]
    scalar_maps: dict[str, dict[str, float]]
    split_hash: str
    universe_hash: str
    target_list_hash: str
    item_fold: int
    item_fold_seed: int
    gate_assertions: dict[str, Any]


def _apply_item_dropout(
    item_indices: list[int],
    unk_idx: int,
    dropout_p: float,
    seed: int,
    student_key: str,
) -> list[int]:
    """Deterministic per-position item-ID dropout to UNK on seen items."""
    out = []
    for pos, idx in enumerate(item_indices):
        if idx == 0 or idx == unk_idx:
            out.append(idx)
            continue
        key = f"{student_key}|{pos}|{idx}|{seed}|dropout"
        h = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
        if (h % 10_000) / 10_000.0 < dropout_p:
            out.append(unk_idx)
        else:
            out.append(idx)
    return out


def load_coldstart_bundle(dataset: str, item_fold: int, cfg: dict) -> ColdStartBundle:
    folds = build_item_folds(dataset, cfg)
    target_items = set(folds.loc[folds["item_fold"] == item_fold, "item_id_hash"])
    all_items = sorted(folds["item_id_hash"].tolist())
    seen_items = [i for i in all_items if i not in target_items]
    seen_set = set(seen_items)
    target_set = set(target_items)

    # Index map: 0=pad, 1=UNK_ITEM, then seen items only (targets never get private embeddings).
    item_to_idx = {h: i + 2 for i, h in enumerate(seen_items)}
    unk_idx = 1
    idx_to_item = {i: h for h, i in item_to_idx.items()}
    idx_to_item[unk_idx] = "UNK_ITEM"

    ix = _load_interactions_cached(dataset).copy()
    # Remove ALL training interactions on target items.
    train_ix = ix[ix["split_assignment"] == "train"]
    train_ix = train_ix[~train_ix["item_id_hash"].isin(target_set)]
    # Val: exclude target-item outcomes from early stopping.
    val_ix = ix[ix["split_assignment"] == "val"]
    val_ix = val_ix[~val_ix["item_id_hash"].isin(target_set)]
    test_ix = ix[ix["split_assignment"] == "test"]

    # Gate: zero training interactions on targets
    n_target_train = int(((ix["split_assignment"] == "train") & (ix["item_id_hash"].isin(target_set))).sum())
    # After removal:
    n_target_train_kept = int(train_ix["item_id_hash"].isin(target_set).sum())

    scalar_maps = build_scalar_maps_coldstart(dataset, seen_set, set(all_items), cfg)
    max_len = cfg["train"]["max_seq_len"]
    dropout_p = float(cfg["item_id_dropout"])
    drop_seed = int(cfg["item_fold_seed"])

    def map_item(item_id: str) -> int:
        if item_id in target_set:
            return unk_idx
        return item_to_idx[item_id]

    def build_train_val(df: pd.DataFrame, apply_dropout: bool) -> list[dict]:
        seqs = []
        for sid, g in df.groupby("student_id_hash"):
            g = g.sort_values("sequence_index").head(max_len)
            raw_ids = g["item_id_hash"].tolist()
            idxs = [map_item(i) for i in raw_ids]
            if apply_dropout:
                idxs = _apply_item_dropout(idxs, unk_idx, dropout_p, drop_seed, str(sid))
            corrects = g["correct"].astype(float).tolist()
            seqs.append(
                {
                    "items": idxs,
                    "corrects": corrects,
                    "scalars": [0.0] * len(idxs),
                    "mask": [True] * len(idxs),
                    "eval_mask": [True] * len(idxs),
                    "item_hashes": raw_ids,
                    "student_id": str(sid),
                }
            )
        return seqs

    train_seqs = build_train_val(train_ix, apply_dropout=True)
    val_seqs = build_train_val(val_ix, apply_dropout=True)  # identical dropout policy

    # Test sequences: keep full history; evaluate only target positions.
    test_first_seqs = []
    test_all_seqs = []
    for sid, g in test_ix.groupby("student_id_hash"):
        g = g.sort_values("sequence_index").head(max_len)
        raw_ids = g["item_id_hash"].tolist()
        idxs = [map_item(i) for i in raw_ids]
        corrects = g["correct"].astype(float).tolist()
        is_target = [i in target_set for i in raw_ids]
        if not any(is_target):
            continue
        # First attempt per learner x target item
        seen_pair = set()
        first_mask = []
        for item_id, tgt in zip(raw_ids, is_target):
            if tgt and item_id not in seen_pair:
                first_mask.append(True)
                seen_pair.add(item_id)
            else:
                first_mask.append(False)
        all_mask = is_target
        base = {
            "items": idxs,
            "corrects": corrects,
            "scalars": [0.0] * len(idxs),
            "mask": [True] * len(idxs),
            "item_hashes": raw_ids,
            "student_id": str(sid),
        }
        if any(first_mask):
            test_first_seqs.append({**base, "eval_mask": first_mask})
        if any(all_mask):
            test_all_seqs.append({**base, "eval_mask": all_mask})

    gate_assertions = {
        "n_target_train_before_removal": n_target_train,
        "n_target_train_after_removal": n_target_train_kept,
        "zero_target_train_interactions": n_target_train_kept == 0,
        "n_seen_items": len(seen_items),
        "n_target_items": len(target_set),
        "unk_idx": unk_idx,
        "targets_have_no_private_embedding_slots": True,
        "val_excludes_target_items": True,
        "item_id_dropout": dropout_p,
        "identical_dropout_policy_all_conditions": True,
    }

    return ColdStartBundle(
        dataset=dataset,
        backbone_ready=True,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        unk_idx=unk_idx,
        seen_items=seen_set,
        target_items=target_set,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        test_first_seqs=test_first_seqs,
        test_all_seqs=test_all_seqs,
        scalar_maps=scalar_maps,
        split_hash=sha256_file(PROCESSED / dataset / "splits.parquet"),
        universe_hash=sha256_text(",".join(all_items)),
        target_list_hash=sha256_text(",".join(sorted(target_set))),
        item_fold=item_fold,
        item_fold_seed=cfg["item_fold_seed"],
        gate_assertions=gate_assertions,
    )


def _attach_scalars(seqs: list[dict], smap: dict[str, float], default: float) -> list[dict]:
    out = []
    for s in seqs:
        hashes = s.get("item_hashes")
        if hashes is None:
            # limited RQ4-style: recover from idx if needed — coldstart always has hashes
            scalars = list(s["scalars"])
        else:
            scalars = [float(smap.get(h, default)) for h in hashes]
        out.append({**s, "scalars": scalars})
    return out


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_model(
    model: nn.Module,
    train_seqs: list[dict],
    val_seqs: list[dict],
    use_scalar: bool,
    cfg: dict,
    device: torch.device,
) -> tuple[nn.Module, float, int]:
    tcfg = cfg["train"]
    opt = optim.Adam(model.parameters(), lr=tcfg["lr"])
    train_loader = (
        DataLoader(
            KTSequenceDataset(train_seqs),
            batch_size=tcfg["batch_size"],
            shuffle=True,
            collate_fn=collate_sequences,
        )
        if train_seqs
        else None
    )
    val_loader = (
        DataLoader(
            KTSequenceDataset(val_seqs),
            batch_size=tcfg["batch_size"],
            shuffle=False,
            collate_fn=collate_sequences,
        )
        if val_seqs
        else None
    )

    best_val = float("inf")
    best_state = None
    patience = 0
    epoch = -1
    model.to(device)
    if train_loader and val_loader and len(train_seqs) > 0:
        for epoch in range(tcfg["max_epochs"]):
            model.train()
            for batch in train_loader:
                batch = _batch_to_device(batch, device)
                logits = model(batch["items"], batch["corrects"], batch["scalars"] if use_scalar else None)
                mask = batch["mask"] & (batch["corrects"] >= 0)
                if mask.sum() == 0:
                    continue
                loss = nn.functional.binary_cross_entropy_with_logits(logits[mask], batch["corrects"][mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_loss = _eval_loss(model, val_loader, use_scalar, device)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience > tcfg["early_stop_patience"]:
                    break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_val, epoch + 1


def _eval_loss(model, loader, use_scalar, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            logits = model(batch["items"], batch["corrects"], batch["scalars"] if use_scalar else None)
            mask = batch["mask"] & (batch["corrects"] >= 0)
            if mask.sum() == 0:
                continue
            losses.append(
                nn.functional.binary_cross_entropy_with_logits(logits[mask], batch["corrects"][mask]).item()
            )
    return float(np.mean(losses)) if losses else float("inf")


def collect_predictions(model, seqs: list[dict], use_scalar: bool, cfg: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if not seqs:
        return np.asarray([]), np.asarray([])
    loader = DataLoader(
        KTSequenceDataset(seqs),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_sequences,
    )
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            logits = model(batch["items"], batch["corrects"], batch["scalars"] if use_scalar else None)
            mask = batch["eval_mask"] & (batch["corrects"] >= 0)
            if mask.sum() == 0:
                continue
            prob = torch.sigmoid(logits[mask]).detach().cpu().numpy()
            ys.extend(batch["corrects"][mask].detach().cpu().numpy().tolist())
            ps.extend(prob.tolist())
    return np.asarray(ys, dtype=float), np.asarray(ps, dtype=float)


def run_coldstart_cell(
    bundle: ColdStartBundle,
    backbone: str,
    condition: str,
    seed: int,
    cfg: dict,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device()
    cond = cfg["conditions"][condition]
    use_scalar = cond["use_scalar"]
    src = cond["scalar_source"]
    default = float(bundle.scalar_maps.get("global_train_mean", 0.5))
    smap = bundle.scalar_maps.get(src, {}) if src else {}

    train_seqs = _attach_scalars(bundle.train_seqs, smap, default)
    val_seqs = _attach_scalars(bundle.val_seqs, smap, default)
    test_first = _attach_scalars(bundle.test_first_seqs, smap, default)
    test_all = _attach_scalars(bundle.test_all_seqs, smap, default)

    n_items = max(bundle.item_to_idx.values(), default=1)  # includes UNK at 1; embedding size = max_idx
    # Embedding table indexes 0..n_items inclusive where n_items = max index.
    model = build_model(backbone, n_items, use_scalar, cfg)
    model, best_val, best_epoch = train_model(model, train_seqs, val_seqs, use_scalar, cfg, device)

    y1, p1 = collect_predictions(model, test_first, use_scalar, cfg, device)
    ya, pa = collect_predictions(model, test_all, use_scalar, cfg, device)
    m_first = metrics_from_arrays(y1, p1)
    m_all = metrics_from_arrays(ya, pa)

    return {
        "backbone": backbone,
        "condition": condition,
        "seed": seed,
        "item_fold": bundle.item_fold,
        "item_fold_seed": bundle.item_fold_seed,
        "mask_dropout_seed": cfg["item_fold_seed"],
        "best_val_log_loss": best_val,
        "best_epoch": best_epoch,
        "n_parameters": model.count_parameters(),
        "split_hash": bundle.split_hash,
        "universe_hash": bundle.universe_hash,
        "target_item_list_hash": bundle.target_list_hash,
        "gate_assertions": bundle.gate_assertions,
        "primary_y": y1,
        "primary_p": p1,
        "secondary_y": ya,
        "secondary_p": pa,
        "primary_metrics": m_first,
        "secondary_metrics": m_all,
    }


@dataclass
class LimitedBundle:
    dataset: str
    exposure: Any
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]
    train_seqs: list[dict]
    val_seqs: list[dict]
    test_seqs: list[dict]
    scalar_maps: dict[str, dict[str, float]]
    split_hash: str
    universe_hash: str
    mask_hash: str


def load_limited_bundle(dataset: str, exposure, cfg: dict) -> LimitedBundle:
    """Learner-split limited-response bundle using existing deterministic masks."""
    scoreable = load_scoreable_universe(dataset)
    item_ids = sorted(scoreable["item_id_hash"].tolist())
    item_to_idx = {h: i + 1 for i, h in enumerate(item_ids)}
    idx_to_item = {i: h for h, i in item_to_idx.items()}

    # Use limited_kt config shape for mask builder
    mask_cfg = {"model": {"mask_seed": cfg["train"]["mask_seed"]}}
    mask_df, mask_hash = build_exposure_mask(dataset, exposure, mask_cfg)
    kept_pairs = set(zip(mask_df["student_id_hash"], mask_df["item_id_hash"]))

    ix = _load_interactions_cached(dataset).copy()
    if exposure != "warm":
        ix_train = ix[ix["split_assignment"] == "train"].copy()
        ix_train["_key"] = list(zip(ix_train["student_id_hash"], ix_train["item_id_hash"]))
        ix_train = ix_train[ix_train["_key"].isin(kept_pairs)].drop(columns=["_key"])
        other = ix[ix["split_assignment"] != "train"]
        ix = pd.concat([ix_train, other], ignore_index=True)

    scalar_maps = build_scalar_maps_limited(dataset, set(item_ids), exposure, cfg)
    max_len = cfg["train"]["max_seq_len"]

    def build(df: pd.DataFrame) -> list[dict]:
        seqs = []
        for sid, g in df.groupby("student_id_hash"):
            g = g.sort_values("sequence_index").head(max_len)
            raw = g["item_id_hash"].tolist()
            idxs = [item_to_idx[i] for i in raw]
            corrects = g["correct"].astype(float).tolist()
            seqs.append(
                {
                    "items": idxs,
                    "corrects": corrects,
                    "scalars": [0.0] * len(idxs),
                    "mask": [True] * len(idxs),
                    "eval_mask": [True] * len(idxs),
                    "item_hashes": raw,
                    "student_id": str(sid),
                }
            )
        return seqs

    return LimitedBundle(
        dataset=dataset,
        exposure=exposure,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        train_seqs=build(ix[ix["split_assignment"] == "train"]),
        val_seqs=build(ix[ix["split_assignment"] == "val"]),
        test_seqs=build(ix[ix["split_assignment"] == "test"]),
        scalar_maps=scalar_maps,
        split_hash=sha256_file(PROCESSED / dataset / "splits.parquet"),
        universe_hash=sha256_text(",".join(item_ids)),
        mask_hash=mask_hash,
    )


def run_limited_cell(
    bundle: LimitedBundle,
    backbone: str,
    condition: str,
    seed: int,
    cfg: dict,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device()
    cond = cfg["conditions"][condition]
    use_scalar = cond["use_scalar"]
    src = cond["scalar_source"]
    default = float(bundle.scalar_maps.get("global_train_mean", 0.5))
    smap = bundle.scalar_maps.get(src, {}) if src else {}

    train_seqs = _attach_scalars(bundle.train_seqs, smap, default)
    val_seqs = _attach_scalars(bundle.val_seqs, smap, default)
    test_seqs = _attach_scalars(bundle.test_seqs, smap, default)

    n_items = len(bundle.item_to_idx)
    model = build_model(backbone, n_items, use_scalar, cfg)
    model, best_val, best_epoch = train_model(model, train_seqs, val_seqs, use_scalar, cfg, device)
    y, p = collect_predictions(model, test_seqs, use_scalar, cfg, device)
    metrics = metrics_from_arrays(y, p)
    return {
        "backbone": backbone,
        "condition": condition,
        "seed": seed,
        "exposure": bundle.exposure,
        "best_val_log_loss": best_val,
        "best_epoch": best_epoch,
        "n_parameters": model.count_parameters(),
        "split_hash": bundle.split_hash,
        "universe_hash": bundle.universe_hash,
        "mask_hash": bundle.mask_hash,
        **metrics,
    }


def append_registry(path: Path, row: dict) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: v for k, v in row.items() if not isinstance(v, (np.ndarray, dict, list))}
    if "gate_assertions" in row and isinstance(row["gate_assertions"], dict):
        flat["gate_assertions_json"] = json.dumps(row["gate_assertions"], sort_keys=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            new_df = pd.DataFrame([flat])
            if path.exists() and path.stat().st_size > 0:
                old = pd.read_csv(path)
                if "run_id" in old.columns and "run_id" in flat:
                    old = old[old["run_id"] != flat["run_id"]]
                new_df = pd.concat([old, new_df], ignore_index=True)
            new_df.to_csv(path, index=False)
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
