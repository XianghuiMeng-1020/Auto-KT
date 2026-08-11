"""Leakage-safe 1PL Rasch estimation via joint maximum likelihood (alternating optimization)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RaschResult:
    item_difficulties: pd.DataFrame
    student_abilities: pd.DataFrame
    convergence: dict
    orientation: str = "higher_is_harder"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def fit_rasch_1pl(
    responses: pd.DataFrame,
    *,
    max_iter: int = 80,
    tol: float = 1e-4,
    seed: int = 2024,
    min_responses: int = 5,
) -> RaschResult:
    """Fit 1PL Rasch: P(correct) = sigmoid(theta_s - beta_i). Larger beta => harder item."""
    rng = np.random.default_rng(seed)
    sub = responses[["student_id_hash", "item_id_hash", "correct"]].dropna().copy()
    sub["correct"] = sub["correct"].astype(int)

    students, s_idx = np.unique(sub["student_id_hash"].values, return_inverse=True)
    items, i_idx = np.unique(sub["item_id_hash"].values, return_inverse=True)
    y = sub["correct"].values.astype(float)
    n_obs = len(y)

    item_counts = np.bincount(i_idx, minlength=len(items))
    item_correct = np.bincount(i_idx, weights=y, minlength=len(items))
    item_error = item_correct / np.maximum(item_counts, 1)

    perfect = (item_counts >= min_responses) & (item_error >= 1.0 - 1e-9)
    zero = (item_counts >= min_responses) & (item_error <= 1e-9)
    extreme = perfect | zero
    identifiable = (item_counts >= min_responses) & ~extreme

    theta = rng.normal(0, 0.01, len(students))
    beta = np.zeros(len(items))
  # initialize harder items higher
    beta = np.log((1 - item_error + 1e-3) / (item_error + 1e-3))
    beta[~identifiable] = np.nan

    prev_beta = beta.copy()
    converged = False
    max_delta = float("inf")
    for it in range(max_iter):
        eta = theta[s_idx] - beta[i_idx]
        p = _sigmoid(eta)
        grad_s = np.bincount(s_idx, weights=(y - p), minlength=len(students))
        hess_s = np.bincount(s_idx, weights=(p * (1 - p)), minlength=len(students)) + 1e-6
        theta += grad_s / hess_s
        grad_i = np.bincount(i_idx, weights=(p - y), minlength=len(items))
        hess_i = np.bincount(i_idx, weights=(p * (1 - p)), minlength=len(items)) + 1e-6
        beta += grad_i / hess_i
        beta[~identifiable] = np.nan
        if identifiable.any():
            beta[identifiable] -= np.nanmean(beta[identifiable])
            max_delta = float(np.max(np.abs(beta[identifiable] - prev_beta[identifiable])))
        else:
            max_delta = float("nan")
        if it > 0 and max_delta < tol:
            converged = True
            break
        prev_beta = beta.copy()

    # approximate SE from Fisher information diagonal
    se = np.full(len(items), np.nan)
    for i in range(len(items)):
        if not identifiable[i]:
            continue
        mask = i_idx == i
        p = _sigmoid(theta[s_idx[mask]] - beta[i])
        info = np.sum(p * (1 - p))
        se[i] = 1.0 / np.sqrt(info + 1e-6)

    item_df = pd.DataFrame({
        "item_id_hash": items,
        "rasch_difficulty": beta,
        "rasch_se": se,
        "n_responses": item_counts,
        "identifiable": identifiable,
        "perfect_score": perfect,
        "zero_score": zero,
    })
    student_df = pd.DataFrame({
        "student_id_hash": students,
        "rasch_ability": theta,
    })
    convergence = {
        "converged": converged,
        "iterations": it + 1,
        "max_delta": max_delta,
        "n_observations": n_obs,
        "n_students": len(students),
        "n_items": len(items),
        "n_identifiable_items": int(identifiable.sum()),
        "n_extreme_items": int(extreme.sum()),
    }
    return RaschResult(item_difficulties=item_df, student_abilities=student_df, convergence=convergence)
