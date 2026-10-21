"""TREC-style linear-gain nDCG, MRR, recall, and paired query bootstrap."""
from __future__ import annotations
import math
import numpy as np

def ndcg(ranking, qrels, k=10):
    gains = [max(0, qrels.get(doc, 0)) for doc in ranking[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(g / math.log2(i + 2) for i, g in
                enumerate(sorted((max(0, r) for r in qrels.values()), reverse=True)[:k]))
    return dcg / ideal if ideal else 0.0

def evaluate(ranking, qrels):
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate document IDs")
    relevant = {d for d, r in qrels.items() if r > 0}
    return {
        "ndcg10": ndcg(ranking, qrels),
        "mrr10": next((1 / (i + 1) for i, d in enumerate(ranking[:10]) if d in relevant), 0.0),
        "recall50": len(set(ranking[:50]) & relevant) / len(relevant) if relevant else 0.0,
        "recall100": len(set(ranking[:100]) & relevant) / len(relevant) if relevant else 0.0,
    }

def bootstrap(values, seed=20261021, samples=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "low": float(np.quantile(means, .025)),
            "high": float(np.quantile(means, .975))}

def paired_delta(left, right, seed=20261021, samples=5000):
    if len(left) != len(right) or not len(left):
        raise ValueError("Paired observations required")
    return bootstrap(np.asarray(left) - np.asarray(right), seed, samples)

def aggregate(rows, ids):
    chosen = [rows[q] for q in ids]
    result = {k: float(np.mean([r[k] for r in chosen]))
              for k in ("ndcg10", "mrr10", "recall50", "recall100")}
    ms = [r["latency_ms"] for r in chosen]
    result.update(n=len(chosen), latency_p50_ms=float(np.median(ms)),
                  latency_p95_ms=float(np.quantile(ms, .95)),
                  mean_latency_ms=float(np.mean(ms)))
    result["ndcg10_ci"] = bootstrap([r["ndcg10"] for r in chosen])
    return result
