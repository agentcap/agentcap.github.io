"""Full-corpus BM25 + E5, RRF, and real GPU cross-encoder measurements."""
from __future__ import annotations
import argparse, gc, json, platform, re, time
from pathlib import Path
import numpy as np
import torch
from nltk.stem.snowball import SnowballStemmer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from .data import dump, fingerprint, load_config, load_dataset
from .metrics import evaluate

STEM = SnowballStemmer("english")
TOKEN = re.compile(r"\b\w+\b")

def tokenize(text):
    return [STEM.stem(t) for t in TOKEN.findall(text.lower())]

def top(scores, ids, k):
    # Stable original corpus ID order breaks exact score ties.
    return [ids[i] for i in np.argsort(-np.asarray(scores), kind="stable")[:k]]

def rrf(rankings, k=60, depth=100):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))[:depth], scores

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed(call):
    synchronize()
    start = time.perf_counter()
    value = call()
    synchronize()
    return value, (time.perf_counter() - start) * 1000

def model_revision(model):
    try:
        obj = model[0].auto_model if isinstance(model, SentenceTransformer) else model.model
        return getattr(obj.config, "_commit_hash", None)
    except (AttributeError, TypeError):
        return None

def prepare(name, config, device):
    corpus, queries, qrels, manifest = load_dataset(name, config)
    out = Path("results") / name
    retrieval_file = out / "retrieval.json"
    if retrieval_file.exists():
        print(f"{name}: reuse frozen retrieval", flush=True)
        if not (out / "passages.json").exists():
            revisions = json.loads(Path("models.lock.json").read_text()) if Path("models.lock.json").exists() else {}
            tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2",
                revision=revisions.get("cross-encoder/ms-marco-MiniLM-L6-v2"))
            clipped = {d: tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)
                       [:config["rerank_passage_tokens"]]) for d,text in corpus.items()}
            dump(out/"passages.json",clipped)
        return corpus, queries, qrels, manifest, json.loads(retrieval_file.read_text())
    ids = sorted(corpus)
    texts = [corpus[d] for d in ids]
    revisions = json.loads(Path("models.lock.json").read_text()) if Path("models.lock.json").exists() else {}
    model = SentenceTransformer(config["dense_model"], device=device, revision=revisions.get(config["dense_model"]))
    model.max_seq_length = config["dense_max_tokens"]
    print(f"{name}: index {len(ids)} full-corpus documents", flush=True)
    start = time.perf_counter()
    bm25 = BM25Okapi([tokenize(t) for t in texts], k1=1.2, b=.75)
    bm25_index_seconds = time.perf_counter() - start
    cache = Path("cache") / f"{name}-{fingerprint([config['dense_model'], ids, texts, config['dense_max_tokens']])}.npy"
    cache.parent.mkdir(exist_ok=True)
    start = time.perf_counter()
    if cache.exists():
        embeddings = np.load(cache)
    else:
        embeddings = model.encode(["passage: " + t for t in texts], batch_size=64,
                                  normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
        np.save(cache, embeddings)
    dense_index_seconds = time.perf_counter() - start
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2",
        revision=revisions.get("cross-encoder/ms-marco-MiniLM-L6-v2"))
    # All rerankers receive the same decoded, bounded query and passage text.
    clipped = {d: tokenizer.decode(tokenizer.encode(corpus[d], add_special_tokens=False)
                                  [:config["rerank_passage_tokens"]]) for d in ids}
    dump(out / "passages.json", clipped)
    all_qids = manifest["calibration_queries"] + manifest["test_queries"]
    model.encode(["query: warmup"], normalize_embeddings=True)
    rows = {}
    for i, q in enumerate(all_qids):
        query = queries[q]
        bm_scores, bm_ms = timed(lambda: bm25.get_scores(tokenize(query)))
        bm = top(bm_scores, ids, config["retrieval_depth"])
        query_vec, enc_ms = timed(lambda: model.encode(["query: " + query],
                           normalize_embeddings=True, convert_to_numpy=True)[0])
        dense_scores, search_ms = timed(lambda: embeddings @ query_vec)
        dense = top(dense_scores, ids, config["retrieval_depth"])
        # Remove identical-ID query/document matches, as in BEIR evaluation.
        bm, dense = [d for d in bm if d != q], [d for d in dense if d != q]
        (hybrid, fused), fusion_ms = timed(lambda: rrf([bm, dense], config["rrf_k"], config["retrieval_depth"]))
        # Retrieval is measured sequentially here, including both retrievers.
        hybrid_ms = bm_ms + enc_ms + search_ms + fusion_ms
        metrics = {
            "bm25": {**evaluate(bm, qrels[q]), "latency_ms": bm_ms},
            "dense": {**evaluate(dense, qrels[q]), "latency_ms": enc_ms + search_ms},
            "hybrid": {**evaluate(hybrid, qrels[q]), "latency_ms": hybrid_ms},
        }
        overlap = len(set(bm[:10]) & set(dense[:10])) / len(set(bm[:10]) | set(dense[:10]))
        margin = (fused[hybrid[0]] - fused[hybrid[1]]) / max(fused[hybrid[0]], 1e-9)
        rows[q] = {"query": query, "clipped_query": tokenizer.decode(tokenizer.encode(query, add_special_tokens=False)[:config["rerank_query_tokens"]]),
                   "bm25": bm, "dense": dense, "hybrid": hybrid, "metrics": metrics,
                   "features": [overlap, margin, len(tokenize(query)),
                                float(bm_scores.max()), float(dense_scores.max()),
                                float(np.std(dense_scores))]}
        if (i + 1) % 100 == 0:
            print(f"{name}: retrieval {i+1}/{len(all_qids)}", flush=True)
    dump(retrieval_file, rows)
    dump(out / "index.json", {"bm25_index_seconds": bm25_index_seconds,
        "dense_index_or_cache_load_seconds": dense_index_seconds, "dense_model": config["dense_model"],
        "dense_revision": model_revision(model) or revisions.get(config["dense_model"]), "bm25": {"k1": 1.2, "b": .75,
        "tokenizer": "Unicode word regex, lowercase, English Snowball; no stopword removal"},
        "dense_prompt": {"query": "query: ", "passage": "passage: "},
        "latency": "Warm serial query execution, excluding index construction and disk output."})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return corpus, queries, qrels, manifest, rows

def rerank(name, config, qrels, retrieval, device):
    out = Path("results") / name
    passages = json.loads((out / "passages.json").read_text())
    for model_id in config["cross_encoders"]:
        short = "ce_" + ("tinybert" if "TinyBERT" in model_id else "minilm")
        path = out / f"{short}.json"
        if path.exists():
            print(f"{name}: reuse {short}", flush=True)
            continue
        print(f"{name}: scoring top-{config['rerank_depth']} with {model_id}", flush=True)
        revisions = json.loads(Path("models.lock.json").read_text()) if Path("models.lock.json").exists() else {}
        model = CrossEncoder(model_id, device=device, max_length=512, revision=revisions.get(model_id))
        model.predict([["warmup", "This is a warmup passage."]], show_progress_bar=False)
        rows = {}
        for i, (q, row) in enumerate(retrieval.items()):
            candidates = row["hybrid"][:config["rerank_depth"]]
            scores, ms = timed(lambda: model.predict(
                [[row["clipped_query"], passages[d]] for d in candidates],
                batch_size=32, show_progress_bar=False))
            reordered = sorted(zip(candidates, map(float, scores)), key=lambda p: -p[1])
            ranking = [d for d, _ in reordered] + row["hybrid"][len(candidates):]
            rows[q] = {"ranking": ranking, "scores": dict(reordered),
                       **evaluate(ranking, qrels[q]), "rerank_ms": ms,
                       "latency_ms": row["metrics"]["hybrid"]["latency_ms"] + ms}
            if (i + 1) % 100 == 0:
                print(f"{name} {short}: {i+1}/{len(retrieval)}", flush=True)
        dump(path, rows)
        dump(out / f"{short}.model.json", {"model": model_id, "revision": model_revision(model) or revisions.get(model_id),
                                         "max_length": 512, "batch_size": 32, "device": device})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", choices=["scifact", "nfcorpus", "all"], default="all")
    opts = args.parse_args()
    config = load_config()
    torch.set_num_threads(6)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dump("results/environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_cpu_threads": torch.get_num_threads(), "config": config})
    for name in config["datasets"] if opts.dataset == "all" else [opts.dataset]:
        _, _, qrels, manifest, retrieval = prepare(name, config, device)
        rerank(name, config, qrels, retrieval, device)
    print("Local benchmark complete.", flush=True)

if __name__ == "__main__":
    main()
