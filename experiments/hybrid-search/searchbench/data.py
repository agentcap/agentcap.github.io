"""Download official BEIR archives and freeze query selections before evaluation."""
from __future__ import annotations
import csv, hashlib, json, urllib.request, zipfile
from pathlib import Path

DATASETS = {
    "scifact": "5f7d1de60b170fc8027bb7898e2efca1",
    "nfcorpus": "a89dba18a62ef92f7d323ec890a0d38d",
}

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)

def load_config(path="config.json"):
    return json.loads(Path(path).read_text())

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]

def stable_sample(ids, count, seed, namespace):
    return sorted(ids, key=lambda x: hashlib.sha256(f"{seed}:{namespace}:{x}".encode()).hexdigest())[:count]

def download(name, root="data"):
    root = Path(root)
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{name}.zip"
    if not archive.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        print(f"Downloading official BEIR {name}", flush=True)
        urllib.request.urlretrieve(url, archive)
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if md5 != DATASETS[name]:
        raise ValueError(f"{name}: archive checksum mismatch")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise ValueError("Unsafe archive path")
        zf.extractall(root)
    dump(root / f"{name}.provenance.json", {
        "source": f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip",
        "md5": md5, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    })
    return target

def load_dataset(name, config):
    folder = download(name)
    corpus = {}
    for line in (folder / "corpus.jsonl").open():
        row = json.loads(line)
        corpus[str(row["_id"])] = (row.get("title", "") + " " + row["text"]).strip()
    queries = {str(r["_id"]): r["text"] for r in map(json.loads, (folder / "queries.jsonl").open())}
    qrels = {}
    for split in ("train", "dev", "test"):
        path = folder / "qrels" / f"{split}.tsv"
        if path.exists():
            qrels[split] = {}
            for r in csv.DictReader(path.open(), delimiter="\t"):
                qrels[split].setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    calibration_split = "dev" if "dev" in qrels else "train"
    test_ids = sorted(q for q, rels in qrels["test"].items() if any(v > 0 for v in rels.values()))
    test_texts = {" ".join(queries[q].casefold().split()) for q in test_ids}
    calibration_pool = [q for q in qrels[calibration_split] if q not in set(test_ids)
                        and " ".join(queries[q].casefold().split()) not in test_texts
                        and any(v > 0 for v in qrels[calibration_split][q].values())]
    cal_ids = stable_sample(calibration_pool, config["calibration_queries"], config["seed"], name + ":cal")
    api_ids = stable_sample(test_ids, config["api_test_queries_per_dataset"], config["seed"], name + ":api")
    manifest = {"dataset": name, "corpus_size": len(corpus), "test_queries": test_ids,
                "calibration_split": calibration_split, "calibration_queries": cal_ids,
                "api_test_queries": api_ids, "seed": config["seed"],
                "config_hash": fingerprint(config)}
    path = Path("results") / name / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Selection/config changed. Use a fresh results directory.")
    dump(path, manifest)
    rels = {**qrels[calibration_split], **qrels["test"]}
    assert not set(cal_ids) & set(test_ids)
    return corpus, queries, rels, manifest
