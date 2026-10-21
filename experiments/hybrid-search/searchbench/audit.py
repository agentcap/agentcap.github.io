"""Fail loudly on incomplete, mismatched or contaminated final result sets."""
from pathlib import Path
import hashlib, json
from .data import load_config, load_dataset, dump
from .api import parse_answers

def main():
    config=load_config()
    audit={"datasets":{},"source_sha256":{}}
    for path in sorted(Path("searchbench").glob("*.py")):
        audit["source_sha256"][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [Path("config.json"),Path("uv.lock"),Path("models.lock.json"),Path("EXPERIMENT_PROTOCOL.md")]:
        audit["source_sha256"][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in config["datasets"]:
        corpus,queries,qrels,m=load_dataset(name,config)
        cal,test=m["calibration_queries"],m["test_queries"]
        normalize=lambda text:" ".join(text.casefold().split())
        assert not set(cal)&set(test)
        assert not {normalize(queries[q]) for q in cal}&{normalize(queries[q]) for q in test}
        folder=Path("results")/name
        retrieval=json.loads((folder/"retrieval.json").read_text())
        expected=set(cal+m["api_test_queries"])
        model_summary={}
        for method in ("ce_tinybert","ce_minilm"):
            rows=json.loads((folder/f"{method}.json").read_text())
            assert set(rows)==set(cal+test)
            for q,r in rows.items():
                assert set(r["ranking"])==set(retrieval[q]["hybrid"])
                assert len(r["ranking"])==len(set(r["ranking"]))
        for provider in ("decisions","jev","llm"):
            raw=json.loads((folder/f"api_{provider}_rerank.json").read_text())
            assert set(raw)==expected, f"{name}/{provider}: incomplete API run"
            for q,r in raw.items():
                assert set(r["answers"])=={f"p{i:03d}" for i in range(config["rerank_depth"])}
            model_summary[provider]=sorted({r["model"] for r in raw.values()},key=str)
        for provider in ("decisions","jev"):
            gate=json.loads((folder/f"api_{provider}_gate.json").read_text())
            assert set(gate)==expected
            assert all(set(r["answers"])=={"cross_encoder","llm"} for r in gate.values())
        audit["datasets"][name]={"corpus_size":len(corpus),"dataset_provenance":json.loads((Path("data")/f"{name}.provenance.json").read_text()),"calibration_queries":len(cal),
             "test_queries":len(test),"paired_api_test_queries":len(m["api_test_queries"]),
             "calibration_test_id_overlap":0,"calibration_test_text_overlap":0,"observed_models":model_summary}
    ledger=json.loads(Path("results/api_ledger.json").read_text())
    audit["estimated_api_spend_usd"]=sum(r["cost_usd"] for r in ledger)
    assert audit["estimated_api_spend_usd"]<=5
    audit["checks_passed"]=True
    dump("results/audit.json",audit)
    print(json.dumps(audit["datasets"],indent=2))
    print("Final integrity audit passed.")

if __name__=="__main__":
    main()
