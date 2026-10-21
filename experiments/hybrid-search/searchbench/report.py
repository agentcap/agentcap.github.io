"""Aggregate paired results and evaluate pre-calibrated selective reranking policies."""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from .data import dump, load_config, load_dataset
from .metrics import aggregate, evaluate, paired_delta

LABELS = {"bm25":"BM25", "dense":"E5 dense", "hybrid":"Hybrid RRF",
          "ce_tinybert":"TinyBERT cross-encoder", "ce_minilm":"MiniLM cross-encoder",
          "llm":"GPT-6 Luna", "jev":"Jev 1.14", "decisions":"Decisions API"}

def api_rankings(provider, raw, retrieval, qrels, depth):
    rows={}
    for q, response in raw.items():
        ids = retrieval[q]["hybrid"][:depth]
        scores = [response["answers"][f"p{i:03d}"] for i in range(len(ids))]
        ordered = [ids[i] for i in sorted(range(len(ids)), key=lambda i:-scores[i])]
        ranking = ordered + retrieval[q]["hybrid"][depth:]
        rows[q] = {**evaluate(ranking,qrels[q]), "ranking":ranking,
                   "rerank_ms":response["request_ms"],
                   "latency_ms":response["request_ms"]+retrieval[q]["metrics"]["hybrid"]["latency_ms"],
                   "cost_usd":response["cost_usd"], "scores":dict(zip(ids,scores)),
                   "model":response["model"]}
    return rows

def choose_threshold(prob, base, rerank, tolerance=.01):
    """Minimize reranking rate within calibration mean loss tolerance vs always-rerank."""
    prob,base,rerank = map(np.asarray,(prob,base,rerank))
    choices=[]
    for threshold in np.unique(np.r_[0,np.linspace(0,1,21),1.01]):
        mask=prob>=threshold
        quality=np.where(mask,rerank,base).mean()
        if quality>=rerank.mean()-tolerance-1e-12:
            choices.append((float(mask.mean()),-float(quality),float(threshold)))
    return min(choices)[2]

def policy_rows(prob, threshold, base, target, overhead):
    rows={}
    for q in prob:
        use=prob[q]>=threshold
        source=target[q] if use else base[q]
        rows[q]={**source,"reranked":bool(use),
            "latency_ms":base[q]["latency_ms"]+overhead[q]["latency_ms"]+
                         (target[q]["rerank_ms"] if use else 0),
            "cost_usd":overhead[q]["cost_usd"]+(target[q].get("cost_usd",0) if use else 0)}
    return rows

def auc_if_defined(y,p):
    return float(roc_auc_score(y,p)) if len(set(y))>1 else None

def summarize_policy(rows, target, base, ids, prob, config):
    result=aggregate(rows,ids)
    mask=np.array([rows[q]["reranked"] for q in ids])
    result.update(
        rerank_rate=float(mask.mean()), skip_rate=float(1-mask.mean()),
        api_cost_per_1000=float(np.mean([rows[q]["cost_usd"] for q in ids])*1000),
        mean_latency_delta_vs_always_ms=float(np.mean([rows[q]["latency_ms"]-target[q]["latency_ms"] for q in ids])),
        mean_quality_delta_vs_always=paired_delta([rows[q]["ndcg10"] for q in ids],[target[q]["ndcg10"] for q in ids]),
        mean_quality_delta_vs_hybrid=paired_delta([rows[q]["ndcg10"] for q in ids],[base[q]["ndcg10"] for q in ids]),
    )
    y=[target[q]["ndcg10"]-base[q]["ndcg10"]>=config["gate_benefit_delta"] for q in ids]
    result["benefit_prevalence"]=float(np.mean(y))
    result["benefit_auc"]=auc_if_defined(y,[prob[q] for q in ids])
    result["benefit_brier"]=float(brier_score_loss(y,[prob[q] for q in ids]))
    positive=sum(y)
    result["beneficial_query_recall"]=float(sum(m and label for m,label in zip(mask,y))/positive) if positive else None
    result["harmful_skip_rate"]=float(np.mean([not rows[q]["reranked"] and
        target[q]["ndcg10"]-base[q]["ndcg10"]>=config["gate_benefit_delta"] for q in ids]))
    return result

def run():
    config=load_config()
    summary={"config":config,"datasets":{}}
    all_csv=[]
    gates_csv=[]
    for name in config["datasets"]:
        _,queries,qrels,manifest=load_dataset(name,config)
        folder=Path("results")/name
        retrieval=json.loads((folder/"retrieval.json").read_text())
        methods={method:{q:{**r["metrics"][method],"cost_usd":0,"ranking":r[method]}
                          for q,r in retrieval.items()} for method in ("bm25","dense","hybrid")}
        for method in ("ce_tinybert","ce_minilm"):
            path=folder/(method+".json")
            if path.exists():
                methods[method]=json.loads(path.read_text())
        for provider in ("decisions","jev","llm"):
            path=folder/f"api_{provider}_rerank.json"
            if not path.exists():
                continue
            raw=json.loads(path.read_text())
            methods[provider]=api_rankings(provider,raw,retrieval,qrels,config["rerank_depth"])
            dump(folder/f"evaluated_{provider}.json",methods[provider])
        test=manifest["test_queries"]
        paired=manifest["api_test_queries"]
        cal=manifest["calibration_queries"]
        common=[q for q in paired if all(q in rows for rows in methods.values())]
        d={"corpus_size":manifest["corpus_size"],"full_test_queries":len(test),
           "calibration_queries":len(cal),"calibration_split":manifest["calibration_split"],
           "paired_test_queries":len(common),"full":{},"paired":{},"gates":{},"gate_curves":{}}
        for method,rows in methods.items():
            for scope,ids in (("full",test),("paired",common)):
                if not ids or any(q not in rows for q in ids):
                    continue
                result=aggregate(rows,ids)
                result["api_cost_per_1000"]=float(np.mean([rows[q].get("cost_usd",0) for q in ids])*1000)
                result["delta_vs_hybrid"]=paired_delta([rows[q]["ndcg10"] for q in ids],
                                                      [methods["hybrid"][q]["ndcg10"] for q in ids])
                if "rerank_ms" in rows[ids[0]]:
                    result["rerank_p50_ms"]=float(np.median([rows[q]["rerank_ms"] for q in ids]))
                    result["rerank_p95_ms"]=float(np.quantile([rows[q]["rerank_ms"] for q in ids],.95))
                d[scope][method]=result
                all_csv.append({"dataset":name,"scope":scope,"method":method,
                                **{k:v for k,v in result.items() if not isinstance(v,dict)}})
        d["paired_differences"]={}
        if len(common)==len(paired):
            names=list(methods)
            for i,left in enumerate(names):
                for right in names[i+1:]:
                    d["paired_differences"][f"{left} - {right}"]=paired_delta(
                        [methods[left][q]["ndcg10"] for q in common],
                        [methods[right][q]["ndcg10"] for q in common])
        base=methods["hybrid"]
        for backend,target_name in (("cross_encoder","ce_minilm"),("llm","llm")):
            if target_name not in methods or any(q not in methods[target_name] for q in cal+paired):
                continue
            target=methods[target_name]
            d["gates"][backend]={}
            d["gate_curves"][backend]={}
            for provider in ("heuristic","decisions","jev"):
                ids=cal+paired
                if provider=="heuristic":
                    prob={q:1-retrieval[q]["features"][0] for q in ids}
                    overhead={q:{"latency_ms":0,"cost_usd":0} for q in ids}
                else:
                    path=folder/f"api_{provider}_gate.json"
                    if not path.exists():
                        continue
                    gate=json.loads(path.read_text())
                    if any(q not in gate for q in ids):
                        continue
                    prob={q:gate[q]["answers"][backend] for q in ids}
                    overhead={q:{"latency_ms":gate[q]["request_ms"],"cost_usd":gate[q]["cost_usd"]} for q in ids}
                threshold=choose_threshold([prob[q] for q in cal],[base[q]["ndcg10"] for q in cal],
                                           [target[q]["ndcg10"] for q in cal],config["gate_max_ndcg_loss"])
                for mode,t in (("fixed_0.5",.5),("calibrated",threshold)):
                    rows=policy_rows(prob,t,base,target,overhead)
                    result=summarize_policy(rows,target,base,paired,prob,config)
                    result.update(threshold=t,calibration_ndcg=float(np.mean([rows[q]["ndcg10"] for q in cal])),
                                  calibration_always_ndcg=float(np.mean([target[q]["ndcg10"] for q in cal])),
                                  gate_mean_ms=float(np.mean([overhead[q]["latency_ms"] for q in paired])),
                                  saved_rerank_mean_ms=float(np.mean([0 if rows[q]["reranked"] else target[q]["rerank_ms"] for q in paired])))
                    d["gates"][backend][provider+"_"+mode]=result
                    dump(folder/f"policy_{backend}_{provider}_{mode}.json",{q:rows[q] for q in paired})
                    gates_csv.append({"dataset":name,"backend":backend,"gate":provider,"mode":mode,
                                      **{k:v for k,v in result.items() if not isinstance(v,dict)}})
                curve=[]
                for t in np.r_[np.linspace(0,1,21),1.01]:
                    rows=policy_rows(prob,float(t),base,target,overhead)
                    curve.append({"threshold":float(t),"skip_rate":float(np.mean([not rows[q]["reranked"] for q in paired])),
                        "ndcg10":float(np.mean([rows[q]["ndcg10"] for q in paired])),
                        "mean_latency_ms":float(np.mean([rows[q]["latency_ms"] for q in paired]))})
                d["gate_curves"][backend][provider]=curve
            d["oracle_"+backend]=float(np.mean([max(target[q]["ndcg10"],base[q]["ndcg10"]) for q in paired]))
            d["rerank_effect_"+backend]={
                "improves":sum(target[q]["ndcg10"]>base[q]["ndcg10"]+1e-9 for q in paired),
                "ties":sum(abs(target[q]["ndcg10"]-base[q]["ndcg10"])<=1e-9 for q in paired),
                "hurts":sum(target[q]["ndcg10"]<base[q]["ndcg10"]-1e-9 for q in paired)}
        if "llm" in methods and len(common)==len(paired):
            cases=[]
            for q in common:
                cases.append({"query_id":q,"query":queries[q],
                    "ndcg10":{method:rows[q]["ndcg10"] for method,rows in methods.items()},
                    "top1":{method:rows[q]["ranking"][0] for method,rows in methods.items()},
                    "top1_judgment":{method:qrels[q].get(rows[q]["ranking"][0],0) for method,rows in methods.items()}})
            dump(folder/"case_studies.json",sorted(cases,key=lambda r:r["ndcg10"]["llm"]-r["ndcg10"]["hybrid"],reverse=True))
        summary["datasets"][name]=d
    reliability = {}
    for name in config["datasets"]:
        folder = Path("results")/name
        reliability[name] = {}
        for path in sorted(folder.glob("api_*.json")):
            rows = json.loads(path.read_text())
            reliability[name][path.stem] = {
                "queries":len(rows), "failed_queries":sum(bool(r.get("failed")) for r in rows.values()),
                "extra_attempts_in_saved_queries":sum(r.get("attempts",1)-1 for r in rows.values()),
                "input_tokens_successful_final_attempt":sum(r["usage"].get("input_tokens",0) for r in rows.values()),
                "output_tokens_successful_final_attempt":sum(r["usage"].get("output_tokens",0) for r in rows.values()),
            }
    summary["reliability"] = reliability
    summary["invalid_response_artifacts"] = len(list(Path("cache/api/invalid").glob("*.json")))
    ledger=Path("results/api_ledger.json")
    if ledger.exists():
        entries=json.loads(ledger.read_text())
        summary["api_spend_usd"]=sum(e["cost_usd"] for e in entries)
        summary["api_requests"]=len(entries)
        summary["api_http_statuses"]={str(s):sum(e["http_status"]==s for e in entries)
                                       for s in sorted(set(e["http_status"] for e in entries),key=str)}
    dump("results/summary.json",summary)
    for filename,rows in (("results/metrics.csv",all_csv),("results/gates.csv",gates_csv)):
        if rows:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            with open(filename,"w") as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({name:{"n":d["paired_test_queries"],
         "ndcg10":{m:round(s["ndcg10"],4) for m,s in d["paired"].items()}}
         for name,d in summary["datasets"].items()},indent=2))

if __name__=="__main__":
    run()
