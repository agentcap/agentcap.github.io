import math
import numpy as np
import pytest
from sklearn.metrics import ndcg_score
from searchbench.metrics import ndcg, evaluate, paired_delta
from searchbench.data import stable_sample
from searchbench.local import rrf
from searchbench.api import request_body, parse_answers

def test_ndcg_agrees_with_independent_sklearn_reference():
    qrels = {"a": 2, "b": 1, "c": 0, "d": 1}
    ranking = ["b", "c", "a", "d"]
    truth = np.array([[qrels[d] for d in ranking]])
    predicted = np.array([[4,3,2,1]])
    assert ndcg(ranking,qrels,3) == pytest.approx(ndcg_score(truth,predicted,k=3))
    assert ndcg(["a","b","d","c"],qrels) == pytest.approx(1)

def test_metrics_recall_and_mrr_cutoff():
    m = evaluate(["x","a","z"], {"a":2,"b":1})
    assert m["mrr10"] == .5
    assert m["recall50"] == .5
    assert m["ndcg10"] == pytest.approx((2/math.log2(3))/(2+1/math.log2(3)))
    with pytest.raises(ValueError):
        evaluate(["a","a"],{"a":1})

def test_rrf_agreement_and_deterministic_ties():
    result, _ = rrf([["a","b","c"],["b","d","a"]],k=60)
    assert result[0] == "b"
    assert len(result) == len(set(result))
    assert rrf([["z"],["a"]])[0] == ["a","z"]

def test_sampling_independent_of_input_order():
    assert stable_sample(list("abcdef"),3,17,"test") == stable_sample(list("fedcba"),3,17,"test")

def test_paired_bootstrap_zero_identical():
    assert paired_delta([0,1,.5],[0,1,.5]) == {"mean":0,"low":0,"high":0}

def test_provider_payloads_and_shared_rubric():
    config={"rerank_depth":2,"openai_model":"gpt-6-luna","jev_model":"jev-1.14.0"}
    row={"hybrid":["a","b"],"clipped_query":"test","features":[.5,.1]}
    docs={"a":"document a","b":"document b"}
    d=request_body("decisions","rerank",row,docs,config)
    j=request_body("jev","rerank",row,docs,config)
    assert [x["description"] for x in d["questions"][0]["levels"]] == j["questions"]["p000"]["criteria"]
    assert len(d["questions"]) == 2
    for provider in ("decisions","jev"):
        gate=request_body(provider,"gate",row,docs,config)
        assert "scores" not in str(gate)
        assert len(gate["questions"]) == 2

def test_refusals_and_missing_scores_are_failures():
    with pytest.raises(ValueError):
        parse_answers("decisions","rerank",{"answers":[{"name":"p000","type":"refusal"}]},1)
    with pytest.raises(KeyError):
        parse_answers("jev","rerank",{"answers":{}},1)
    with pytest.raises(ValueError):
        parse_answers("jev","rerank",{"answers":{"p000":{"score":float("nan")}}},1)

def test_llm_reads_final_phase_only():
    response={"output":[
        {"type":"message","phase":"commentary","content":[{"type":"output_text","text":'{"scores":[0]}'}]},
        {"type":"message","phase":"final_answer","content":[{"type":"output_text","text":'{"scores":[3]}'}]}]}
    assert parse_answers("llm","rerank",response,1) == {"p000":3}

from searchbench.report import choose_threshold, policy_rows

def test_threshold_fits_calibration_and_can_select_never_rerank():
    assert choose_threshold([.2,.8],[1,1],[0,0]) > .8
    assert choose_threshold([.2,.8],[0,0],[1,1],0) <= .2

def test_policy_includes_gate_latency_and_cost():
    base={"q":{"ndcg10":.5,"latency_ms":10}}
    target={"q":{"ndcg10":.7,"latency_ms":110,"rerank_ms":100,"cost_usd":.002}}
    overhead={"q":{"latency_ms":30,"cost_usd":.0001}}
    skip=policy_rows({"q":.2},.5,base,target,overhead)["q"]
    use=policy_rows({"q":.8},.5,base,target,overhead)["q"]
    assert skip["latency_ms"] == 40
    assert skip["cost_usd"] == .0001
    assert use["latency_ms"] == 140
    assert use["cost_usd"] == pytest.approx(.0021)

def test_incomplete_llm_response_is_rejected():
    with pytest.raises(ValueError):
        parse_answers("llm","rerank",{"status":"incomplete","output":[]},1)

def test_reranking_cannot_change_candidate_recall():
    from searchbench.report import api_rankings
    candidates=[str(i) for i in range(100)]
    raw={"q":{"answers":{f"p{i:03d}":float(i%4) for i in range(50)},
              "request_ms":20,"cost_usd":.001,"model":"test"}}
    retrieval={"q":{"hybrid":candidates,"metrics":{"hybrid":{"latency_ms":5}}}}
    qrels={"q":{"0":1,"49":2,"75":1}}
    result=api_rankings("test",raw,retrieval,qrels,50)["q"]
    assert result["recall50"] == evaluate(candidates,qrels["q"])["recall50"]
    assert set(result["ranking"][:50]) == set(candidates[:50])
    assert result["ranking"][50:] == candidates[50:]
