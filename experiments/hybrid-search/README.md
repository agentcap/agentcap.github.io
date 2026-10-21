# Hybrid search: when should you rerank?

[Read the portfolio article](https://agentcap.github.io/writing/hybrid-search-reranking.html) · [Download the complete code and per-query results](hybrid-search-benchmark.zip)

This folder publishes the implementation and aggregate measurements. The ZIP additionally contains all per-query numeric results and original experiment artifacts.

A reproducible comparison of hybrid retrieval, two cross-encoders, GPT-6 Luna, Jev, and OpenAI Decisions, plus selective reranking with Decisions/Jev gates.

## Outputs

- `blog/hybrid-search-reranking.md`: portfolio-ready article grounded in the actual run.
- `blog/index.html`: standalone local reading copy with figures and result tables.
- `results/summary.json`, `results/metrics.csv`, `results/gates.csv`: aggregate results.
- `results/<dataset>/manifest.json`: frozen query IDs and configuration fingerprint.
- `results/<dataset>/*.json`: rankings, relevance scores, API timings and policy outcomes.
- `results/api_ledger.json`: usage, estimated list-price charges, attempts and errors.
- `EXPERIMENT_PROTOCOL.md`: candidates, splits, prompts, metrics and limitations.

## Reproduce

Python 3.12, uv, and optional CUDA. This run used an RTX 4070. Downloaded datasets, model caches, and raw API responses are excluded by `.gitignore`. The software environment is pinned in `uv.lock`.

```sh
uv sync --frozen --group dev
uv run python -m pytest -q
uv run python -m searchbench.local
```

Then set OPENAI_API_KEY and TYPESAFE_API_KEY in the process environment. Alternatively pass `--openai-key-file /path/to/key.txt --jev-key-file /path/to/key.txt`; credential files are read directly into memory. Never put keys in source, result files or command arguments.

```sh
uv run python -m searchbench.api --budget 5
uv run python -m searchbench.report
uv run python -m searchbench.figures
```

API calls resume from validated response caches and per-query outputs. The USD budget covers the cumulative persisted ledger across resumes. Costs use documented standard uncached token list prices, including recorded retry attempts; they are estimates, not invoices. Tiny preflight calls and a few interrupted in-flight requests are outside the ledger. Reported API latency uses the final HTTP request, including error responses when fallback is used. Deliberate dispatch pacing, earlier failed attempts, and retry backoff are excluded; reliability events are reported separately.

Local baselines evaluate all 623 official test queries. API comparisons and gate policies use the same preselected 120 test queries per dataset, with 80 separate calibration queries per dataset. Calibrated routing is replayed from measured query outcomes; its serial latency is an estimate, not a live deployment result.

Run a calibration-only API smoke test:

```sh
uv run python -m searchbench.api --dataset scifact --limit 2 --budget 5
```

Do not edit `config.json` while reusing a results directory: the loader rejects changed manifests. To run a different protocol, copy the source into a fresh experiment directory. No reranker or gate receives qrels. Text-duplicate query checks complement ID-based split separation.

## Sources

- [BEIR benchmark and dataset checksums](https://github.com/beir-cellar/beir)
- [SciFact](https://github.com/allenai/scifact)
- [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
- [E5-small-v2](https://huggingface.co/intfloat/e5-small-v2)
- [TinyBERT-L2 reranker](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2)
- [MiniLM-L6 reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
- [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [TypeSafe models and pricing](https://docs.typesafe.ai/models)

The bundle retains measured results for inspection and resume. To take fresh measurements, use a source-only copy with results, cache, and data directories absent. The raw passage cache is deliberately omitted from the bundle; running the local stage regenerates it from the official downloads. Run the searchbench.render_blog module after editing the Markdown article.
