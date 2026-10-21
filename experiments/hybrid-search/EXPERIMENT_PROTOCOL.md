# Frozen protocol

Date: 2026-10-21. Seed: 20261021. Configuration: `config.json`.

## Data and splits

Use the complete official BEIR SciFact (5,183 abstracts) and NFCorpus (3,633 abstracts) corpora. Verify official archive MD5 and record SHA-256. Use all 300 SciFact and 323 NFCorpus test queries for local baselines. Freeze 120 test queries per dataset by SHA-256 ordering before API outcomes. Freeze 80 calibration queries from a disjoint official split (SciFact train; NFCorpus dev). Exclude normalized duplicate query text across calibration and test, even when IDs differ. Never use judgments in retrieval, reranking, or gate inputs.

## Retrieval and reranking

- BM25Okapi k1=1.2, b=0.75; Unicode word tokens, lowercase, English Snowball stemming, no stopword removal.
- E5-small-v2, normalized embeddings, exact full-corpus cosine search, query/passage prefixes, 512-token limit.
- Equal-weight RRF over each retriever's top 100, RRF k=60.
- Rerank the first 50 hybrid candidates; append the untouched tail through rank 100. Identical IDs and excerpts for all rerankers: 384 MiniLM WordPiece passage tokens and 96 query tokens. Stable hybrid order breaks ties.
- Two cross-encoders: TinyBERT-L2 and MiniLM-L6. APIs: GPT-6 Luna Responses with reasoning=none; Jev 1.14; Decisions with GPT-6 Luna.
- Same four-level relevance rubric for APIs. Decisions/Jev return expected rubric scores; GPT returns numeric scores in one structured response per query. Its batched context differs from independently evaluated typed questions. No best-in-family claims.

## Selective reranking

Gate inputs: query, top-10 snippets at 90 whitespace words each, ranks 11-50 snippets at 35 words, retrieval agreement and fusion margin. No labels, cross-encoder/LLM scores, or reranker outputs. Ask separately about MiniLM and GPT reranking benefit of at least 0.02 absolute nDCG@10. Decisions predicate and Jev Noul return gate scores; both backend questions share one request.

Compare threshold 0.5 and calibration-only thresholds: minimize reranking rate subject to calibration mean nDCG@10 >= always-rerank minus 0.01; break ties by higher quality then lower threshold. This heuristic gives no guaranteed test loss bound. Include a free disagreement gate using one minus top-10 Jaccard overlap. Report AUC, Brier score, skip rate, harmful skips, paired quality differences, and gate-inclusive latency/cost. A never-rerank calibration outcome is a degenerate skip policy, not evidence of query-difficulty detection.

Replay policies from independent per-query measured outcomes. Policy latency is an estimated serial composition, not a live end-to-end deployment measurement. Include the whole measured two-backend gate request overhead for each route. A dedicated single-question gate could differ.

## Metrics and uncertainty

TREC-style linear-gain nDCG@10, MRR@10, Recall@50/100. Unjudged documents count as zero; acknowledge incomplete judgments. API comparisons use identical 120-query samples per dataset. Use paired query bootstrap 95% intervals, 5,000 resamples. No pooled score presented as a BEIR leaderboard result.

Warm local latency excludes index construction, downloads, model loads and disk output. Synchronize GPU around timing. API concurrency two with deliberate dispatch pacing; latency comparisons use final HTTP request wall time, including error responses with declared fallback, excluding rate-limit pacing, earlier failed attempts and retry backoff. Report retries and invalid outputs separately; do not claim production tail latency. Save request time, total attempt time, retries and usage. Remote-vs-local timing reflects this machine and network, not intrinsic architecture speed. Token costs use public standard uncached list prices, excluding local compute, startup and tiny preflight calls.

## Reproducibility

Persist frozen query IDs, checksums, model revisions where available, lockfile, rankings/scores/timings, ignored raw API cache, request hashes, usage ledger and summaries. Keys remain in process memory. Combined API cap USD 5 across resumes, with conservative pre-dispatch reservations.

Sources: [BEIR](https://github.com/beir-cellar/beir), [E5](https://huggingface.co/intfloat/e5-small-v2), [MiniLM](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), [Decisions](https://developers.openai.com/api/docs/guides/decisions), [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [Jev models](https://docs.typesafe.ai/models), [TypeSafe API](https://docs.typesafe.ai/api/system-one).

Execution amendment before completing API evaluation: preserve non-retryable API rejections as failures. A failed reranker keeps hybrid order; a failed gate requests reranking. Report these failures and include fallback queries in paired quality metrics rather than dropping them. No prompt changes to bypass rejection. Dispatch pacing scales with request size.
