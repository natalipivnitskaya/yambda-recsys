| Method | Best epoch | HitRate@100 | Recall@100 | nDCG@100 | Coverage@100 |
|---|---:|---:|---:|---:|---:|
| Transformer, semantic ID embeddings (codes + item ids, 3x256 behaviour codes, 30 ep), in-batch negatives + logQ | 23 | 0.3774 | 0.1338 | 0.0512 | 0.4414 |
| Transformer, semantic ID embeddings (codes + item ids, 3x256 content codes, 30 ep), in-batch negatives + logQ | 30 | 0.3718 | 0.1311 | 0.0494 | 0.4486 |
| Transformer (2 layers, d=64, len 100, 30 ep), in-batch negatives + logQ | 26 | 0.3546 | 0.1242 | 0.0460 | 0.4069 |
| Generative retrieval over semantic IDs (codes only, 3x256 behaviour codes, beam 100, 20 ep) | 20 | 0.2803 | 0.0871 | 0.0334 | 0.1787 |
| Transformer, semantic ID embeddings (codes only, 3x256 behaviour codes, 20 ep), in-batch negatives + logQ | 19 | 0.2594 | 0.0748 | 0.0288 | 0.4082 |
| Generative retrieval over semantic IDs (codes only, 3x256 content codes, beam 100, 20 ep) | 20 | 0.2159 | 0.0625 | 0.0230 | 0.1602 |
