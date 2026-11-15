| Method | HitRate@100 | Recall@100 | nDCG@100 | Coverage@100 |
|---|---:|---:|---:|---:|
| Transformer, semantic ID embeddings (codes + item ids, 3x256 behaviour codes, 23 ep), in-batch negatives + logQ | 0.3866 | 0.1385 | 0.0529 | 0.4403 |
| Transformer, semantic ID embeddings (codes + item ids, 3x256 content codes, 30 ep), in-batch negatives + logQ | 0.3733 | 0.1310 | 0.0495 | 0.4593 |
| Transformer (2 layers, d=64, len 100, 26 ep), in-batch negatives + logQ | 0.3557 | 0.1236 | 0.0459 | 0.4015 |
| Transformer (2 layers, d=64, len 100, 10 ep), in-batch negatives + logQ | 0.3225 | 0.1070 | 0.0384 | 0.2420 |
| Generative retrieval over semantic IDs (codes only, 3x256 behaviour codes, beam 100, 20 ep) | 0.2877 | 0.0900 | 0.0342 | 0.1952 |
| Transformer, semantic ID embeddings (codes only, 3x256 behaviour codes, 19 ep), in-batch negatives + logQ | 0.2616 | 0.0766 | 0.0280 | 0.4153 |
| Generative retrieval over semantic IDs (codes only, 3x256 content codes, beam 100, 20 ep) | 0.2119 | 0.0596 | 0.0219 | 0.1590 |
| Transformer, semantic ID embeddings (codes only, 3x256 content codes, 20 ep), in-batch negatives + logQ | 0.1311 | 0.0317 | 0.0117 | 0.6122 |
| Transformer (2 layers, d=64, len 100, 10 ep), in-batch negatives | 0.1071 | 0.0247 | 0.0076 | 0.3644 |
