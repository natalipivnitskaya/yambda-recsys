| Method | HitRate@100 | Recall@100 | nDCG@100 | Coverage@100 |
|---|---:|---:|---:|---:|
| Transformer, semantic ID embeddings (codes + item ids, 3x256 codes, 20 ep), in-batch negatives + logQ | 0.3710 | 0.1284 | 0.0479 | 0.3957 |
| Transformer (2 layers, d=64, len 100, 20 ep), in-batch negatives + logQ | 0.3532 | 0.1222 | 0.0458 | 0.3577 |
| Transformer (2 layers, d=64, len 100, 10 ep), in-batch negatives + logQ | 0.3225 | 0.1070 | 0.0384 | 0.2420 |
| Generative retrieval over semantic IDs (codes only, 3x256 codes, beam 100, 20 ep) | 0.2088 | 0.0581 | 0.0218 | 0.1671 |
| Transformer, semantic ID embeddings (codes only, 3x256 codes, 20 ep), in-batch negatives + logQ | 0.1267 | 0.0298 | 0.0107 | 0.6296 |
| Transformer (2 layers, d=64, len 100, 10 ep), in-batch negatives | 0.1071 | 0.0247 | 0.0076 | 0.3644 |
