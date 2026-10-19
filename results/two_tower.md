| Method | HitRate@100 | Recall@100 | nDCG@100 | Coverage@100 |
|---|---:|---:|---:|---:|
| Sampled softmax, in-batch negatives + logQ | 0.3330 | 0.1084 | 0.0394 | 0.4336 |
| Sampled softmax, uniform negatives | 0.3271 | 0.1046 | 0.0381 | 0.4305 |
| Full softmax | 0.3155 | 0.0991 | 0.0359 | 0.5550 |
| Sampled softmax, in-batch + uniform negatives | 0.2901 | 0.0880 | 0.0324 | 0.5058 |
| BPR, uniform negatives | 0.2274 | 0.0631 | 0.0218 | 0.1412 |
| Sampled softmax, in-batch negatives | 0.2081 | 0.0564 | 0.0202 | 0.7265 |
| BCE, uniform negatives | 0.1955 | 0.0514 | 0.0178 | 0.2100 |
