# Week 11: Recommendation Systems (Lab 11A to 11C)

| Folder | Content |
|---|---|
| `data/` | Teaching CSVs supplied with the labs (identical across 11A, 11B, 11C) |
| `reco_contract.py` | Shared data and evaluation contract (validation, temporal split, mappings, Precision/Recall/NDCG@K) |
| `lab11A/` | Foundations: validation, sparse representation, popularity baseline |
| `lab11B/` | Content-based and collaborative filtering |
| `lab11C/` | Hybrid recommendation and validation |
| `comparison/` | Comparison with newer recommender methods |

Run from any directory:

```bash
python3 -m pip install -r w11_RecommendationSystems/requirements.txt
python3 w11_RecommendationSystems/lab11A/lab11A_foundations.py
python3 w11_RecommendationSystems/lab11B/lab11B_content_cf.py
python3 w11_RecommendationSystems/lab11C/lab11C_hybrid_validation.py
python3 w11_RecommendationSystems/comparison/ease_check.py
```
