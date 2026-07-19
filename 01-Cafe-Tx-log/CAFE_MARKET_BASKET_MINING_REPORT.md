# Mock Cafe Market Basket Data & Association Rule Mining Report

**Project Folder:** `/Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/01-Cafe-Tx-log`  
**Execution Plan:** Adheres strictly to [`plan.md`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/01-Cafe-Tx-log/plan.md) (All 6 Decisions Locked & Confirmed)  

---

## 1. Executive Summary & Configuration Decisions

To support student labs and executive demonstration of **Market Basket Analysis (MBA)** with verifiable ground truth, we implemented the 5-phase execution plan in `01-Cafe-Tx-log/`. By locking in our 6 configuration decisions in [`config.py`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/01-Cafe-Tx-log/config.py), we generated a realistic latent-class dataset of **$5,000$ transactions across 3 customer archetypes (`Coffee-crowd 45%`, `Tea-crowd 35%`, `Mixed 20%`)**.

### Locked Configuration Parameters (`config.py`):
| # | Parameter | Value / Options | Status & Justification |
|---|---|---|---|
| **1** | **Item Catalog** | `["Coffee", "Bread", "Butter", "Tea", "Jam"]` | **Locked:** Keeps the 5-item core catalog. |
| **2** | **Customer Archetypes** | 3 (`Coffee-crowd`, `Tea-crowd`, `Mixed`) | **Locked:** Ensures clear cluster separation plus realistic background noise. |
| **3** | **Target Associations** | `Coffee <-> Butter` (Bonded), `Tea <-> Jam` (Bonded), `Coffee <-> Jam` (Repel) | **Locked:** Ground truth verified via automated `assert` checks. |
| **4** | **Basket Size** | Latent-class probabilistic sampling with non-empty resample | **Locked:** Generates variable basket sizes ($1$ to $5$ items) naturally. |
| **5** | **Mining Thresholds** | `min_support = 0.02 (2.0%)`, `min_confidence = 0.50 (50%)`, `min_lift > 1.0` | **Locked:** Captures all meaningful positive rules while filtering independence noise. |
| **6** | **Random Seed** | Fixed (`RANDOM_SEED = 42`) | **Locked:** Guarantees $100\%$ reproducible data generation and rule mining. |

---

## 2. Phase 3 Verification: Designed vs. Discovered Ground Truth

Our data generation engine ([`generate.py`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/01-Cafe-Tx-log/generate.py)) verified our designed pairwise associations by asserting that empirical `support`, `confidence`, and `lift` matched our intended direction. All assertions passed:

```
=====================================================================================
TABLE 1: DESIGNED VS DISCOVERED LIFT FOR TARGET ASSOCIATIONS
=====================================================================================
Target Pair (A -> B)   | Designed Intent    | Discovered Lift  | Discovered Conf.
-------------------------------------------------------------------------------------
Coffee -> Butter       | Bonded (> 1.3x)    |  1.615x          |  79.81%
Tea -> Jam             | Bonded (> 1.3x)    |  1.827x          |  73.59%
Coffee -> Jam          | Repel (< 1.0x)     |  0.304x          |  12.26%
=====================================================================================
```

> **Ground Truth Verification:** Because `Coffee-crowd` and `Tea-crowd` live in distinct archetypes, `Coffee` and `Butter` achieve a strong positive lift of **`1.615x`** ($79.81\%$ confidence), while `Tea` and `Jam` achieve a positive lift of **`1.827x`** ($73.59\%$ confidence). Conversely, `Coffee` and `Jam` exhibit a strong repelling relationship (`Lift = 0.304x`, confidence only $12.26\%$), confirming that naive independent sampling (`Lift ≈ 1.0`) was successfully avoided.

---

## 3. Phase 4 & 5 Mining Results (`FP-Growth` vs `Apriori`)

Our analysis engine ([`analyze.py`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/01-Cafe-Tx-log/analyze.py)) ran both **FP-Growth (`fpgrowth`)** and **Apriori (`apriori`)** at `min_support = 0.02`. The cross-check verified that both algorithms produced exactly **$24$ identical frequent itemsets** and **$16$ actionable association rules** (`confidence >= 50%`, `lift > 1.0`).

### Top Discovered Association Rules (Sorted by Lift):

| Rank | Antecedent(s) | Consequent(s) | Support | Confidence | Lift | Strategic Interpretation |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | `Tea, Bread` | `Jam` | 14.64% | 74.31% | **1.845x** | Bread acts as a vehicle reinforcing the Tea-Jam breakfast bond |
| **2** | `Jam` | `Tea` | 30.26% | 75.12% | **1.827x** | $75.1\%$ of shoppers buying Jam are purchasing Tea |
| **3** | `Tea` | `Jam` | 30.26% | 73.59% | **1.827x** | Reciprocal tea-drinker preference for jam |
| **4** | `Jam, Bread` | `Tea` | 14.64% | 74.01% | **1.800x** | High-confidence tri-item breakfast bundle |
| **5** | `Butter, Bread` | `Coffee` | 24.38% | 87.07% | **1.652x** | $87.1\%$ of shoppers buying Butter+Bread also purchase Coffee |
| **6** | `Coffee, Bread` | `Butter` | 24.38% | 81.38% | **1.647x** | Strong morning coffee routine cross-sell opportunity |
| **7** | `Coffee` | `Butter` | 42.06% | 79.81% | **1.615x** | Core morning beverage-spread relationship |
| **8** | `Butter` | `Coffee` | 42.06% | 85.11% | **1.615x** | Reciprocal morning purchase link |
| **9** | `Butter, Coffee` | `Bread` | 24.38% | 57.96% | **1.088x** | Bridge item connecting to bakery goods |
| **10** | `Coffee` | `Bread` | 29.96% | 56.85% | **1.067x** | General baseline morning staple overlap |

- **Association Lift Chart:** [`cafe_association_rules_lift.png`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/01-Cafe-Tx-log/cafe_association_rules_lift.png)

---

## 4. Strategic Shelf-Layout Recommendations

Grounded in our empirical association rule lift values, we recommend the following 3-part physical shelf optimization for the cafe:

1. **Co-Locate High-Lift Bonded Pairs (Cross-Merchandising Zone):**
   - **Coffee & Butter (`Lift = 1.615x`, `Conf = 79.8%`):** Position a refrigerated Butter cooler right beside the morning Coffee ordering station or self-serve condiment bar to capture immediate impulse cross-selling.
   - **Tea & Jam (`Lift = 1.827x`, `Conf = 73.6%`):** Display artisan Jam jars on eye-level shelving directly adjacent to loose-leaf Tea tins and boxed tea displays.
2. **Separate Repelling / Competing Pairs (Store Circulation Strategy):**
   - **Coffee & Jam (`Lift = 0.304x`, `Conf = 12.3%`):** Shoppers buying Jam are predominantly Tea drinkers. Placing Jam jars next to the Coffee counter wastes prime impulse-buy shelf real estate. Keep Jam strictly confined to the Tea & Bakery aisle.
3. **Bread Hub Centralization:**
   - **Bread as Universal Bridge:** Bread bridges both Coffee (`56.8%` confidence) and Tea (`52.6%` confidence) archetypes. Position fresh Bread in a central bakery island accessible from both morning beverage queues.
