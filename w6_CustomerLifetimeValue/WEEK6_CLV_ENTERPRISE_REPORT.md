# Master Executive Report: Customer Lifetime Value (CLV) & Enterprise Valuation

**Subject:** Business Analysis & Business Data Analytics (Week 06)  
**Curriculum Document:** [`Slide/Week_06_CLV_Enterprise_Upgrade_20260801.pdf`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/Slide/Week_06_CLV_Enterprise_Upgrade_20260801.pdf)  
**Stakeholders:** Board of Directors, Chief Executive Officer (CEO), Chief Commercial Officer (CCO), Chief Financial Officer (CFO)  
**Analytical Standard:** [`DATA_SCIENCE_PIPELINE.md`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/DATA_SCIENCE_PIPELINE.md)

---

## 1. Executive Summary & Cross-Study Synthesis

Week 06 establishes the mathematical and operational paradigm for transitioning commercial decision-making from backward-looking heuristic accounting to forward-looking **Probabilistic Customer Lifetime Value (CLV)**.

Across both practicum studies from the lecture curriculum:

1. **Case Study 1: Artisanal Cafe Ticket Log Simulation (Slide 48)**
   - **Baseline:** 10-customer ticket stream representing small-business commercial accounts.
   - **Key Finding:** Naive historical extrapolation overestimated total forward value by **$1,514.47 (-16.4%)**, failing to detect latent defection in accounts like `CUST_1004` ($P(\text{Alive}) = 37.5\%$) and `CUST_1010` ($P(\text{Alive}) = 45.5\%$).
   - **Value Concentration:** **75.4% ($5,810.32)** of all forward net profit resides in just three accounts (`CUST_1005`, `CUST_1003`, `CUST_1009`).
   - Detailed Case Report: [`case1_cafe_clv/CAFE_CLV_REPORT.md`](case1_cafe_clv/CAFE_CLV_REPORT.md).

2. **Case Study 2: Enterprise Online Retail Mining Project (Slide 49)**
   - **Baseline:** 541,909 raw e-commerce transaction logs across UK and global international markets.
   - **Data Hygiene Protocol:** Filtered 144,025 invalid lines (guest checkouts, cancellations, returns, zero prices) into 397,884 verified positive purchases ($8.91M clean revenue).
   - **Key Finding:** BG/NBD and Gamma-Gamma ML scoring established a **$2,582,482.64** 12-month net forward margin run-rate across 4,338 verified customer accounts.
   - **VIP Champions (Top Quartile):** 1,085 accounts drive **$1,773,365.25 (68.7%)** of all forward net cash flow at an average valuation of **$1,634.44 per account**.
   - Detailed Case Report: [`case2_online_retail_clv/ONLINE_RETAIL_CLV_REPORT.md`](case2_online_retail_clv/ONLINE_RETAIL_CLV_REPORT.md).

---

## 2. Cross-Case Comparative Matrix

| Metric | Case Study 1: Artisanal Cafe | Case Study 2: Enterprise Online Retail |
| :--- | :---: | :---: |
| **Curriculum Source** | Slide 48 | Slide 49 |
| **Raw Dataset Scale** | 10 accounts (30-day ticket log) | 541,909 transaction rows |
| **Clean Customers Analyzed** | 10 verified accounts | 4,338 verified accounts |
| **Gross Revenue Baseline** | $9,218.40 (historical margin base) | $8,911,407.90 clean transactions |
| **Predicted 12M Net Margin CLV** | **$7,703.93** | **$2,582,482.64** |
| **Top Cohort Concentration** | Top 3 accounts = **75.4%** | Top 25% accounts = **68.7%** |
| **Average Forward CLV / Account** | $770.39 | $595.32 |
| **Average Probability Active** | 76.7% | 99.9% |
| **Fitted BG/NBD Parameters** | $r=0.55, \alpha=10.8, a=0.78, b=2.45$ | $r=0.8265, \alpha=68.9593, a=0.0174, b=65.7896$ |
| **Primary Commercial Focus** | VIP lock-in & churn prevention | Warehouse priority bays & dynamic dynamic promotions |

---

## 3. Strategic Intervention Taxonomy: Type 1 vs Type 2 Decisions (Slides 45-47)

```
                       ┌───────────────────────────────────────────────────────────┐
                       │          PROBABILISTIC CLV COMMERCIAL DECISIONS           │
                       └─────────────────────────────┬─────────────────────────────┘
                                                     │
                     ┌───────────────────────────────┴───────────────────────────────┐
                     ▼                                                               ▼
   ┌───────────────────────────────────┐                           ┌───────────────────────────────────┐
   │    TYPE 1: IRREVERSIBLE ACTION    │                           │     TYPE 2: REVERSIBLE ACTION     │
   │  High Capex / Contractual Lock-In │                           │   Real-Time Digital Experimentation   │
   ├───────────────────────────────────┤                           ├───────────────────────────────────┤
   │ • Dedicated VIP fulfillment bays  │                           │ • In-app dynamic checkout voucher │
   │ • Multi-year penalty-backed SLAs  │                           │ • Automated churn trigger emails  │
   │ • Custom warehouse planograms     │                           │ • Minimum order tier thresholding │
   │ • Dedicated Key Account Managers  │                           │ • Real-time REST API integration  │
   └───────────────────────────────────┘                           └───────────────────────────────────┘
```

1. **Deploy Type 1 Interventions Only on Mathematically Proven Champions:**
   - Physical warehouse redesigns and multi-year SLAs require capital expenditure. Committing Type 1 capital to flawed historical models creates stranded assets.
   - Enforce rigorous BG/NBD ML scoring before authorizing VIP contract guarantees.

2. **Automate Type 2 Interventions via Real-Time API:**
   - Dynamic vouchers can be killed in under 60 seconds if incremental gross margin lift falls below targeted ROI.
   - Connect `clv_pipeline.py` outputs directly to the mobile checkout and CRM systems.

---

## 4. Deliverables Index & Verification Commands

All analyses are fully reproducible from source code:

```bash
# Execute Case Study 1
cd /Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w6_CustomerLifetimeValue/case1_cafe_clv
python3 generate.py && python3 analyze.py

# Execute Case Study 2 (Full Production)
cd /Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w6_CustomerLifetimeValue/case2_online_retail_clv
python3 ingest.py && python3 build_features.py && python3 clv_pipeline.py

# Execute Case Study 2 (Fast Smoke Test)
python3 generate_smoke_data.py && python3 ingest.py --smoke && python3 build_features.py && python3 clv_pipeline.py --smoke
```
