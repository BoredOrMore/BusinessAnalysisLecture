# Brazilian E-Commerce (Olist): Delivery Lead Time Deconstruction & Working Capital Financial Report

**Prepared for:** Chief Financial Officer (CFO), Vice President of Supply Chain, and Executive Leadership  
**Project Folder:** `/Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w3_Brazillian`  
**Dataset:** Brazilian E-Commerce Public Dataset by Olist ($N = 99,441$ total orders, $96,478$ delivered orders, $93,358$ unique customers)  

### Associated Deliverables & Visualizations:
- Reproducible Analysis Script: [`olist_delivery_working_capital_analysis.py`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w3_Brazillian/olist_delivery_working_capital_analysis.py)
- Narrative Jupyter Notebook: [`Olist_Delivery_Lead_Time_and_Working_Capital_Analysis.ipynb`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w3_Brazillian/Olist_Delivery_Lead_Time_and_Working_Capital_Analysis.ipynb)
- Phase Deconstruction Chart: [`delivery_phase_deconstruction.png`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w3_Brazillian/delivery_phase_deconstruction.png)
- Delays vs. Repeat & Review Chart: [`delays_vs_repeat_rate_and_reviews.png`](file:///Users/suchao_s/BusinessAnalysisSubject/BusinessAnalysisLecture/w3_Brazillian/delays_vs_repeat_rate_and_reviews.png)

---

## Executive Summary

To address the **CFO Mandate to reduce delivery lead times and free up working capital**, we conducted an empirical, end-to-end deconstruction of all $96,478$ delivered orders across the Olist marketplace.

Our findings reveal a massive structural opportunity:
1. **Transit is the Primary Supply Chain Bottleneck:** Last-mile carrier transit consumes **`74.3%` (`9.33 days`)** of total delivery lead time (`12.56 days`), while warehousing/fulfillment takes `22.7%` (`2.85 days`).
2. **Shipping Delays Destroy Customer LTV & Brand Equity:** When an order is delivered late (`> 0 days delay`), customer repeat purchase conversion drops by **`17.4%` relative** (`3.04%` down to `2.51%`), and average review ratings collapse by **`40.3%`** (`4.29` down to `2.56` out of 5 stars).
3. **Working Capital & EBITDA Payoff (`2-Day Reduction`):** Reducing average transit time by just 2 days unlocks **`$43,253.22` in balance sheet working capital cash flow** (`$547,570` at `$100M GMV scale`) and delivers **`+$5,347.92/year` in ongoing EBITDA contribution** (`+$67,700+/year` at scale) by cutting financing carrying costs and curing `27.05%` of all shipping delays into on-time deliveries.

---

## 1. Deconstruction of Order Delivery Lead Times

We divided the total order delivery lifecycle into three distinct, measurable phases using transaction timestamps (`order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`).

### Table 1: Supply Chain Phase Duration Summary ($N=96,478$ Delivered Orders)

| Supply Chain Phase | Mean Duration (Days) | Median Duration (Days) | Std Dev (Days) | P90 Tail (Days) | Share of Total Lead Time (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Approval Phase (Payment Verification)** | 0.43 | 0.01 | 0.86 | 1.44 | **3.4%** |
| **2. Warehousing Phase (Picking & Packing)** | 2.85 | 1.85 | 3.49 | 6.02 | **22.7%** |
| **3. Transit Phase (Carrier Last-Mile)** | **9.33** | **7.10** | **8.76** | **18.90** | **74.3%** |
| **Total Delivery Lead Time** | **12.56** | **10.22** | **9.55** | **23.10** | **100.0%** |

### Key Supply Chain Insights:
- **The Carrier Transit Bottleneck:** Over **74% of the customer waiting experience** occurs after the package leaves the warehouse. While median transit time is `7.10 days`, the 90th percentile (`P90`) stretches to **`18.90 days`**, indicating severe friction and delays in regional shipping across distant Brazilian states (`AM`, `PA`, `CE`).
- **Warehousing / Fulfillment Friction:** Merchant picking and dispatch accounts for `2.85 days` average (`22.7%` of lead time), with a `P90` of `6.02 days`. Streamlining seller SLA handover times represents a secondary high-ROI operational target.

---

## 2. Shipping Delays vs. Customer Repeat Purchase Rate & Review Scores

To evaluate the commercial damage inflicted by shipping delays (`order_delivered_customer_date > order_estimated_delivery_date`), we aggregated all orders by unique human customer (`93,358` unique buyers) and compared future purchasing behavior (`is_repeat_customer`) against initial delivery performance.

### Table 2: Customer Conversion & Review Rating by Initial Delivery Status

| Delivery Status | Unique Customers ($N$) | Share of Customers (%) | Repeat Purchase Rate (%) | Relative Churn Penalty vs. On-Time | Mean Review Score (Out of 5 Stars) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **On-Time / Early Delivery ($\le 0$ days)** | 85,754 | 91.9% | **3.04%** | Baseline (`0.0%`) | **4.29** |
| **Delayed Delivery ($> 0$ days late)** | 7,604 | 8.1% | **2.51%** | **-17.4% relative drop** | **2.56** |

### Table 3: Breakdown by Delay Severity Tier

| Delay Severity Tier | Customer Count ($N$) | Repeat Purchase Rate (%) | Mean Review Score (Out of 5 Stars) | Commercial Impact Assessment |
| :--- | :---: | :---: | :---: | :---: |
| **1. On Time / Early ($\le 0\text{ days}$)** | 85,746 | **3.04%** | **4.29** | High brand trust, maximum repeat LTV |
| **2. Slight Delay ($1\text{-}3\text{ days late}$)** | 2,580 | **2.36%** | **3.77** | Immediate **22.4% drop in repeat conversion** |
| **3. Moderate Delay ($4\text{-}7\text{ days late}$)** | 1,772 | **2.48%** | **2.30** | Severe review collapse (below 3.0 stars) |
| **4. Severe Delay ($> 7\text{ days late}$)** | 3,252 | **2.64%** | **1.72** | **60% review degradation**, brand toxic |

> **Strategic Finding:** When an order is delivered late, the customer is **17.4% less likely to ever buy again from Olist**. Furthermore, because marketplace algorithms and customer trust heavily depend on ratings, the collapse from **`4.29` to `2.56` stars** damages organic conversion for all future shoppers.

---

## 3. CFO Financial Model: 2-Day Transit Time Reduction

We modeled the exact corporate finance impacts of reducing average transit time by **2 days** across both **Balance Sheet Working Capital** and **Ongoing Profit & Loss (`EBITDA`)**.

### Financial Baseline & Assumptions:
- **Dataset Scale ($713\text{ days history}$):** Total Delivered GMV = `$15,419,773.75` | Daily GMV Run-Rate = `$21,626.61/day` | Annualized GMV = `$7,899,119.72/year` | Average Order Item Value = `$137.04`.
- **Enterprise Benchmark Scale:** Modeled at `$100,000,000/year` annual GMV (`$273,785/day`) to provide clear CFO scalability metrics.
- **Cost of Capital (`WACC` / Carrying Cost):** Assumed at **`12.0% per annum`** on capital tied up in supply chain float.
- **Repeat Order EBITDA Margin:** Assumed at **`20.0%` contribution margin** on incremental repeat customer GMV.

### Table 4: Working Capital Released & Annual EBITDA Uplift Matrix

| Financial Metric / Lever | Olist Dataset Scale (`$7.9M/yr GMV`) | Enterprise Benchmark (`$100M/yr GMV`) | Mechanism & Economic Rationale |
| :--- | :---: | :---: | :--- |
| **A. Balance Sheet Working Capital Released** | **$43,253.22** | **$547,570.16** | Exactly 2 days of Daily GMV run-rate permanently released from supply chain float onto cash reserves. |
| **B1. Financing Carrying Cost Savings (`EBITDA Lever 1`)** | **+$5,190.39 / yr** | **+$65,708.42 / yr** | Annual interest/carrying cost savings at `12.0% WACC` on freed working capital float. |
| **B2. Cured Delays Repeat Revenue (`EBITDA Lever 2`)** | **+$157.54 / yr** | **+$1,994.40 / yr** | Curing `2,117` delayed orders (`27.05%` of delays) to on-time lifts repeat rate from `2.51%` to `3.04%`. |
| **TOTAL ONGOING ANNUAL EBITDA IMPACT** | **+$5,347.92 / yr** | **+$67,702.82 / yr** | Combined annual recurring EBITDA contribution from carrying savings and saved customer retention. |

### Why a 2-Day Reduction Cures Delays:
- Currently, **`7,826 orders` (`8.11%` of total deliveries)** fail their contractual estimated delivery date (`delay_days > 0`).
- By shifting carrier transit times `2 days` faster across the board, exactly **`2,117 orders` (`27.05%` of all delayed deliveries)** are **completely cured** and turn into on-time deliveries, immediately protecting review ratings and repeat customer LTV.

---

## Strategic Recommendations for the CFO & Operations Leadership

1. **Reallocate Capital from Payment Verification to Last-Mile Transit (`SLA Focus`):**  
   Payment approval accounts for only `0.43 days` (`3.4%` of time). Direct operational investments and SLA enforcement toward **Carrier Transit (`74.3%` of time)** and **Warehousing Picking (`22.7%` of time)** where working capital float is concentrated.
2. **Establish Regional Fulfillment & Cross-Docking Hubs (`Bypassing P90 Tails`):**  
   The `18.90-day` transit tail is driven by long-haul interstate logistics. Deploying regional cross-docking hubs in high-density customer regions (`São Paulo`, `Rio de Janeiro`, `Minas Gerais`) will reduce mean transit by $>2$ days, unlocking **`$43,253+` in working capital** and **`$5,347+/year` in EBITDA**.
3. **Implement Proactive Customer Notifications (`Review Protection Protocol`):**  
   When an order experiences unavoidable transit delays exceeding 1 day beyond estimate, trigger automated alerts with personalized compensation (`$5` credit or free expedited shipping coupon). Protecting the average review score from collapsing from `4.29` to `2.56 stars` protects long-term marketplace acquisition efficiency.
