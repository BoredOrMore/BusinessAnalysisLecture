# Analytical Plan: Case Study 2 — Enterprise Online Retail Mining Project

## 1. Business Context & Objective
- **Stakeholders**: Chief Commercial Officer (CCO), VP of E-Commerce, Chief Financial Officer (CFO).
- **Dataset**: UCI Online Retail Transactional Dataset (500,000+ records spanning UK & international e-commerce).
- **Core Mission**: Build an end-to-end production Customer Lifetime Value (CLV) machine learning pipeline that filters raw log noise, infers latent churn risks ($P(\text{Alive})$), predicts 12-month forward net discounted cash flows, and quantifies EBITDA run-rate preservation via Type 1 and Type 2 interventions.

## 2. Data Hygiene Matrix (Slide 34)
| Rule ID | Constraint | Description & Impact |
| :--- | :--- | :--- |
| `RULE_01` | `CustomerID IS NOT NULL` | Eliminates anonymous guest checkouts where repeat behavior cannot be attributed. |
| `RULE_02` | `Quantity > 0` | Purges canceled orders, returns, and inventory damage logs. |
| `RULE_03` | `UnitPrice > 0.00` | Removes free promotional samples, zero-value adjust entries, and test SKUs. |

## 3. Mathematical & Modeling Pipeline (Slides 41-43)
1. **RFM-T Feature Engineering**:
   - Aggregate line items into customer-day transactions.
   - Calculate $x$ (repeat purchases), $x_{rec}$ (days between first and last purchase), $T$ (days between first purchase and observation cutoff), and $m_x$ (average spend on repeat purchases).
2. **Probabilistic Churn & Transaction Engine (BG/NBD)**:
   - Maximum likelihood estimation with L-BFGS-B optimization.
   - Infer individual purchase rate $\lambda \sim \text{Gamma}(r, \alpha)$ and dropout probability $p \sim \text{Beta}(a, b)$.
   - Predict $P(\text{Alive} \mid x, x_{rec}, T)$ and expected future 12-month transactions $E[X(365)]$.
3. **Monetary Spend Engine (Gamma-Gamma)**:
   - Fit $p, q, v$ parameters over repeat buyer distributions.
   - Infer expected monetary value $E[M \mid x, m_x]$.
4. **DCF Valuation & Strategic Action Segmentation**:
   - $\text{CLV}_{12M} = E[X(365)] \times E[M] \times \text{Gross Margin (30\%)} \times \text{DCF Discount Factor}$.
   - Classify into 4 Action Tiers:
     - **Tier 1: Champions (High CLV + High P(Alive))** -> Type 1 VIP Contracts & Dedicated Inventory Bays.
     - **Tier 2: At-Risk High-LTV (High CLV + Low P(Alive) / Latent Cliff)** -> Urgent Type 2 Dynamic Margin Rescue.
     - **Tier 3: Emerging Loyalists (Moderate CLV + High P(Alive))** -> Cross-sell & Basket Expansion.
     - **Tier 4: Dormant / Low-Value** -> Automated Low-Cost Digital Nudges.

## 4. Anti-Hallucination Guardrails
- Fixed seed `RANDOM_SEED = 42`.
- Exact data invariant assertions ($T \ge x_{rec}$, $x \ge 0$, $P(\text{Alive}) \in [0, 1]$).
- All figures, tables, and financial run-rate numbers generated strictly from script execution.
