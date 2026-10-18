# Lab 11C Report: Hybrid Recommendation and Validation

> ตัวเลขทุกตัวในรายงานนี้มาจากการรัน `python3 lab11C_hybrid_validation.py` (ผลเต็มอยู่ใน `evidence/results.json`) ยกเว้นเวลาและหน่วยความจำ ซึ่งเปลี่ยนเล็กน้อยทุกครั้งที่รัน

## TLDR

- **Factor model:** ลอง 9 ชุดค่า (d ∈ {2, 4, 8} x λ ∈ {0.01, 0.1, 1.0}) แล้วเลือกด้วย validation RMSE ได้ **d = 4, λ = 1.0** (RMSE 1.8412 จาก 17 ratings) ตัวอย่าง one-event update ในโจทย์ตรงกับโค้ดทุกค่า
- **Hybrid:** เลือก α ด้วย validation NDCG@5 ได้ **α = 1.0** คือใช้ content ล้วน ซึ่ง **ต่างจาก reference ของโจทย์ที่ได้ α = 0.75** เพราะผมไม่มี companion script จึง fit factor model ด้วย optimizer และ seed ของตัวเอง
- **Test:** ทั้ง 6 วิธีได้ **Precision@5, Recall@5 และ NDCG@5 = 0** บน 9 users ตรงกับที่โจทย์ระบุไว้ สิ่งที่ต่างกันมีแค่ coverage: popularity และ CF 0.071, factor 0.157, content และ hybrid 0.271
- **Rating error:** บน 18 test ratings ที่รู้จักทั้ง user และ item ค่า global mean (RMSE 1.264) **ชนะ** factor model (RMSE 1.384)
- **Audit:** ทุก object fit ด้วยข้อมูลก่อน 21 May ส่วน hyperparameters เลือกจาก validation และ candidate universe ไม่มีสินค้าจากอนาคต paired bootstrap บน test ได้ช่วง [0, 0] ทุกคู่ ซึ่ง 9 users น้อยเกินกว่าจะสรุปอะไรได้

**ความหมายทางธุรกิจ:** offline evaluation ไม่สามารถบอกได้ว่า hybrid ดีกว่า popularity สิ่งที่บอกได้มีแค่ hybrid กระจายการแนะนำได้กว้างกว่า ถ้าต้องการรู้ผลจริง ต้องทำ A/B test ที่ออกแบบไว้ในข้อ 3.3 และต้องเก็บ exposure log ก่อน

---

## Problem 1: Factor Model

### 1.1 Specify

| องค์ประกอบ | นิยาม |
|---|---|
| Prediction | r̂(u,i) = μ + b_u + b_i + p_u · q_i |
| Objective | J = ½ x mean ของ (r − r̂)² เฉพาะช่องที่มีค่าใน training + λ/2 x (‖P‖² + ‖Q‖² + ‖b_user‖² + ‖b_item‖²) |
| ขนาด | P = 71 x d, Q = 70 x d, b_user = 71, b_item = 70 |
| ค่าที่ fit จาก training เท่านั้น | μ, biases, P, Q, ID mappings |
| ค่าที่เลือกจาก validation | d, λ, α |
| Optimizer (ประกาศเอง) | SGD ทีละ event, 100 epochs, learning rate 0.05, init N(0, 0.1²), seed 42 |

**อธิบายแบบง่าย:** ลองนึกว่าสินค้าแต่ละชิ้นมีบุคลิกเป็นตัวเลขไม่กี่ตัว เช่น "ความเข้ม" และ "ความหวาน" ลูกค้าก็มีรสนิยมเป็นตัวเลขชุดเดียวกัน ถ้าทั้งสองชุดชี้ไปทางเดียวกัน (dot product สูง) ลูกค้าก็น่าจะชอบสินค้านั้น แต่ model ไม่ได้รู้ว่าแกนไหนคือ "ความเข้ม" แกนเหล่านี้หมุนได้โดยที่ผลทำนายไม่เปลี่ยน มิติที่ 1 จึงไม่ใช่หมวดสินค้าเสมอไป

### 1.2 Fit (เก็บผลครบทุกชุด)

| d | λ | Train objective | Val MAE | Val RMSE | Val n |
|---|---|---|---|---|---|
| 2 | 0.01 | 0.7417 | 1.7861 | 2.1590 | 17 |
| 2 | 0.10 | 6.3247 | 1.7440 | 2.0974 | 17 |
| 2 | 1.00 | 19.2976 | 1.5296 | 1.8416 | 17 |
| 4 | 0.01 | 0.7175 | 1.8160 | 2.1599 | 17 |
| 4 | 0.10 | 6.2229 | 1.7762 | 2.0911 | 17 |
| **4** | **1.00** | 19.3427 | 1.5310 | **1.8412** | 17 |
| 8 | 0.01 | 0.6960 | 1.7568 | 2.0917 | 17 |
| 8 | 0.10 | 6.0950 | 1.7246 | 2.0425 | 17 |
| 8 | 1.00 | 19.4628 | 1.5323 | 1.8428 | 17 |

เกณฑ์เลือกคือ validation RMSE ต่ำสุด ถ้าเสมอกันให้เลือก d ที่เล็กกว่าและ λ ที่ใหญ่กว่า λ ใหญ่ชนะทุกขนาด d ซึ่งสอดคล้องกับข้อมูลที่บางมาก (121 ratings) model ที่ถูก regularize หนักจะใกล้ค่าเฉลี่ย ซึ่งปลอดภัยกว่า ส่วน train objective ที่สูงตาม λ เกิดจาก penalty term ไม่ได้แปลว่า fit แย่

> โจทย์บอกให้ "run the supplied hyperparameter candidates" แต่ไม่มีรายการ candidates และไม่มี companion script ให้มา grid ข้างบนจึงเป็นค่าที่ผมประกาศเองก่อนดูผล validation

### 1.3 Verify

- คำนวณ score จาก state ที่บันทึกไว้ด้วยมือ ได้เท่ากับ `predict_mf` (3.509981)
- ขนาดของ P และ Q ถูกต้อง และคำทำนายทั้ง 4,970 คู่เป็นค่า finite ทั้งหมด (ช่วง 0.759 ถึง 5.109)
- **ตัวอย่าง one-event update ในโจทย์** (gradient ทุกตัวคำนวณจาก state เก่า):

| ค่า | ก่อน | หลัง |
|---|---|---|
| dot product, prediction, residual | 0.14, 3.24, 0.76 | |
| b_u | 0.2 | 0.237 |
| b_i | −0.1 | −0.0615 |
| p_u | [0.2, 0.4] | [0.218, 0.4018] |
| q_i | [0.5, 0.1] | [0.5051, 0.1147] |

- **Fallback:** ถ้าไม่รู้จัก user หรือ item ให้ทำนาย rating ด้วย μ และห้ามสร้าง factor แบบสุ่มขึ้นมาให้ดูเหมือนเคย train แล้ว ส่วนการจัดอันดับ ถ้าเป็น user ที่ไม่รู้จัก ให้ใช้ popularity list

**Implicit feedback (บริบท):** การที่ลูกค้าไม่คลิกหรือไม่ซื้อ ไม่ได้แปลว่าไม่ชอบ วิธีของ Hu et al. (2008) ให้ทุกช่องมี preference และ confidence แต่ confidence เป็นสิ่งที่ model สมมติขึ้น ไม่ใช่ exposure ที่เก็บได้จริง lab นี้ใช้ explicit ratings ตาม reference

---

## Problem 2: Hybrid

### 2.1 Blend

- **Calibration:** ใช้ min-max ต่อ request บน candidates ของ user คนนั้น ถ้า component ไหนมีค่าเท่ากันหมด ให้เป็น 0 ทั้งหมด และไม่ใช้ label ในการ scale
- **Routing:** user ที่ไม่รู้จักใช้ popularity ส่วน user ที่ไม่มี positive profile ให้ content = 0 แล้วใช้ factor อย่างเดียว

| α | Val NDCG@5 | Val Recall@5 | Val coverage |
|---|---|---|---|
| 0.00 (factor ล้วน) | 0.0431 | 0.1 | 0.1429 |
| 0.25 | 0.0431 | 0.1 | 0.2000 |
| 0.50 | 0.0431 | 0.1 | 0.2429 |
| 0.75 | 0.0431 | 0.1 | 0.2429 |
| **1.00 (content ล้วน)** | **0.0931** | 0.2 | 0.2143 |

ผลที่ได้คือ α = 1.0 ต่างจาก reference (α = 0.75) ความต่างระหว่าง 0.0431 กับ 0.0931 มาจาก user เพียง **1 คน** ที่ hit ได้หรือไม่ได้ จึงเป็นการเลือกที่ไม่มั่นคง

**ตัวอย่างหนึ่ง score (user 2, item 538):** 1.0 x content 1.0000 + 0.0 x factor 0.4591 = 1.0000

### 2.2 Rank trace (user 2)

Catalog 70 ชิ้น ลบสินค้าที่เคยซื้อ 4 ชิ้น (508, 516, 540, 552) เหลือ 66 candidates เมื่อ score เสมอกัน ให้เรียงตาม popularity rank แล้วตาม ProductID ได้ Top-5 = 538, 598, 553, 503, 533 ระบบใช้ `assert` ตรวจว่า ID ไม่ซ้ำ ทุกชิ้นอยู่ใน eligible set และรายการยาวไม่เกิน 5 (ดูรายละเอียดใน `evidence/hybrid_rank_trace.csv`)

### 2.3 Compare (test เปิดครั้งเดียว หลังล็อกค่าทั้งหมดแล้ว)

| Method | Users | P@5 | R@5 | NDCG@5 | Coverage | Diversity | Warm (NDCG) | Cold (NDCG) |
|---|---|---|---|---|---|---|---|---|
| popularity | 9 | 0 | 0 | 0 | 0.071429 | 0.7675 | 7 (0) | 2 (0) |
| content | 9 | 0 | 0 | 0 | 0.271429 | 0.4918 | 7 (0) | 2 (0) |
| user_cf | 9 | 0 | 0 | 0 | 0.071429 | 0.7675 | 7 (0) | 2 (0) |
| item_cf | 9 | 0 | 0 | 0 | 0.071429 | 0.7675 | 7 (0) | 2 (0) |
| factor | 9 | 0 | 0 | 0 | 0.157143 | 0.7564 | 7 (0) | 2 (0) |
| hybrid (α = 1) | 9 | 0 | 0 | 0 | 0.271429 | 0.4918 | 7 (0) | 2 (0) |

**เทียบกับ reference ในโจทย์:** popularity, CF และ content ได้ coverage ตรงกันทุกตัว (0.071429 และ 0.271429) ส่วน factor (0.157 เทียบกับ 0.171) และ hybrid (0.271 เทียบกับ 0.314) ต่างไป เพราะ factor fit และ α ต่างจาก reference ค่า coverage เหล่านี้ไม่ได้พิสูจน์ว่าวิธีไหนจัดอันดับได้ดีกว่า

**Diversity** = ค่าเฉลี่ยของ 1 − content cosine ระหว่างคู่สินค้าใน list content ได้ diversity ต่ำที่สุด ซึ่งตรงกับธรรมชาติของมัน คือแนะนำของที่คล้ายกัน feature ที่ใช้วัดเป็น synthetic

**Rating errors** (18 test ratings ที่รู้จักทั้ง user และ item ส่วน top-K ใช้ 9 users จำนวนจึงไม่เท่ากัน)

| Predictor | MAE | RMSE |
|---|---|---|
| global mean | 1.0069 | **1.2640** |
| factor | 1.0871 | 1.3840 |
| user mean / user-CF / item-CF | 1.2546 | 1.5175 |

CF ได้ค่าเท่ากับ user mean ทุกตัว เพราะ fallback ทั้งหมด

**Exclusions:** 17 test events เป็นสินค้าที่ไม่เคยเห็นใน training (item cold start) แยกจาก 7 cold test users (user cold start) และนี่คือ **static evaluation** คือใช้ model ที่ train ก่อน 21 May ตลอด ไม่มีการ retrain ในวันที่ 13 June

---

## Problem 3: Audit และ Uncertainty

### 3.1 Audit

| Object | Fit บน | Event ล่าสุดที่ใช้ | ผ่าน boundary |
|---|---|---|---|
| popularity counts, means, mappings + catalog, vocabulary + IDF, profiles, CF neighbourhoods, factors | train | 2024-05-20 18:42:23 UTC | ✓ (< 21 May) |
| hyperparameters (d, λ, α) | valid | 2024-06-12 15:54:30 UTC | ✓ (< 13 June) |

- Candidate universe ประกอบด้วย 70 items จาก training และตัดสินค้าที่มีเฉพาะในอนาคต 20 ชิ้นออกทั้งหมด
- ใช้ full ranking ทั้ง catalog โดยไม่ใช้ negative sampling
- ทดสอบ **failing assertion:** สร้างสำเนาที่ fit บน train + valid แล้วระบบหยุดด้วย `AssertionError: Fit uses events on/after the training boundary`
- บันทึกค่าที่ประกาศไว้ทั้งหมดใน `evidence/declared_settings.json` ก่อนเปิดผล test

### 3.2 Quantify

- **Per-user NDCG@5 บน test:** ทั้ง 9 users ได้ 0 ในทุกวิธี
- **Paired user bootstrap** (2,000 รอบ, seed 42, สุ่ม user แบบทั้งก้อน):

| เปรียบเทียบ | Split | Users | Mean diff | 95% interval |
|---|---|---|---|---|
| hybrid − popularity | test | 9 | 0.00 | [0.00, 0.00] |
| factor − popularity | test | 9 | 0.00 | [0.00, 0.00] |
| content − popularity | test | 9 | 0.00 | [0.00, 0.00] |
| hybrid − popularity | valid | 10 | 0.00 | [0.00, 0.00] |
| factor − popularity | valid | 10 | −0.05 | [−0.15, 0.00] |
| content − popularity | valid | 10 | 0.00 | [0.00, 0.00] |

ช่วง [0, 0] **ไม่ได้แปลว่า "แน่นอนว่าเท่ากัน"** แต่แปลว่าไม่มีความแปรปรวนให้วัด เพราะทุกคนได้ 0 percentile interval ที่คิดจาก 9 users ไม่นิ่ง และผมไม่นับตำแหน่งทั้ง 5 ใน list หรือ rating ซ้ำเป็นตัวอย่างอิสระ

- **Resources:** refit model ที่เลือกแล้วและให้คะแนน 32 test users ใช้เวลาประมาณ 0.5 วินาที peak traced memory ประมาณ 0.04 MiB (Python 3.13, Linux, 121 training events) ค่าที่รันจริงอยู่ใน `results.json`

### 3.3 Pre-registered A/B test

| องค์ประกอบ | การออกแบบ |
|---|---|
| Unit | user (sticky hash ของ pseudonymous UserID) แบ่ง 50/50 |
| Control / Treatment | popularity slot / hybrid slot |
| Primary outcome | purchase conversion ของสินค้าที่แนะนำ ต่อ user ที่เห็นรายการแนะนำ |
| Guardrails | revenue per user, page latency p95, complaint/return rate, catalog coverage |
| Baseline (สมมติฐาน) | 5.00% ต้องแทนด้วยค่าจริงจาก log |
| MDE | ดีขึ้น 10% แบบ relative (5.00% เป็น 5.50%) |
| α / power | 0.05 two-sided / 0.80 |
| Sample size | **31,231 users ต่อกลุ่ม** จากสูตร two-proportion: n = (z₀.₉₇₅ + z₀.₈)² [p₀(1−p₀) + p₁(1−p₁)] / (p₁ − p₀)² |
| Duration | 14 วัน ที่ 5,000 eligible users ต่อวัน (สมมติฐาน) ปัดเป็นสัปดาห์เต็มและครอบคลุมอย่างน้อย 2 รอบสัปดาห์ |
| Stopping rule | fixed horizon วิเคราะห์ครั้งเดียวตอนจบ ห้ามแอบดูผลกลางทาง |

**ข้อมูลที่ต้องเก็บ (exposure log):** user, arm, model version, รายการสินค้าที่แสดงพร้อมตำแหน่ง, เวลา, การคลิกและการซื้อ เหตุผลคือข้อมูลชุดปัจจุบันไม่รู้ว่าลูกค้า "เห็น" อะไรบ้าง จึงแยก "ไม่ชอบ" ออกจาก "ไม่เคยเห็น" ไม่ได้

**Privacy review (PDPA พ.ร.บ.คุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562):** ต้องระบุฐานทางกฎหมาย (lawful basis) และแจ้งในประกาศความเป็นส่วนตัว (privacy notice), ใช้ pseudonymous ID, เก็บข้อมูลเท่าที่จำเป็น (data minimization), กำหนดระยะเวลาเก็บ (retention), ห้ามใช้ข้อมูลอ่อนไหวหรือ demographic ตามสัญญาข้อมูลของ Lab 11A และควรผ่านการตรวจของ DPO ก่อนเริ่ม

ค่า baseline, MDE และ daily users เป็นสมมติฐานเพื่อสาธิตเท่านั้น ปรับได้ผ่าน `argparse` (`--baseline_rate`, `--mde_relative`, `--daily_users`)

---

## Methodological Defense

| ประเด็น | คำอธิบาย |
|---|---|
| **ทำไมผลเป็น 0** | test มีแค่ 9 users และแต่ละคนมี relevant item แค่ 1 ชิ้นในสินค้า 66 ถึง 70 ชิ้น โอกาสสุ่มถูกใน Top-5 จึงต่ำมาก ผลนี้ไม่ใช่ bug และ **ห้ามเปลี่ยน split หรือ seed เพื่อซ่อนผล** |
| **Scale ของ hybrid** | rating (1 ถึง 5) กับ cosine (0 ถึง 1) อยู่คนละ scale ต้องแปลงก่อนผสม min-max ต่อ request เป็นวิธีง่ายแต่ไวต่อ outlier ทางเลือกอื่นเช่น rank-based fusion หรือ learning-to-rank |
| **Selection บนข้อมูลเล็ก** | ลอง 9 + 5 ชุดค่าบน validation 10 ถึง 17 ตัวอย่าง ทำให้เสี่ยงเลือกได้ค่าที่ "โชคดี" ผมจึงเก็บผลทุกชุดไว้ และเปิด test เพียงครั้งเดียว |
| **Regularization ชนะ** | λ = 1.0 ชนะทุก d และ global mean ชนะ factor บน test ข้อมูลนี้บางเกินกว่าจะเรียน latent factors ได้ ซึ่งตรงกับข้อค้นพบที่ว่า MF ต้องการ interactions ต่อ user มากพอ (Koren et al., 2009) |
| **Offline ไม่เท่ากับ online** | offline metric วัดได้แค่สิ่งที่ลูกค้าเลือกเองภายใต้ระบบเดิม (selection bias) A/B test ที่เก็บ exposure log เป็นวิธีเดียวที่วัดผลเชิงเหตุผล (causal) ได้ (Kohavi et al., 2020) |
| **Uncertainty** | ใช้ user เป็นหน่วย bootstrap และใช้ paired design เพื่อลดความแปรปรวน แต่ 9 users ยังน้อยเกินไปสำหรับ percentile interval ที่น่าเชื่อถือ |

## Key Points สำหรับมือใหม่

1. **MF = บีบรสนิยมเป็นตัวเลขไม่กี่ตัว** ต้องมีข้อมูลพอ ถ้าข้อมูลน้อย ค่าเฉลี่ยธรรมดาอาจชนะได้
2. **Regularization (λ)** กัน model จำข้อมูลแทนที่จะเรียนรู้ ยิ่งข้อมูลน้อย ยิ่งต้องใช้มาก
3. **ห้ามผสมคะแนนต่าง scale** ต้องแปลงให้เป็น scale เดียวกันก่อน
4. **เลือกค่าด้วย validation แล้วเปิด test ครั้งเดียว** และต้องเก็บผลทุกชุดที่ลอง
5. **Bootstrap ต้องสุ่มที่ระดับ user** ไม่ใช่ระดับตำแหน่งในรายการ
6. **ความจริงอยู่ที่ A/B test** ต้องออกแบบ sample size และ stopping rule ไว้ก่อน และเก็บ exposure log เสมอ

## หมายเหตุ

- ไม่มี companion script และรายการ hyperparameter candidates ของอาจารย์ grid, optimizer และ seed ในรายงานนี้จึงเป็นค่าที่ผมประกาศเองก่อนเปิดผล validation
- ผลที่ต่างจาก reference มี α (1.0 เทียบกับ 0.75), factor coverage (0.157 เทียบกับ 0.171) และ hybrid coverage (0.271 เทียบกับ 0.314) ส่วนผลที่ตรงกับ reference มี test metrics ที่เป็น 0 ทั้ง 6 วิธีบน 9 users และ coverage ของ popularity, CF และ content

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8). https://doi.org/10.1109/MC.2009.263
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. *ICDM 2008*. https://doi.org/10.1109/ICDM.2008.22
- Krichene, W., & Rendle, S. (2020). On Sampled Metrics for Item Recommendation. *KDD 2020*.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Personal Data Protection Act B.E. 2562 (2019), Thailand; Personal Data Protection Committee guidance. https://www.pdpc.or.th/
