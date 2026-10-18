# Lab 11A Report: Recommendation System Foundations

> ตัวเลขทุกตัวในรายงานนี้มาจากการรัน `python3 lab11A_foundations.py` (ผลเต็มอยู่ใน `evidence/results.json`)

## TLDR

- ข้อมูล 200 events ผ่านการตรวจครบ: ไม่มีค่าว่าง ไม่มีคู่ (UserID, ProductID) ซ้ำ และ Rating เป็นจำนวนเต็ม 1 ถึง 5
- แบ่งตามเวลาได้ train 121 / validation 38 / test 41 events ตรงกับตัวเลขในโจทย์
- Training matrix มีขนาด 71 users x 70 items = 4,970 ช่อง แต่มีคะแนนจริงแค่ 121 ช่อง (density 0.024346) ช่องที่เหลือคือ "ไม่รู้" ไม่ใช่ 0
- Popularity baseline (K = 5) ได้ **Precision@5 = 0.04, Recall@5 = 0.20, NDCG@5 = 0.0931 บน validation (10 users)** และ **0 ทุก metric บน test (9 users)**
- ตัวเลข test ครอบคลุมแค่ 9 users เท่านั้น เพราะ 17 จาก 41 test events เป็นสินค้าที่ไม่เคยเห็นใน training จึงแนะนำและวัดผลไม่ได้

**ความหมายทางธุรกิจ:** list "สินค้าขายดี" แทบไม่ช่วยลูกค้ากลุ่มนี้ และข้อมูลบางมากจนต้องระวังการอ่านตัวเลข Lab 11B และ 11C ต้องชนะ baseline นี้ภายใต้กติกาเดียวกัน จึงจะนับว่าดีขึ้นจริง

---

## Problem 1: Inspect, Validate, Partition

### 1.1 Inspect

| รายการ | ค่า |
|---|---|
| ไฟล์ | `data/interaction_events.csv` |
| SHA256 | `f55471cadb4740a4e15187bea6b8a7b8143f7c9ac3c764269031348fcfbcc079` |
| ขนาด | 200 events, 87 users, 90 products, 4 columns |
| ช่วงเวลา | 2024-03-09 19:14:56 UTC ถึง 2024-07-03 03:17:47 UTC |
| ตัวอย่างการแปลงเวลา | event แรก: `1713914080` วินาที = 2024-04-23 23:14:40 UTC |
| Rating frequencies | 1: 47 (23.5%), 2: 40 (20.0%), 3: 35 (17.5%), 4: 40 (20.0%), 5: 38 (19.0%) |

**ทำไม UserID และ ProductID เป็น label:** ID เป็นแค่ป้ายชื่อ เหมือนเบอร์เสื้อนักฟุตบอล การบวกหรือเฉลี่ยมันจึงไม่มีความหมาย ถ้าเก็บเป็นตัวเลขจะเสี่ยงสองอย่าง คือโปรแกรมอาจตัดเลข 0 ข้างหน้าทิ้ง หรือ model อาจเข้าใจผิดว่า ID 600 "มากกว่า" ID 500 ผมจึงอ่านเป็น `string` ตั้งแต่ตอนโหลดไฟล์

### 1.2 Validate

ฟังก์ชัน `validate_events()` ตรวจข้อมูลด้วย `assert` แต่ละข้อดังนี้: ต้องมีครบ 4 columns, ID ห้ามว่าง, Rating ต้องเป็นจำนวนเต็มที่ finite และอยู่ใน [1, 5], Timestamp ต้องถูกต้อง, และคู่ (UserID, ProductID) ห้ามซ้ำ

ผมทดสอบโดยสร้าง **สำเนา** ที่ใส่ Rating = 7 ลงไป ผลคือระบบหยุดทำงานด้วย `AssertionError: Rating outside [1,5]: [7]` ส่วนข้อมูลต้นฉบับไม่ถูกแก้ไข

**ทำไมต้องหยุดทันทีแทนที่จะแก้เงียบๆ:** ถ้าระบบแอบเปลี่ยน 7 เป็น 5 ก็เท่ากับแต่งข้อมูลขึ้นมาเอง กรณี rating ซ้ำก็เหมือนกัน ถ้าจะนำเข้า matrix ต้องมีกติกาก่อนว่าจะใช้ค่าล่าสุดหรือค่าเฉลี่ย ห้ามเลือกเอาเอง

### 1.3 Partition

| Partition | ช่วงเวลา | Events | Users | Items | Min UTC | Max UTC |
|---|---|---|---|---|---|---|
| train | ก่อน 21 May 2024 | 121 | 71 | 70 | 2024-03-09 19:14:56 | 2024-05-20 18:42:23 |
| validation | 21 May ถึงก่อน 13 June | 38 | 30 | 32 | 2024-05-21 03:34:19 | 2024-06-12 15:54:30 |
| test | ตั้งแต่ 13 June | 41 | 32 | 37 | 2024-06-13 00:30:36 | 2024-07-03 03:17:47 |

**ทำไม model ต้อง frozen ทั้งช่วง validation และ test:** ร้านค้าตัดสินใจแนะนำสินค้าในวันที่ 21 May ด้วยข้อมูลเท่าที่มี ณ วันนั้นเท่านั้น ถ้าปล่อยให้ model เรียนจากข้อมูลหลังวันนั้น หรือเลือก parameter โดยดูคะแนน test ตัวเลขที่ได้จะดีเกินจริง ปัญหานี้เรียกว่า **data leakage** งานวิจัยของ Ji et al. (2023) พบว่าการแบ่งข้อมูลที่ไม่เคารพลำดับเวลาทำให้อันดับของ model เปลี่ยนไปได้จริง

---

## Problem 2: Representation

### 2.1 Map

ผมเรียง ID ที่ไม่ซ้ำของ training แล้วสร้าง dictionary แปลง ID เป็น index (`user_mapping.csv`, `item_mapping.csv`) จากนั้นใช้ `assert` ตรวจว่าทุก coordinate แปลงกลับเป็น ID เดิมได้ครบ ผลคือ U = 71 และ I = 70

### 2.2 Represent

| รายการ | ค่า |
|---|---|
| Observations ที่เก็บจริง (E) | 121 ค่า เก็บเป็น arrays `row`, `col`, `rating`, `observed` ที่ยาวเท่ากัน |
| Matrix เชิงแนวคิด | 71 x 70 = 4,970 ช่อง |
| Density | 121 / 4,970 = **0.024346** |
| Sparsity | **0.975654** |

รูป `figures/masked_submatrix.png` แสดงตาราง 8 x 8 ช่องสีเทาคือ **unknown** แยกให้เห็นชัดจาก rating ต่ำ (เช่น 1)

**อธิบายแบบง่าย:** การเก็บแบบ sparse (COO) คือจดเฉพาะช่องที่มีค่าจริง เหมือนสมุดโทรศัพท์ที่จดเบอร์ไว้แค่บางคน ถ้าเติม 0 ในช่องที่ไม่รู้ ก็เท่ากับกุหลักฐานว่าลูกค้า "ให้ 0 คะแนน" ทั้งที่ลูกค้าอาจไม่เคยเห็นสินค้าชิ้นนั้นเลย

### 2.3 Diagnose

- Interactions ต่อ user: เฉลี่ย 1.70, median 1, สูงสุด 4
- Interactions ต่อ item: เฉลี่ย 1.73, median 1, สูงสุด 5

| Holdout | Events | Events ของ cold user | Cold users | Events ของ unseen item | Unseen items |
|---|---|---|---|---|---|
| validation | 38 | 18 | 12 | 9 | 9 |
| test | 41 | 11 | 7 | 17 | 14 |

| ประเภท | ความหมาย | Fallback |
|---|---|---|
| **Cold user** | ลูกค้าที่ไม่มีแถวใน training | ให้ popularity list เดียวกับทุกคน (ไม่สร้างแถวหรือ factor ขึ้นมาเอง) |
| **Cold item** | สินค้าที่ไม่อยู่ใน catalog 70 ชิ้นตอน train | แนะนำไม่ได้และไม่นับเป็น positive โดยรายงานจำนวนแยกไว้ |

---

## Problem 3: Popularity Baseline

### 3.1 Rank

นับจำนวน interaction ต่อสินค้าใน training แล้วเรียงจากมากไปน้อย ถ้าเสมอกันให้เรียงตาม ProductID แบบตัวอักษร Top 6 คือ 540 (5 ครั้ง) ตามด้วย 524, 546, 556, 557, 588 (ชิ้นละ 4 ครั้ง)

| ตัวอย่าง | ประวัติใน training | Top-5 ที่ได้ |
|---|---|---|
| Warm user `2` | 508, 516, 540, 552 (ตัด 540 ออก) | 524, 546, 556, 557, 588 |
| Cold user `3` | ไม่มี | 540, 524, 546, 556, 557 |

ตัวอย่างนี้แสดงการตัดสินค้าที่เคยซื้อออก และแสดงว่าการเรียงลำดับเมื่อเสมอกันให้ผลเหมือนเดิมทุกครั้ง (deterministic)

### 3.2 Calculate

**Worked example (ตรงกับโจทย์ทุกค่า):** list E1 ถึง E5 และ relevant = {E1, E3, E4, E6}

| Metric | สูตรแบบง่าย | ค่า |
|---|---|---|
| Precision@5 | ถูก 3 จากที่แนะนำ 5 | 0.600000 |
| Recall@5 | เจอ 3 จากของที่ชอบจริง 4 | 0.750000 |
| DCG@5 | 1/log2(2) + 1/log2(4) + 1/log2(5) | 1.930677 |
| IDCG@5 | กรณีที่ดีที่สุด คือของที่ชอบทั้ง 4 ชิ้นอยู่อันดับ 1 ถึง 4 | 2.561606 |
| NDCG@5 | DCG / IDCG | 0.753698 |

**ผล Popularity baseline**

| Holdout | Users ที่วัดผล | ในนั้นเป็น cold users | Users ที่ถูกตัดออก | Precision@5 | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|---|
| validation | 10 | 6 | 20 | 0.0400 | 0.2000 | 0.0931 |
| test | 9 | 2 | 23 | 0.0000 | 0.0000 | 0.0000 |

ใน validation มีแค่ 2 users ที่ทายถูก คือ user 28 (hit ที่อันดับ 1, NDCG = 0.5) และ user 88 (hit ที่อันดับ 3, NDCG = 0.4307) ใน test ทายไม่ถูกเลยสักคน รายละเอียดรายคนอยู่ใน `evidence/popularity_per_user_*.csv`

### 3.3 Qualify

ใน test มี 41 events แต่ **17 events เป็นสินค้าที่ไม่เคยเห็นใน training** และเหลือ users ที่วัดผลได้ **9 คน (เป็น cold users 2 คน)**

**ทำไมตัวเลขนี้แทน 41 events ทั้งหมดไม่ได้:** metric นับเฉพาะ positive (Rating ≥ 4) ที่อยู่บนชั้นวาง 70 ชิ้นและลูกค้ายังไม่เคยซื้อ มี 3 กลุ่มที่อยู่นอกตัวหารทั้งหมด ได้แก่ (1) สินค้าใหม่ 17 events, (2) events ที่ได้คะแนนต่ำ และ (3) users ที่ไม่มี positive เลย ดังนั้นการบอกว่า "model แม่น 0%" หรือ "แม่น 20%" จึงหมายถึงแค่ประชากรกลุ่มเล็กนี้เท่านั้น และไม่ใช่ click-through rate จริงบนเว็บ

---

## Methodological Defense

| ประเด็น | การตัดสินใจ | เหตุผลและหลักฐานทางวิชาการ |
|---|---|---|
| **Missingness** | ช่องที่ไม่มีข้อมูล = unknown ห้ามเติม 0 | Rating ที่ "หายไป" ไม่ได้หายแบบสุ่ม (missing not at random) คนมักให้คะแนนเฉพาะของที่ตัวเองเลือกเอง การเติม 0 จึงสร้าง bias (Marlin & Zemel, 2009; Steck, 2010) |
| **Temporal split** | แบ่งตามเวลาและใช้ cutoff ตายตัว | การสุ่มแบ่งข้อมูลทำให้ model ได้เห็นอนาคต จึงได้คะแนนสูงเกินจริงและอันดับ model เปลี่ยน (Meng et al., 2020; Ji et al., 2023) |
| **Catalog eligibility** | catalog คือ 70 items จาก training และห้ามขยายด้วยข้อมูลอนาคต | ณ วันที่ 21 May ร้านยังไม่รู้จักสินค้าที่จะออกในอนาคต ถ้านับสินค้าเหล่านั้นด้วยก็คือ leakage |
| **Cold-start** | cold user ได้ popularity, cold item ถูกตัดออกแต่รายงานจำนวนแยกไว้ | ซื่อสัตย์กว่าการแต่ง factor หรือ profile ขึ้นมาเอง |
| **Denominators** | Precision หารด้วย K เสมอ, Recall หารด้วยจำนวน relevant, NDCG ตัด user ที่ IDCG = 0, และรายงานจำนวน user ที่ใช้ในทุกค่าเฉลี่ย | ตัวหารที่ไม่ชัดเจนทำให้เทียบงานกันไม่ได้ (Järvelin & Kekäläinen, 2002) |
| **Full ranking** | จัดอันดับทั้ง catalog ไม่สุ่มตัวอย่าง negatives | Sampled metrics ทำให้อันดับของ model ผิดเพี้ยนได้ (Krichene & Rendle, 2020) |
| **Pre-specification** | กำหนด K = 5 และ cutoffs ไว้ล่วงหน้า ไม่ปรับตามคะแนน test | ป้องกันการ tune บน test ซึ่งเป็นปัญหาที่พบบ่อยในงานวิจัย recommender (Dacrema et al., 2019) |

---

## Key Points สำหรับมือใหม่

1. **Contract มาก่อน model:** ถ้ากติกาข้อมูลและการวัดผลไม่นิ่ง คะแนนของ model ไหนก็เชื่อไม่ได้
2. **ไม่รู้ ≠ ไม่ชอบ:** matrix ของระบบแนะนำว่างเกือบทั้งหมด (97.6%) และต้องเก็บแบบ sparse
3. **แบ่งตามเวลาเสมอ:** การสุ่มแบ่งข้อมูลคือการแอบดูอนาคต
4. **ต้องมี baseline:** popularity ง่ายแต่เป็นเกณฑ์ที่ทุก model ต้องชนะ งานวิจัยหลายชิ้นพบว่า model ซับซ้อนหลายตัวแพ้ baseline ง่ายๆ ที่ tune ดีแล้ว (Dacrema et al., 2019)
5. **อ่านตัวหารทุกครั้ง:** "แม่น X%" ต้องถามต่อเสมอว่าคิดจากใครบ้าง และตัดใครออกไป

---

## หมายเหตุ

โจทย์อ้างถึง companion script และ reference `results.json` ของอาจารย์ แต่ไฟล์ทั้งสองไม่ได้อยู่ในชุดที่ได้รับ ผมจึงเขียนโค้ดเองตามสูตรในเอกสาร และใส่ `assert` ตรวจกับตัวเลขที่โจทย์ระบุไว้ทั้งหมด (121 / 38 / 41, U = 71, I = 70, density 0.024346, worked example, 17 unseen-item events, 9 evaluable users ที่มี cold users 2 คน) ทุกค่าผ่าน

## References

- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS*, 20(4). https://doi.org/10.1145/582415.582418
- Krichene, W., & Rendle, S. (2020). On Sampled Metrics for Item Recommendation. *KDD 2020*.
- Marlin, B. M., & Zemel, R. S. (2009). Collaborative prediction and ranking with non-random missing data. *RecSys 2009*.
- Steck, H. (2010). Training and testing of recommender systems on data missing not at random. *KDD 2010*.
- Meng, Z., McCreadie, R., Macdonald, C., & Ounis, I. (2020). Exploring Data Splitting Strategies for the Evaluation of Recommendation Models. *RecSys 2020*.
- Ji, Y., Sun, A., Zhang, J., & Li, C. (2023). A Critical Study on Data Leakage in Recommender System Offline Evaluation. *ACM TOIS*, 41(3).
- Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. *RecSys 2019*.
