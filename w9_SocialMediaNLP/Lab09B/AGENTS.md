# Agent Guidelines: Week 09B — Social Media NLP (Main-Idea Extraction & Topic Modeling)

Governing Procedure: [`Lab 09B - Conversational Dynamics & Resolution SLA.pdf`](Lab%2009B%20-%20Conversational%20Dynamics%20&%20Resolution%20SLA.pdf)  
Course: **961731 Business Data Analytics** (Chiang Mai University, Data Science Program)  
Primary Dataset: Kaggle Customer Support on Twitter (TWCS, ~2.8M rows -> 100,000 deterministic sample)  
Case Directory: `w9_SocialMediaNLP/Lab09B/`  
Outputs Directory: `Lab09B_outputs/` | Figures Directory: `figures/`

---

## 1. Executive Context & Educational "Why" (ทำไมเราถึงทำสิ่งนี้?)

ในงานวิเคราะห์ข้อความบริการลูกค้า (Customer Support Text Analytics) เรามักเจอปัญหาว่า **"ลูกค้ากำลังพูดถึงเรื่องอะไรกันแน่?"** (Main Ideas) โดยที่ข้อความเหล่านี้ **ไม่มีใครมาติดป้ายกำกับ Sentiment ไว้ล่วงหน้า** (Unsupervised Learning)

### เหตุผลเบื้องหลังของแต่ละกระบวนการ (The "Why")
1. **ทำไมต้องสุ่มตัวอย่างด้วย SHA-256 (100,000 rows)?**
   - ชุดข้อมูล TWCS มีขนาดใหญ่มาก (~2.8 ล้านข้อความ) การรันทั้งหมดจะกิน Memory เกินความจำเป็น
   - การสุ่มแบบสุ่มเดี่ยว (`df.sample()`) จะทำให้ผลลัพธ์ของแต่ละคนไม่เหมือนกัน ไม่สามารถทำ Reproducible Audit ได้
   - การแฮชสตริง `"961731:" + tweet_id` ด้วย SHA-256 แล้วหยิบ 100,000 แถวแรก รับประกันว่า **ทุกคนบนโลกจะได้ 100,000 แถวเดียวกันเป๊ะ 100%**
2. **ทำไมต้องกรองเฉพาะ Inbound Messages?**
   - `inbound == True` คือข้อความที่ **ลูกค้าทักเข้ามา** เพื่อแจ้งปัญหาหรือสอบถาม
   - `inbound == False` (Outbound) คือข้อความที่ **แอดมินหรือบอทตอบกลับ** ซึ่งมักเป็นเทมเพลตซ้ำๆ เช่น *"สวัสดีครับ รบกวนแจ้งเลขที่บัญชีทาง DM"*
   - หากนำ Outbound มารวม หัวข้อที่โมเดลค้นพบจะกลายเป็น "คำตอบสำเร็จรูปของแอดมิน" ไม่ใช่ "ปัญหาของลูกค้า"
3. **ทำไมต้อง Deduplicate หา Earliest Representative?**
   - ข้อความไวรัล สแปม หรือการรีทวีตคำเดิมซ้ำๆ จะทำให้จำนวนเคสบวมเกินจริง
   - การยุบข้อความที่ Clean แล้วตรงกันให้เหลือ 1 ตัวแทนแรกสุดตามเวลา UTC ช่วยให้โมเดลค้นพบ **"แก่นของปัญหา" (Topics)** โดยไม่ถูกบิดเบือนด้วยปริมาณสแปม
4. **ทำไมต้องเพิ่ม Domain Stop Words (27 คำ)?**
   - คำทั่วไปใน Twitter Support เช่น `usertoken`, `urltoken`, `hi`, `hello`, `thanks`, `please`, `dm` ปรากฏในเกือบทุกทวีต
   - ถ้าไม่ตัดทิ้ง ทุกหัวข้อจะมีแต่คำทักทายเหล่านี้ ทำให้แยกความแตกต่างของแต่ละแผนกงานไม่ออก
5. **ทำไมเลือก NMF ($K=6$) แทนโมเดลอื่น?**
   - Nonnegative Matrix Factorization บังคับให้น้ำหนักทุกตัวไม่ติดลบ ($W \ge 0, H \ge 0$) ทำให้มองข้อมูลเป็น **"การประกอบรวมของชิ้นส่วน" (Additive Parts)** ตีความง่าย
   - ฟิกซ์ $K=6$ สอดคล้องกับการจัดคิวงานจริงของ Support Operations (เช่น แผนกขนส่ง, แผนกเงินคืน, แผนกบัญชีผู้ใช้, แผนกเทคนิค ฯลฯ)
6. **ทำไมต้อง Normalization สเกลของ $H$ และ $W$?**
   - สมการ $WH$ มีปัญหา Arbitrary Scale (คูณ 10 ที่คอลัมน์ของ $W$ แล้วหาร 10 ที่แถวของ $H$ ได้ผลลัพธ์เท่าเดิม)
   - การ Normalize แถวของ $H$ ให้ยาว 1 หน่วย ($L_2$ norm) ทำให้เราสามารถเปรียบเทียบน้ำหนักระหว่าง Topic ได้อย่างเป็นธรรม และนำไปวัด Cosine Similarity ได้อย่างแม่นยำ
7. **ทำไมต้อง Extractive Summaries & Human Review?**
   - โมเดลสถิติรู้แค่คำที่เกิดร่วมกัน แต่ **ไม่เข้าใจความหมาย** มนุษย์จึงต้องตรวจเช็กจริง (20 ตัวอย่างต่อหัวข้อ)
   - การดึงข้อความจริง 3 โพสต์ที่ Cosine สูงสุดมาเป็นตัวแทน ช่วยให้ฝ่ายบริหารเห็น **หลักฐานที่เป็นคำพูดจริงของลูกค้า** โดยไม่ต้องให้ Generative AI มาแต่งเรื่องขึ้นมาเอง

---

## 2. กฎเหล็กและข้อห้าม (Strict Guardrails)

1. **Inbound/Outbound คือทิศทางข้อความ ไม่ใช่ Sentiment:**  
   ห้ามสรุปหรือตั้งกฎว่า Inbound = Negative หรือ Outbound = Positive เด็ดขาด ข้อมูลนี้ไม่มี Gold Sentiment Labels
2. **ห้ามสร้าง Dense Matrix ขนาด $N \times N$:**  
   ด้วยข้อมูล 100,000 แถว เมทริกซ์ $100,000 \times 100,000$ จะใช้แรมมหาศาล ($>70\text{ GB}$) และทำให้ระบบล่ม ให้ใช้ Sparse Matrix operations และ Groupby เท่านั้น
3. **อนุรักษ์จำนวนแถวใน Registry ($N = 100,000$):**  
   `twcs_sample_registry.parquet` ต้องมี 100,000 แถวพอดี แถวที่ไม่ผ่านเกณฑ์ (เช่น Outbound หรือ Text ว่าง) ให้ติดแฟล็ก `exclusion_reason` ห้ามลบทิ้งเงียบๆ
4. **แสดงตัวหาร $n_b$ เสมอในกราฟแนวโน้มตามเวลา:**  
   ยอดเคสที่เพิ่มขึ้นอาจมีสัดส่วนลดลงได้ หากยอดติดต่อรวมโตเร็วกว่า ต้องแสดงทั้ง Count และ Share คู่กันเสมอ

---

## 3. สัญญาข้อมูลและสคีมา (Data Contract & Ingestion)

### ตารางฟิลด์ใน `twcs_sample_registry.parquet` (100,000 แถว)

| Field Name | Type | Description & Rule |
| :--- | :--- | :--- |
| `tweet_id` | `string` | รหัสทวีตต้นฉบับ เก็บเป็น string ห้ามแปลงเป็น float |
| `author_id` | `string` | รหัสผู้เขียน (ลูกค้าหรือแบรนด์) หากว่างให้คงเป็น unknown ห้ามรวมเป็นคนเดียวกัน |
| `inbound` | `bool` | `True` = ลูกค้าทักมา, `False` = แบรนด์ตอบกลับ (ตรวจจับ Casing ที่ผิดปกติ) |
| `created_at` | `string` | เวลาดั้งเดิม |
| `created_at_utc` | `datetime` | แปลงเวลาเป็น UTC สำหรับการจัด Time Bins |
| `text` | `string` | ข้อความดิบดั้งเดิม |
| `text_clean` | `string` | ข้อความที่ผ่านการ Clean (NFKC, `usertoken`, `urltoken`, คงคำปฏิเสธ) |
| `is_representative`| `bool` | `True` สำหรับข้อความตัวแทนแรกสุดในแต่ละกลุ่มข้อความที่ซ้ำกัน |
| `analysis_eligible`| `bool` | `True` iff `inbound == True` และ `text_clean != ''` และ `is_representative == True` |
| `exclusion_reason` | `string` | ระบุสาเหตุที่ตัดออก: `'OUTBOUND'`, `'DUPLICATE_TEXT'`, `'EMPTY_TEXT'`, `'INVALID_TIME'` |
| `feature_nonzero` | `bool` | `True` ถ้าเวกเตอร์ TF-IDF มีค่ามากกว่า 0 |
| `topic_id` | `int` / `string` | รหัสหัวข้อหลัก (0–5) หรือ `'unassigned'` หากค่าน้ำหนักเป็นศูนย์ |

---

## 4. พีชคณิตและคณิตศาสตร์ NMF (Mathematical Formulations)

### 4.1 Sublinear TF และ Smoothed IDF
$$TF^*(t, d) = \begin{cases} 1 + \ln(c) & \text{ถ้า } c > 0 \\ 0 & \text{ถ้า } c = 0 \end{cases}$$
$$IDF(t) = \ln\left(\frac{N + 1}{df(t) + 1}\right) + 1$$

### 4.2 วัตถุประสงค์ Frobenius NMF ($K=6$)
$$\min_{W \ge 0, H \ge 0} \frac{1}{2} \|X - WH\|_F^2 = \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^v (X_{ij} - (WH)_{ij})^2$$
พารามิเตอร์: `n_components=6`, `init='nndsvda'`, `max_iter=300`, `tol=1e-4`, `random_state=961731`

### 4.3 Factor Scale Normalization & Invariance Proof
ปรับขนาดแถวของ $H$ ให้เป็น Unit $L_2$ norm:
$$s_k = \|H_k\|_2 = \sqrt{\sum_{j=1}^v H_{kj}^2}$$
$$H'_k = \frac{H_k}{s_k}, \quad W'_k = W_k \cdot s_k$$
**การพิสูจน์:** $W'_k H'_k = (W_k \cdot s_k) \left(\frac{H_k}{s_k}\right) = W_k H_k \implies W' H' = WH$  
*(ในโค้ดยืนยันด้วย `assert np.allclose(W @ H, W_prime @ H_prime)`)*

### 4.4 Cosine Similarity สำหรับ Extractive Summaries
วัดความคล้ายคลึงระหว่างเวกเตอร์เอกสาร $x$ กับเวกเตอร์หัวข้อที่ Normalize แล้ว $H'_k$:
$$\text{Cosine}(x, H'_k) = \frac{x \cdot H'_k}{\|x\|_2 \|H'_k\|_2} = \frac{x \cdot H'_k}{\|x\|_2}$$
*(เนื่องจาก $\|H'_k\|_2 = 1$ เรียบร้อยแล้ว)*

---

## 5. รายการไฟล์ส่งมอบ (Deliverables Manifest)

ทุกไฟล์จะถูกสร้างใน `w9_SocialMediaNLP/Lab09B/`:
- **Code & Test Suite:**
  - `analyze.py`: สคริปต์ไปป์ไลน์หลัก รันได้ผ่าน CLI
  - `test_analyze.py`: สคริปต์ตรวจ Invariant และ Unit Tests
  - `Lab09B_Main_Idea_Extraction.ipynb`: โน้ตบุ๊กบรรยายพร้อมภาพและตารางผลลัพธ์
- **Data & Artifacts (`Lab09B_outputs/`):**
  - `twcs_sample_ids.csv`: 100,000 ID ที่สุ่มด้วย SHA-256 พร้อมค่า digest
  - `twcs_sample_registry.parquet`: 100,000 แถวข้อมูลพร้อมแฟล็กตรวจสอบ
  - `keyphrases.csv`: Top unigrams/bigrams เรียงตาม Document Frequency และ TF-IDF
  - `nmf_topic_terms.csv`: Top 10 คำสำคัญของแต่ละ Topic ทั้ง 6 หัวข้อ
  - `topic_assignments.parquet`: ผลการจัดหัวข้อให้แต่ละตัวแทนเอกสาร
  - `topic_review.csv`: ผลการตรวจสอบโดยมนุษย์ 20 ตัวอย่างต่อหัวข้อ (รวม 120 แถว)
  - `topic_summaries.csv`: การ์ดสรุปหัวข้อพร้อม 3 โพสต์จริงที่เป็นตัวแทน + 1 โพสต์ที่เป็น Outlier
  - `validation_report.json`: รายงานสถิติและ Invariant assertions
- **Reports & Visualizations:**
  - `Lab09B_REPORT.md`: Findings Board สรุป 3 ประเด็นสำคัญเชิงปฏิบัติการ
  - `DATA_DEFENSE.md`: รายงานแก้ต่าง 2 หน้า (ระเบียบวิธีวิจัย ขีดจำกัดของข้อมูล และจริยธรรม)
  - `figures/`:
    - `cooccurrence_heatmap.png`: เมทริกซ์ 20 คำที่เกิดร่วมกันบ่อยสุด
    - `topic_term_heatmap.png`: การกระจายตัวของคำใน 6 Topics
    - `temporal_topic_shares.png`: แนวโน้มสัดส่วนและจำนวนของแต่ละ Topic ตามเวลา ($n_b$ visible)

---

## 6. เกณฑ์การประเมิน (100 Points Rubric)

| หมวดหมู่ | คะแนน | เกณฑ์การตรวจประเมิน |
| :--- | :---: | :--- |
| **Diagnosis** | 25 | สุ่มตัวอย่าง SHA-256 ถูกต้อง (10); การ Clean และกระแสตัวหาร (10); คำนวณ TF-IDF/Cosine ถูกต้อง (5) |
| **Technical Execution** | 35 | สกัด Keyphrase และรัน 6-Topic NMF (15); Label ผ่านการ Audit และมี Extractive Summary (10); พล็อต Co-occurrence และกราฟเวลาถูกต้อง (10) |
| **Methodological Defense**| 25 | ชี้แจงขอบเขตการสำรวจและข้อจำกัดของกลุ่มตัวอย่าง (10); แยกแยะ Topic ออกจาก Sentiment อย่างเด็ดขาด (10); อ้างอิงวรรณกรรมทางวิชาการ (5) |
| **Automation** | 15 | รันซ้ำได้แบบ Fresh-run (5); ผ่าน Invariant assertions ทั้งหมด (5); แพ็กเกจส่งมอบครบถ้วน (5) |

---

## 7. Communication Language

สื่อสารกับผู้ใช้เป็นภาษาไทย (ภาษาไทย) โดยคงคำศัพท์เทคนิค คณิตศาสตร์ และชื่อโค้ดเป็นภาษาอังกฤษอย่างแม่นยำ (เช่น TF-IDF, NMF, Frobenius norm, Cosine similarity, Inbound, Outbound)
