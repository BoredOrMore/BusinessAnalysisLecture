# บันทึกเทคนิคและคณิตศาสตร์: Lab 09A (Operational Sentiment & Triage)

ไฟล์นี้รวบรวมข้อสรุป การพิสูจน์คณิตศาสตร์ และหลักฐานเชิงประจักษ์จากการตรวจทานเอกสาร [`Lab 09A - Operational Sentiment & Triage.pdf`](Lab%2009A%20-%20Operational%20Sentiment%20&%20Triage.pdf) และชุดข้อมูล `Tweets.csv` (14,640 แถว)

---

## 1. Data Integrity: การกักกัน 18 กลุ่ม Tweet ID ที่ขัดแย้ง (36 แถว)

### ข้อมูลเชิงประจักษ์จาก `Tweets.csv`
- พบ `tweet_id` ซ้ำกันทั้งหมด 310 แถว (155 คู่)
- ในจำนวนนี้มี **18 คู่ (รวม 36 แถว)** ที่ Annotator ให้ผล Sentiment ขัดแย้งกันโดยสิ้นเชิง (เกิดขึ้นที่ค่าย American Airlines ช่วง Index 11,886 ถึง 12,180)
- ตัวอย่างข้อมูลจริง:
  - `tweet_id: 570268326250745856` $\to$ แถว 12019 ได้ `neutral` (conf: 0.6304) แต่แถว 12180 ได้ `negative` (conf: 0.6670) พร้อม `negativereason='Cancelled Flight'`
  - `tweet_id: 570268875872473088` $\to$ แถว 12016 ได้ `positive` (conf: 0.6907) แต่แถว 12177 ได้ `negative` (conf: 0.6432)
  - `tweet_id: 570270435478122497` $\to$ แถว 12011 ได้ `neutral` (conf: 0.6266) แต่แถว 12172 ได้ `positive` (conf: 1.0000)

### ทำไมถึงห้าม Majority Vote หรือลบทิ้ง?
1. **โหวตไม่ได้จริง (Tie 1 ต่อ 1):** แต่ละกลุ่มมี 2 แถวที่คนตรวจ 2 คนให้ผลต่างกัน ไม่มีเสียงข้างมากให้ตัดสิน การสุ่มเลือกคือการเดา Ground Truth เอง
2. **รักษาความสมบูรณ์ของ Denominator (Conserved Rows $N = 14,640$):** การลบทิ้งจะทำให้ตัวหารข้อมูลเพี้ยนกลายเป็น 14,604 ส่งผลให้การ Audit ข้อมูลใน Production เสียหาย ข้อมูลที่เป็น Defect ต้องถูกมองเห็นเสมอ
3. **ป้องกัน Label Noise:** ห้ามนำข้อมูลที่มีความขัดแย้งเข้าไปเทรนหรือทดสอบโมเดล

### สถานะใน `airline_registry.parquet`
ทุกแถวใน 18 กลุ่มนี้ (36 แถว) ต้องบันทึกสถานะดังนี้:
- `audit_flags`: `'CONFLICTING_LABEL'`
- `model_eligible`: `False`
- `split`: `'quarantined'`
- `is_representative`: `False`

---

## 2. Annotation Leakage Trap: กับดักคอลัมน์ `negativereason`

### ตาราง Cross-tabulation จากข้อมูลจริง 14,640 แถว

| Sentiment | `is_negativereason_null == False` (มีค่า) | `is_negativereason_null == True` (ว่าง/NaN) | รวม | สัดส่วนที่ Non-null (%) |
| :--- | :---: | :---: | :---: | :---: |
| **negative** | **9,178** | **0** | 9,178 | **100.0%** |
| **neutral** | **0** | **3,099** | 3,099 | **0.0%** |
| **positive** | **0** | **2,363** | 2,363 | **0.0%** |
| **รวม** | 9,178 | 5,462 | 14,640 | 62.7% |

### ผลกระทบและเหตุผลที่ห้ามใช้
1. **เป็น Post-Annotation Artifact:** CrowdFlower จะแสดงช่องให้กรอกสาเหตุความไม่พอใจ *เฉพาะเคสที่ Annotator ตัดสินว่า Negative แล้วเท่านั้น* ทำให้คลาสอื่นเป็น NaN 100%
2. **สร้างภาพลวงตา (False Perfection):** หากใส่ฟีเจอร์ Boolean `is_negativereason_null` โมเดลจะได้ Precision/Recall 100% ในคลาส Negative ทันทีโดยไม่ได้อ่านเนื้อหา Text เลย
3. **Inference Time Failure:** ในระบบจริง (Live Triage) โพสต์ใหม่ของลูกค้าจะไม่มีมนุษย์มากรอก `negativereason` ล่วงหน้า โมเดลที่พึ่งพาฟีเจอร์นี้จะใช้งานไม่ได้เลย
4. **กฎของ Spec:** ใช้ได้เฉพาะ Text-derived features เท่านั้น และต้องนำเสนอตารางนี้ในรายงานเพื่อแฉ Leakage Trap

---

## 3. ขั้นตอนการคำนวณคณิตศาสตร์ TF-IDF & Normalization

กำหนดโจทย์ทดสอบ: เอกสารใน Training set $N_{train} = 3$ ชุด คำว่า $t$ ปรากฏใน:
- $d_1$: 2 ครั้ง ($c = 2$)
- $d_2$: 1 ครั้ง ($c = 1$)
- $d_3$: 0 ครั้ง ($c = 0$)

### Step 1: Document Frequency ($df(t)$)
นับจำนวนเอกสารที่มีคำว่า $t$ อย่างน้อย 1 ครั้ง:
$$df(t) = 1 (d_1) + 1 (d_2) + 0 (d_3) = 2$$
*(Total Token Count = 3 แต่ Document Frequency = 2)*

### Step 2: Sublinear Term Frequency ($TF^*$)
ใช้สูตร Sublinear เพื่อลดทอนคำที่เกิดซ้ำซาก:
$$TF^*(t, d) = \begin{cases} 1 + \ln(c) & \text{ถ้า } c > 0 \\ 0 & \text{ถ้า } c = 0 \end{cases}$$
ใน $d_1$ ($c = 2$):
$$TF^*(t, d_1) = 1 + \ln(2) \approx 1 + 0.693147 = \mathbf{1.6931}$$

### Step 3: Smoothed Inverse Document Frequency ($IDF$)
ใช้สูตร Scikit-learn (`smooth_idf=True`):
$$IDF(t) = \ln\left(\frac{N_{train} + 1}{df(t) + 1}\right) + 1$$
แทนค่า $N_{train} = 3, df(t) = 2$:
$$\frac{N+1}{df+1} = \frac{3 + 1}{2 + 1} = \frac{4}{3} \approx 1.3333$$
$$IDF(t) = \ln\left(\frac{4}{3}\right) + 1 \approx 0.287682 + 1 = \mathbf{1.2877}$$

### Step 4: Unnormalized Weight ($w(t, d_1)$)
$$w(t, d_1) = TF^*(t, d_1) \times IDF(t) = 1.693147 \times 1.287682 = \mathbf{2.1802}$$

### Step 5: $L_2$ Normalization
สำหรับเวกเตอร์เอกสาร $v = (3, 4)$:
- ขนาด Euclidean: $\|v\|_2 = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$
- ปรับขนาดเป็น Unit vector:
  $$v_{norm} = \left(\frac{3}{5}, \frac{4}{5}\right) = \mathbf{(0.6, 0.8)}$$
- ตรวจสอบ: $\sqrt{0.6^2 + 0.8^2} = \sqrt{1.0} = 1.0$ (ผ่าน)

---

## 4. Local Contrast Decomposition (การแกะรอยการตัดสินใจของโมเดล)

### สมการกระจายคะแนน (Mathematical Formulation)
สำหรับโพสต์ที่มีเวกเตอร์ฟีเจอร์ $x$ โมเดลทำนายคลาส $k$ (ชนะเลิศ เช่น `negative`) และมีคลาส $j$ เป็นรองชนะเลิศ (เช่น `neutral`):
$$\Delta z_{kj} = z_k - z_j = (b_k - b_j) + \sum_{i} (w_{ki} - w_{ji}) x_i$$
โดยที่:
- $z_k, z_j$: คะแนนดิบ (Logit Scores) ของแต่ละคลาส
- $b_k - b_j$: ผลต่างของ Intercept (คะแนนตั้งต้นตามธรรมชาติของคลาส)
- $(w_{ki} - w_{ji}) x_i$: Contribution หรือคะแนนส่วนต่างที่คำที่ $i$ ในโพสต์ส่งผลักดันให้คลาส $k$ ชนะคลาส $j$

### วัตถุประสงค์ (Why)
1. **Explainability / Auditing:** อธิบายได้แบบโปร่งใส 100% ว่าทำไมโมเดลถึงเลือก Negative มากกว่า Neutral
2. **Catching Spurious Logic:** ตรวจสอบว่าโมเดลไม่ได้ตัดสินใจจากคำขยะ เช่น หากคำว่า `usertoken` มีน้ำหนักผลักดันสูง แสดงว่าโมเดลเริ่ม Overfit
3. **Exact Mathematical Verification:** ไม่ต้องเดาแบบ Black-box เพราะสมการนี้เป็นพีชคณิตเชิงเส้นแท้จริง

### การ Verify ใน Python Code
```python
# คำนวณผลต่าง Logit โดยตรง
delta_z_direct = logits[predicted_class] - logits[runner_up_class]

# คำนวณผลต่างจากการแยกส่วน Intercept และ Sum of Word Contributions
delta_intercept = intercept[predicted_class] - intercept[runner_up_class]
word_contributions = (coef[predicted_class, :] - coef[runner_up_class, :]) * x.toarray().flatten()
delta_z_decomposed = delta_intercept + np.sum(word_contributions)

# Numerical Assertion
assert np.isclose(delta_z_direct, delta_z_decomposed, atol=1e-5), "Decomposition mismatch!"
```

---

## 5. Aspect Orthogonality & Denominators

- **Aspect $\ne$ Sentiment:**  
  "Baggage" (กระเป๋า) บอกว่าเรื่องเกี่ยวกับอะไร (Topic)  
  "Terrible" (แย่มาก) บอกความรู้สึก (Sentiment)  
  ทั้งสองอย่างเป็นอิสระต่อกัน (Orthogonal)
- **Multi-label Overlap:**  
  โพสต์ 1 โพสต์สามารถติดแท็ก Aspect ได้หลายตัวพร้อมกัน (เช่น ทั้ง `baggage` และ `delay`)  
  **ดังนั้น ห้ามนำผลรวมของแต่ละ Aspect มาบวกกันตรงๆ แล้วอ้างว่าเป็นยอดเคสรวม**
- **การคิด Proportion และ Heatmap:**  
  ต้องระบุตัวหาร $n_a$ (จำนวนโพสต์ที่ติดแท็ก Aspect นั้นๆ) เสมอ  
  เช่น ในกลุ่มที่ติดแท็ก `baggage` ทั้งหมด 40 โพสต์ มี Negative 24, Neutral 10, Positive 6  
  $\to$ Negative = $24/40 = 60\%$, Neutral = $25\%$, Positive = $15\%$
