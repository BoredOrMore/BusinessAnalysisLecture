# Lab 11B Report: Content and Collaborative Recommendation

> ตัวเลขทุกตัวในรายงานนี้มาจากการรัน `python3 lab11B_content_cf.py` (ผลเต็มอยู่ใน `evidence/results.json`)

## TLDR

- ตัวอย่างคำนวณด้วยมือในโจทย์ทุกตัวตรงกับโค้ด ได้แก่ TF-IDF ของ A = [0.549572, 0, 0.835446, 0], content score B = 0.847392 และ D = 0.556954, User-CF prediction = 4.580127, Item-CF prediction = 4.453079
- บนข้อมูลจริง ทั้ง 4 วิธี (popularity, content, user-CF, item-CF) **ได้ความแม่นเท่ากันทุกตัว** คือ validation NDCG@5 = 0.0931 และ test = 0
- สาเหตุไม่ใช่ว่าทุกวิธีดีเท่ากัน แต่เป็นเพราะ **ข้อมูลมีไม่พอ** user สองคนใดๆ ให้คะแนนสินค้าร่วมกันได้มากสุดแค่ 2 ชิ้น ซึ่งต่ำกว่าเกณฑ์ 3 ชิ้น CF จึงต้อง fallback ทุกช่อง
- Content-based เป็นวิธีเดียวที่ทำให้รายการแนะนำ **หลากหลายขึ้น** (coverage 0.21 ถึง 0.27 เทียบกับ 0.07 ถึง 0.09 ของวิธีอื่น) เพราะไม่ต้องพึ่งพฤติกรรมของคนอื่น

**ความหมายทางธุรกิจ:** ถ้าร้านมีข้อมูลบางขนาดนี้ อย่าเพิ่งลงทุนกับ CF ให้เริ่มจาก content-based ซึ่งใช้คำอธิบายสินค้า ควบคู่กับ popularity แล้วเก็บข้อมูลเพิ่มก่อน

---

## Problem 1: Content-based Recommender

### 1.1 Represent

Catalog ที่ใช้มี 70 items ซึ่งตรงกับ 70 items ใน training ทุกตัว ทุกแถวติดป้าย `SYNTHETIC_TEACHING_ONLY` แปลว่าเป็นข้อมูลสอน ไม่ใช่คำอธิบายสินค้าจริง

**Tiny catalog (N = 4)**

| Term | df | IDF = log10(4/df) |
|---|---|---|
| coffee | 3 | 0.124939 |
| herbal | 2 | 0.301030 |
| roast | 2 | 0.301030 |
| tea | 2 | 0.301030 |

**ตัวอย่างช่องที่คำปรากฏ 2 ครั้ง (item A, คำว่า coffee):** TF = log10(1+2) = 0.477121 แล้วคูณ IDF ได้ 0.477121 x 0.124939 = **0.059611** ส่วนคำว่า roast ใน A ได้ 0.090619 รวมเป็น norm = 0.108468 เมื่อ normalize แล้วได้ A = [0.549572, 0, 0.835446, 0]

**ต่างจากค่า default ของ library อย่างไร:** `TfidfVectorizer` ของ scikit-learn ใช้ raw count (TF = 2) และ smoothed IDF = ln(5/4) + 1 = 1.223144 ช่องเดียวกันจึงได้ 2.446287 ก่อน normalize ซึ่งเป็นคนละตัวเลขกับสูตรของห้องเรียน ดังนั้นต้องประกาศทุกครั้งว่าใช้ convention ไหน

**Catalog จริง:** มี 70 items x 10 terms คำที่พบบ่อยสุดคือ large (df 36, IDF 0.289) ส่วนคำที่พบน้อยสุดคือ juice (df 12, IDF 0.766) ไม่มีสินค้าไหนที่ได้ zero vector และค่า norm อยู่ระหว่าง 0.2325 ถึง 0.3151

### 1.2 Profile

ลูกค้าให้ A = 5 และ C = 4 จึงได้น้ำหนัก max(rating - 3, 0) เป็น 2 และ 1 ตามลำดับ Profile = 2 x A + 1 x C แล้ว normalize ให้ยาว 1 หน่วย

**Fallback ที่ประกาศไว้:** ถ้าลูกค้าไม่มี rating ≥ 4 เลย หรือ profile มีความยาวเป็น 0 ให้ใช้ popularity list ของ catalog ที่ eligible ตามกฎ cold-user ของ Lab 11A

> ประโยคในเอกสารโจทย์ที่ขึ้นต้นว่า "A zero profile falls back to" ถูกตัดหายไปกลางประโยคในไฟล์ .docx ผมจึงเลือกใช้ popularity เพราะเป็น fallback เดียวที่ contract ของ Lab 11A กำหนดไว้แล้ว ถ้าอาจารย์กำหนดเป็นอย่างอื่น ต้องแก้ส่วนนี้

ผมไม่แปลง rating ต่ำเป็นน้ำหนักติดลบ เพราะการที่ลูกค้าไม่ชอบสินค้าชิ้นหนึ่ง ไม่ได้แปลว่าลูกค้าไม่ชอบ "คำ" ทุกคำในคำอธิบายของสินค้านั้น

### 1.3 Recommend

หลังตัดสินค้าที่เคยซื้อแล้ว (A และ C) ออก ได้ B = **0.847392** และ D = **0.556954** B ขึ้นนำเพราะมีคำ roast และ coffee ร่วมกับ profile โดยคำว่า roast ให้คะแนน 0.7229 และ coffee ให้ 0.1245

**ลูกค้าจริง user 2:** ให้ 508 = 5 และ 516 = 4 ได้ Top-5 = 538, 598, 553, 503, 533 สินค้าอันดับ 1 คือ 538 "juice roasted large" เพราะมีคำร่วมกับ profile คือ juice (0.592), roasted (0.218) และ large (0.084)

---

## Problem 2: Neighborhood Models

### 2.1 User-based (Pearson + Shrinkage)

| เพื่อนบ้าน | Overlap | Pearson | น้ำหนักหลัง shrink n/(n+2) | ใช้ไหม |
|---|---|---|---|---|
| U1 | 3 | 0.866025 | 0.519615 | ✓ |
| U2 | 3 | -0.866025 | -0.519615 | |
| U3 | 3 | 1.000000 | 0.600000 | ✓ |
| U4 | 3 | -0.866025 | -0.519615 | |

**Pair centering กับ prediction centering ต่างกันอย่างไร:** ตอนคำนวณ Pearson ให้ลบค่าเฉลี่ยเฉพาะสินค้าที่ทั้งคู่ให้คะแนนร่วมกัน เพื่อเทียบกันอย่างยุติธรรม ส่วนตอนทำนายให้ใช้ค่าเฉลี่ยของ **ทั้งแถว** ของเพื่อนบ้าน เพื่อบอกว่า "ปกติคนนี้ให้คะแนนประมาณเท่าไร"

### 2.2 Item-based (Adjusted Cosine)

| คู่ | Support | Adjusted cosine | เลือก |
|---|---|---|---|
| D-A | 4 | 0.188639 | ✓ |
| D-B | 4 | -0.543201 | |
| D-C | 4 | 0.227710 | ✓ |

**ต่างจาก text cosine อย่างไร:** สูตรหน้าตาเหมือนกัน แต่ใช้หลักฐานคนละแบบ text cosine เทียบ "คำ" ในคำอธิบายสินค้า ส่วน adjusted cosine เทียบ "พฤติกรรม" ว่าคนกลุ่มเดียวกันให้คะแนนสองสินค้าไปทางเดียวกันไหม หลังลบนิสัยการให้คะแนน (ใจดีหรือใจร้าย) ของแต่ละคนออกแล้ว

### 2.3 Verify

| วิธี | การคำนวณด้วยมือ | ผลจากโค้ด |
|---|---|---|
| User-CF | 4 + [0.6(4-4) + 0.519615(5-3.75)] / (0.6 + 0.519615) | **4.580127** |
| Item-CF | 4 + [0.188639(5-4) + 0.227710(4-4)] / (0.188639 + 0.227710) | **4.453079** |

สำหรับ Item-CF ตัวคูณ shrinkage 4/6 ตัดกันหายไปเพราะ support ของทั้งสองคู่เท่ากับ 4

**Boundary case:** ถ้า U0 ให้คะแนน A, B, C เท่ากันหมดที่ 3 ความแปรปรวนจะเป็น 0 และ Pearson หาค่าไม่ได้ ระบบจะ fallback ไปใช้ค่าเฉลี่ยของ user ได้ **3.0000** ซึ่งเป็นค่าจำกัด (finite) ไม่ใช่ NaN

---

## Problem 3: เปรียบเทียบพฤติกรรม

### 3.1 Same warm user (user 2)

| วิธี | Top-5 | ที่มาของคะแนน |
|---|---|---|
| Popularity | 524, 546, 556, 557, 588 | ความนิยมใน training ตัดสินค้าที่เคยซื้อออก |
| Content | 538, 598, 553, 503, 533 | content profile |
| User-CF | 524, 546, 556, 557, 588 | fallback ค่าเฉลี่ยของ user ทั้ง 5 ช่อง |
| Item-CF | 524, 546, 556, 557, 588 | fallback ค่าเฉลี่ยของ user ทั้ง 5 ช่อง |

**ทำไมอันดับต่างกัน:** content ใช้คำอธิบายสินค้า ซึ่งมีครบทุกชิ้น จึงคำนวณได้จริง ส่วน CF ต้องมีลูกค้าที่ให้คะแนนสินค้าร่วมกันอย่างน้อย 3 ชิ้น แต่ใน training คู่ที่ overlap มากที่สุดมีแค่ **2 ชิ้น** CF จึง fallback ทุกช่อง เมื่อคะแนนเสมอกันหมด จึงเรียงตาม popularity แทน

### 3.2 Fair evaluation

| Split | Method | Users (cold) | P@5 | R@5 | NDCG@5 | Coverage | Fallback share |
|---|---|---|---|---|---|---|---|
| valid | popularity | 10 (6) | 0.04 | 0.20 | 0.0931 | 0.0857 | 1.00 |
| valid | content | 10 (6) | 0.04 | 0.20 | 0.0931 | 0.2143 | 0.80 |
| valid | user_cf | 10 (6) | 0.04 | 0.20 | 0.0931 | 0.0857 | 1.00 |
| valid | item_cf | 10 (6) | 0.04 | 0.20 | 0.0931 | 0.0857 | 1.00 |
| test | popularity | 9 (2) | 0 | 0 | 0 | 0.0714 | 1.00 |
| test | content | 9 (2) | 0 | 0 | 0 | 0.2714 | 0.56 |
| test | user_cf | 9 (2) | 0 | 0 | 0 | 0.0714 | 1.00 |
| test | item_cf | 9 (2) | 0 | 0 | 0 | 0.0714 | 1.00 |

- **Coverage** = จำนวนสินค้าไม่ซ้ำที่ถูกแนะนำให้ users ที่ถูกวัดผล หารด้วย 70
- **Fallback share** = สำหรับ content คือสัดส่วน users ที่ได้ popularity แทน ส่วน CF คือสัดส่วนช่องในรายการที่มาจาก fallback

ทุกวิธีทาย hit ได้เท่ากันบน validation (2 users) เพราะ users ทั้ง 2 คนนั้นได้ popularity list เหมือนกันหมด

### 3.3 Stress tests (ทุกกรณีทำบนสำเนา)

| กรณี | ผล |
|---|---|
| Missing metadata (B ไม่มีคำอธิบาย) | B ได้ zero vector และ score = 0 ซึ่งเป็นค่าจำกัด แล้วเรียงตาม tie-break |
| Zero text vector (A = "coffee" และ coffee อยู่ในทุก doc) | IDF = 0 ทำให้ norm = 0 และ profile = None จึงใช้ popularity fallback |
| Constant ratings (U0 ให้ 3 ทุกชิ้น) | Pearson หาค่าไม่ได้ จึงใช้ user mean = 3.0000 |
| No overlap (U9 ให้คะแนนแค่สินค้า E) | user-CF และ item-CF ให้ 4.0000 จาก user mean |

**ข้อมูลไม่พอ ไม่ใช่การคำนวณพัง:** ไม่มีกรณีไหนได้ NaN หรือเกิด division by zero ทุกกรณีคือ "ไม่มีหลักฐานพอให้คำนวณ" แล้วจึงใช้ fallback ที่ประกาศไว้ ซึ่งต่างจาก bug ทางคณิตศาสตร์

---

## Methodological Defense

| ประเด็น | คำอธิบาย |
|---|---|
| **ทำไมอันดับต่างกัน** | แต่ละวิธีใช้หลักฐานคนละแบบ content ใช้ text ส่วน CF ใช้พฤติกรรมร่วม ถ้าหลักฐานแบบไหนไม่มี วิธีนั้นจะ fallback ผลลัพธ์จึงสะท้อน **data availability** ไม่ใช่ความเก่งของ algorithm |
| **ทำไมไม่เติม 0** | การเติม 0 สร้างหลักฐานปลอมว่าลูกค้าเกลียดสินค้าที่ไม่เคยเห็น ทำให้ similarity เพี้ยน fallback ที่ซื่อสัตย์พร้อมคะแนนพอประมาณดีกว่า |
| **Shrinkage และ min overlap** | correlation ที่คิดจาก 2 ถึง 3 จุดมีความแปรปรวนสูงมาก การลดน้ำหนักตาม n/(n+λ) เป็นแนวทางมาตรฐาน (Herlocker et al., 1999; Koren, 2008) |
| **Item-based vs user-based** | Item similarity มักนิ่งกว่า user similarity เพราะสินค้าหนึ่งชิ้นมีคนให้คะแนนหลายคน (Sarwar et al., 2001; Linden et al., 2003) แต่ในข้อมูลนี้ทั้งคู่มี support ไม่พอ |
| **Content-based กับ cold-start** | Content ใช้งานได้ทันทีกับลูกค้าที่มี positive history แม้มีแค่ 1 ชิ้น และไม่ต้องพึ่งข้อมูลของคนอื่น (Lops et al., 2011) แต่เสี่ยงเรื่อง over-specialization คือแนะนำแต่ของคล้ายเดิม |
| **ข้อจำกัดของ tiny matrix** | ตัวอย่าง 5 x 4 ใช้สาธิตการคำนวณเท่านั้น ไม่ใช่หลักฐานเชิงประจักษ์ว่า CF ชนะ baseline |
| **Computational boundary** | การคำนวณ similarity แบบ all-pairs ใช้ O(n²) ซึ่งรับได้เฉพาะข้อมูลขนาดนี้ ถ้าเป็นข้อมูลจริงต้องสร้าง candidate pairs จาก sparse postings, จำกัดจำนวนเพื่อนบ้าน หรือใช้ approximate nearest neighbor |

## Key Points สำหรับมือใหม่

1. **Content = คำอธิบายสินค้า, CF = พฤติกรรมของคน** สองวิธีนี้ใช้หลักฐานคนละแบบ
2. **TF-IDF:** คำที่พบในทุกสินค้า (เช่น "large") มีค่าน้อย คำที่หายากมีค่ามาก
3. **Pearson ไม่สนว่าใครใจดี** แค่ดูว่าให้คะแนนขึ้นลงไปทางเดียวกันไหม
4. **Shrinkage:** เห็นด้วยกันแค่ 3 ครั้งยังไม่น่าเชื่อเท่าเห็นด้วยกัน 30 ครั้ง
5. **CF ต้องการข้อมูลหนา** ถ้าข้อมูลบางขนาดนี้ (density 2.4%) CF จะ fallback ทั้งหมด
6. **ผลเท่ากันไม่ได้แปลว่าวิธีเท่ากัน** ต้องดู fallback share และ coverage ประกอบเสมอ

## References

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*, Ch. 6 (TF-IDF weighting, vector space model). Cambridge University Press.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-Based Collaborative Filtering Recommendation Algorithms. *WWW 2001*.
- Herlocker, J. L., Konstan, J. A., Borchers, A., & Riedl, J. (1999). An Algorithmic Framework for Performing Collaborative Filtering. *SIGIR 1999*.
- Koren, Y. (2008). Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model. *KDD 2008*.
- Linden, G., Smith, B., & York, J. (2003). Amazon.com Recommendations: Item-to-Item Collaborative Filtering. *IEEE Internet Computing*, 7(1).
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based Recommender Systems: State of the Art and Trends. In *Recommender Systems Handbook*. Springer.
