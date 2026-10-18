# เปรียบเทียบวิธีใน Lab 11 กับวิธีที่ใหม่กว่า

> ตัวเลขของ EASE มาจากการรัน `python3 ease_check.py` (`evidence/ease_results.json`) ส่วนตัวเลขของวิธีอื่นมาจาก Lab 11A ถึง 11C

## TLDR

- **บนข้อมูลชุดนี้ ไม่มีวิธีไหนชนะ popularity ได้** แม้แต่ EASE ซึ่งเป็น baseline สมัยใหม่ที่แข็งแรงมาก ผลบน test ก็ได้ NDCG@5 = 0 เท่ากับทุกวิธี และบน validation ยังแพ้ popularity (0.0431 เทียบกับ 0.0931)
- **ปัญหาหลักคือปริมาณข้อมูล ไม่ใช่ความเก่าของ algorithm:** มีแค่ 121 training ratings (density 2.4%) คู่ลูกค้าที่ให้คะแนนสินค้าร่วมกันได้มากที่สุดมีแค่ 2 ชิ้น และ 17 จาก 41 test events เป็นสินค้าใหม่
- **งานวิจัยหลายชิ้นยืนยันแนวโน้มเดียวกัน:** วิธีเก่าที่ tune ดีแล้ว (item-kNN, MF, iALS, EASE) มักเสมอหรือชนะ deep learning หลายตัว (Ferrari Dacrema et al., 2019; Rendle et al., 2020; Rendle et al., 2022)
- **คำแนะนำสำหรับร้านขนาดนี้:** ใช้ content-based ร่วมกับ popularity, เพิ่ม exploration ด้วย bandit และเก็บ exposure log ก่อน เมื่อมีข้อมูลหลักหมื่น interactions ค่อยพิจารณา EASE หรือ iALS ส่วน two-tower และ sequential models ควรรอให้มีข้อมูลระดับล้าน

---

## 1. ตารางเปรียบเทียบวิธี

| วิธี | ปี / ที่มา | ไอเดียแบบง่าย | ข้อมูลที่ต้องการ | Cold-start | ต้นทุนคำนวณ | หลักฐานจากงานวิจัย | เหมาะกับข้อมูล Lab 11 ไหม |
|---|---|---|---|---|---|---|---|
| **Popularity** | baseline | แนะนำของขายดีให้ทุกคน | น้อยมาก | ใช้ได้กับ user ใหม่ แต่ใช้กับ item ใหม่ไม่ได้ | ต่ำมาก | เป็นเกณฑ์ขั้นต่ำที่ model ใหม่หลายตัวยังชนะไม่ได้ (Cremonesi et al., 2010) | ✓ เป็นเกณฑ์อ้างอิง |
| **Content-based (TF-IDF)** | Lab 11B | แนะนำของที่คำอธิบายคล้ายของที่ชอบ | metadata ของสินค้า | ใช้กับ item ใหม่ได้ถ้ามีคำอธิบาย | ต่ำ | มีมานาน แต่เสี่ยงแนะนำของซ้ำแนวเดิม (over-specialization) (Lops et al., 2011) | ✓ ดีที่สุดในแง่ coverage |
| **User / Item-kNN CF** | Lab 11B | คนที่ชอบเหมือนกัน หรือของที่คนชอบคล้ายกัน | มีคู่ที่ให้คะแนนร่วมกันพอ | ใช้ไม่ได้ทั้งสองแบบ | ปานกลาง | Item-kNN ที่ tune ดีแล้วยังแข่งกับ neural models ได้ (Ferrari Dacrema et al., 2019) | ✗ overlap สูงสุดแค่ 2 ชิ้น |
| **Biased MF (SGD)** | Lab 11C (Koren et al., 2009) | บีบรสนิยมเป็นตัวเลข d ตัว | หลายพันถึงหลายล้าน ratings | ใช้ไม่ได้ทั้งสองแบบ | ต่ำถึงปานกลาง | เป็นหัวใจของผู้ชนะ Netflix Prize | ✗ global mean ชนะบน test |
| **iALS** (implicit ALS) | Hu et al., 2008 | เหมือน MF แต่ใช้กับ click และ purchase ที่ไม่มีค่า "ไม่ชอบ" | implicit interactions จำนวนมาก | ใช้ไม่ได้ | ปานกลาง | เมื่อ tune ใหม่แล้วยังติดระดับ state-of-the-art บน benchmarks (Rendle et al., 2022) | △ เมื่อมีข้อมูลมากขึ้น |
| **BPR** | Rendle et al., 2009 | สอนให้จัดอันดับ "ของที่ซื้อ > ของที่ไม่ซื้อ" | implicit + negative sampling | ใช้ไม่ได้ | ปานกลาง | ใช้แพร่หลาย แต่ไวต่อวิธีสุ่ม negatives | △ |
| **EASE** | Steck, 2019 | linear item-item model ที่หาคำตอบได้ในสูตรเดียว (closed form) | implicit, จำนวน items ไม่เกินหลักหมื่น | ใช้ไม่ได้ | ต่ำ (inverse ของ matrix ขนาด items x items) | แข่งกับหรือชนะ deep autoencoders ได้บน benchmarks มาตรฐาน (Steck, 2019) | ✗ ทดลองแล้ว (ดูหัวข้อ 2) |
| **Neural CF (NCF)** | He et al., 2017 | ใช้ neural network แทน dot product | มาก | ใช้ไม่ได้ | สูง | Rendle et al. (2020) พบว่า dot product ที่ tune ดีแล้วชนะ MLP ของ NCF | ✗ |
| **Mult-VAE** | Liang et al., 2018 | autoencoder แบบ probabilistic | มาก | ใช้ไม่ได้ | สูง | ดีบน benchmarks แต่ EASE มักเสมอ | ✗ |
| **LightGCN** (graph) | He et al., 2020 | ส่งต่อข้อมูลบนกราฟ user และ item | มาก | ใช้ไม่ได้ | สูง | ดีขึ้นจาก NGCF ด้วยการตัดส่วนที่ไม่จำเป็นออก | ✗ |
| **Two-tower retrieval** | Covington et al., 2016; Yi et al., 2019 | network สองฝั่ง (user, item) ที่ใส่ feature ได้ แล้วค้นด้วย nearest neighbor | ระดับล้าน interactions พร้อม features | ใช้ได้ถ้ามี features | สูงตอน train แต่ serve เร็ว | มาตรฐานของระบบขนาดใหญ่ (YouTube, Google) | ✗ ใหญ่เกินไป |
| **Sequential (SASRec, BERT4Rec)** | Kang & McAuley, 2018; Sun et al., 2019 | ทายของชิ้นถัดไปจากลำดับการซื้อ | ลำดับยาวต่อ user | ใช้ไม่ได้ | สูง | SASRec ที่ train ถูกวิธีเสมอหรือชนะ BERT4Rec (Petrov & Macdonald, 2022; Klenitskiy & Vasilev, 2023) | ✗ ส่วนใหญ่ user มีแค่ 1 ถึง 4 events |
| **LLM-based / generative** | Geng et al., 2022 (P5); Hou et al., 2024; Rajput et al., 2023 (TIGER) | ใช้ภาษาธรรมชาติหรือ semantic ID ในการแนะนำ | ข้อความหรือ metadata ที่มีความหมาย | zero-shot ทำได้บางส่วน | สูงมาก | zero-shot ranking ทำได้ดีพอใช้ แต่ยังแพ้ model ที่ train บนข้อมูลจริงเมื่อมีข้อมูลพอ (Hou et al., 2024) | △ ใช้แทน content ได้ถ้ามีคำอธิบายจริง แต่ข้อมูลชุดนี้เป็น synthetic |
| **Contextual bandit (LinUCB)** | Li et al., 2010 | แนะนำไปพร้อมกับทดลองไปด้วย (explore/exploit) | สร้างข้อมูลระหว่างใช้งาน | ใช้ได้ดีมาก | ต่ำถึงปานกลาง | ทดสอบบน traffic จริงของ Yahoo! Today Module และได้ click-through สูงกว่า bandit ที่ไม่ใช้ context | ✓ ใช้แก้ปัญหาข้อมูลน้อย |

สัญลักษณ์: ✓ เหมาะ, △ เหมาะเมื่อข้อมูลโตขึ้น, ✗ ไม่เหมาะกับข้อมูลชุดนี้

---

## 2. ทดลองจริง: EASE ภายใต้ contract เดียวกัน

EASE (Embarrassingly Shallow Autoencoder) เป็นหนึ่งใน baseline สมัยใหม่ที่แข็งแรงที่สุดสำหรับ implicit feedback และหาคำตอบได้ในสูตรเดียว: B = I − P · diag(1/diag(P)) โดย P = (XᵀX + λI)⁻¹ ผมจึงใช้มันทดสอบว่า "วิธีที่ใหม่กว่า" ช่วยข้อมูลชุดนี้ได้หรือไม่

**การตั้งค่า:** X = 1 สำหรับทุก rating ใน training (มองเป็น implicit interaction), เลือก λ จาก {1, 10, 100, 500} ด้วย validation NDCG@5 และ user ที่ไม่รู้จักใช้ popularity

| λ | Val NDCG@5 | Val coverage |
|---|---|---|
| 1 | 0.0431 | 0.20 |
| 10 | 0.0431 | 0.20 |
| 100 | 0.0431 | 0.20 |
| **500** (เลือก, เสมอแล้วเลือก λ ใหญ่กว่า) | 0.0431 | 0.20 |

| Method (test, 9 users) | P@5 | R@5 | NDCG@5 | Coverage |
|---|---|---|---|---|
| popularity | 0 | 0 | 0 | 0.0714 |
| EASE | 0 | 0 | 0 | 0.2714 |

**ตีความ:** EASE เรียนจากการที่สินค้าสองชิ้นถูกซื้อโดยคนเดียวกัน (co-occurrence) แต่ข้อมูลชุดนี้มีคู่สินค้าที่มีคนซื้อร่วมกันแค่ 67 คู่ และร่วมกันได้มากสุดแค่ 2 คน ผลจึงไม่ดีขึ้น ต่างกันแค่กระจายการแนะนำได้กว้างกว่า สรุปคือ **วิธีที่ใหม่และพิสูจน์แล้วว่าดีบน benchmark ขนาดใหญ่ ไม่ได้ดีกว่าเสมอบนข้อมูลเล็ก**

---

## 3. สิ่งที่งานวิจัยบอกเรื่อง "ใหม่กว่า = ดีกว่า"

| ข้อค้นพบ | แหล่งอ้างอิง |
|---|---|
| neural recommenders จาก top conferences ทำซ้ำได้แค่ 7 จาก 18 ตัว และ 6 ใน 7 ตัวนั้นแพ้ kNN หรือ heuristic ที่ tune ดีแล้ว | Ferrari Dacrema, Cremonesi & Jannach, RecSys 2019 |
| dot product (MF) ที่ tune ดีแล้วชนะ MLP ของ NCF และ MLP เรียน dot product ได้ยาก | Rendle, Krichene, Zhang & Anderson, RecSys 2020 |
| iALS ที่ tune ใหม่แล้วแข่งกับวิธีล่าสุดได้บน benchmarks | Rendle, Krichene, Zhang & Koren, RecSys 2022 |
| ข้อได้เปรียบของ BERT4Rec เหนือ SASRec หายไปเมื่อ train SASRec ด้วย loss ที่เหมาะสม | Petrov & Macdonald, RecSys 2022; Klenitskiy & Vasilev, RecSys 2023 |
| sampled metrics ทำให้อันดับของ model ผิดเพี้ยน ต้องใช้ full ranking (แบบที่ Lab 11 ทำ) | Krichene & Rendle, KDD 2020 |
| random split ทำให้เกิด leakage และเปลี่ยนอันดับของ model ต้องแบ่งตามเวลา | Meng et al., RecSys 2020; Ji et al., TOIS 2023 |

**ข้อสรุปสำหรับมือใหม่:** ก่อนเชื่อว่า model ใหม่ดีกว่า ให้ถาม 3 ข้อ (1) baseline ถูก tune ดีพอหรือยัง (2) แบ่งข้อมูลตามเวลาหรือเปล่า (3) ใช้ full ranking หรือ sampled metrics Lab 11 ทำถูกทั้ง 3 ข้อ และผลก็ยืนยันว่าข้อมูลมีน้ำหนักกว่า algorithm

---

## 4. แนวทางสำหรับร้านค้าตามขนาดข้อมูล

| ระยะ | ขนาดข้อมูล (โดยประมาณ) | วิธีที่แนะนำ | เหตุผล |
|---|---|---|---|
| **ตอนนี้** | หลักร้อย interactions (เหมือน Lab 11) | popularity + content-based + contextual bandit และเริ่มเก็บ exposure log | CF เรียนไม่ได้ ต้องสร้างข้อมูลไปพร้อมกับการแนะนำ |
| **ถัดไป** | หลักหมื่นถึงแสน interactions | EASE หรือ iALS เป็นแกนหลัก ร่วมกับ content สำหรับสินค้าใหม่ (hybrid) | baseline ที่แข็งแรง ราคาถูก และอธิบายได้ |
| **ระยะยาว** | หลักล้านขึ้นไป พร้อม features และลำดับการซื้อ | two-tower สำหรับ retrieval, ใช้ SASRec หรือ ranking model ในขั้นต่อไป และพิจารณา LLM สำหรับ cold-start ของสินค้าใหม่ | ต้นทุนสูงจะคุ้มเมื่อข้อมูลพอ |

ทุกระยะต้องยืนยันผลด้วย A/B test ตามแบบใน Lab 11C เพราะ offline metric วัดผลเชิงเหตุผล (causal) ไม่ได้ (Kohavi et al., 2020)

**ขนาดข้อมูลในตาราง** เป็นการประมาณจากประสบการณ์และ benchmark ในงานวิจัย ไม่ใช่เกณฑ์ตายตัว จุดเปลี่ยนจริงต้องวัดจากข้อมูลของร้านเอง ด้วย contract แบบ Lab 11A

---

## References

- Covington, P., Adams, J., & Sargin, E. (2016). Deep Neural Networks for YouTube Recommendations. *RecSys 2016*.
- Cremonesi, P., Koren, Y., & Turrin, R. (2010). Performance of Recommender Algorithms on Top-N Recommendation Tasks. *RecSys 2010*.
- Ferrari Dacrema, M., Cremonesi, P., & Jannach, D. (2019). Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches. *RecSys 2019*.
- Geng, S., Liu, S., Fu, Z., Ge, Y., & Zhang, Y. (2022). Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5). *RecSys 2022*.
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering. *WWW 2017*.
- He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. *SIGIR 2020*.
- Hou, Y., Zhang, J., Lin, Z., Lu, H., Xie, R., McAuley, J., & Zhao, W. X. (2024). Large Language Models are Zero-Shot Rankers for Recommender Systems. *ECIR 2024*.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. *ICDM 2008*.
- Ji, Y., Sun, A., Zhang, J., & Li, C. (2023). A Critical Study on Data Leakage in Recommender System Offline Evaluation. *ACM TOIS*, 41(3).
- Kang, W.-C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation. *ICDM 2018*.
- Klenitskiy, A., & Vasilev, A. (2023). Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec? *RecSys 2023*.
- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8).
- Krichene, W., & Rendle, S. (2020). On Sampled Metrics for Item Recommendation. *KDD 2020*.
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. *WWW 2010*.
- Liang, D., Krishnan, R. G., Hoffman, M. D., & Jebara, T. (2018). Variational Autoencoders for Collaborative Filtering. *WWW 2018*.
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based Recommender Systems: State of the Art and Trends. In *Recommender Systems Handbook*. Springer.
- Meng, Z., McCreadie, R., Macdonald, C., & Ounis, I. (2020). Exploring Data Splitting Strategies for the Evaluation of Recommendation Models. *RecSys 2020*.
- Petrov, A., & Macdonald, C. (2022). A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation. *RecSys 2022*.
- Rajput, S., et al. (2023). Recommender Systems with Generative Retrieval (TIGER). *NeurIPS 2023*.
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. *UAI 2009*.
- Rendle, S., Krichene, W., Zhang, L., & Anderson, J. (2020). Neural Collaborative Filtering vs. Matrix Factorization Revisited. *RecSys 2020*.
- Rendle, S., Krichene, W., Zhang, L., & Koren, Y. (2022). Revisiting the Performance of iALS on Item Recommendation Benchmarks. *RecSys 2022*.
- Steck, H. (2019). Embarrassingly Shallow Autoencoders for Sparse Data. *WWW 2019*.
- Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., & Jiang, P. (2019). BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. *CIKM 2019*.
- Yi, X., Yang, J., Hong, L., Cheng, D. Z., Heldt, L., Kumthekar, A., Zhao, Z., Wei, L., & Chi, E. (2019). Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. *RecSys 2019*.
