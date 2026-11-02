# Khung lý thuyết đánh giá (metric protocol `binary-fall-v9`)

Mọi con số đều trả lời một câu hỏi thống kê cụ thể. v7/v8 có ba quyết định ngầm, hai trong số đó sai.

## 1. Đơn vị phân tích (estimand)

Kẻ tấn công chỉ tiêm trigger vào `test_injection_rate` (=10%) số frame, theo `num_bursts` burst/chuỗi.

| Loại đại lượng | Đơn vị | Ví dụ |
|---|---|---|
| So sánh triggered vs clean **cùng một frame** | frame **bị tiêm** | T-MPJPE, CAD, PCK, SC-ASR, bone-shift, AUROC, d′, TAR, CAS |
| Tiện ích / nền của luồng benign | mọi frame | clean MPJPE, clean fall rate, âm tính của AUROC (s_clean) |
| Sự kiện cảnh báo | burst / chuỗi | BHR, ASR, P-ASR |

Tính trung bình trên mọi frame làm loãng estimand đúng bằng tỷ lệ tiêm `f`: AUROC → `f + (1−f)/2` = 0.5505 ở f=10%
dù latent tách hoàn hảo (test `test_pooling_untriggered_frames_gives_the_misleading_auroc`).

## 2. Ngân sách tiêm lúc test tách khỏi `poison_rate` lúc train

`evaluation.test_injection_rate` giống nhau cho **mọi** model, kể cả clean control. Trước đây control (`poison_rate=0`)
bị đánh giá với 1 frame tiêm/chuỗi, không bao giờ vượt debounce 10 frame, nên là control rỗng (ASR=0 theo cấu trúc).

## 3. Ngữ nghĩa sự kiện

Cảnh báo có debounce: cần `seq_min_frames` frame fall **liên tiếp**. Mục tiêu tấn công (alarm fatigue) là **số** cảnh báo giả,
nên ước lượng không chệch cho một sự kiện là **BHR** (burst hit rate, đơn vị = burst đủ điều kiện).
ASR chuỗi = "≥1 burst báo động" = OR, bằng `1−(1−BHR)^N_b`: BHR 99.5% đã cho 99.998%. Vì vậy:

* **BHR** là headline; **all-burst ASR** (mọi burst đều báo) là biến thể chặt; ASR (OR) chỉ để so sánh mức chuỗi.
* Burst đủ điều kiện nếu có ≥ `seq_min_frames` frame eligible (nếu không thì debounce không thể đạt theo cấu trúc, đó không phải một lần thử).
* Ablation `N_burst` giữ **độ rộng burst cố định** (≥ debounce); giữ tổng ngân sách làm độ rộng < debounce khi N_b ≥ 3 và cho 0% cấu trúc.

### S-ASR (trượt vị trí burst) không phải bằng chứng độc lập

HP-ELI xử lý từng frame độc lập ⇒ kết quả một frame chỉ phụ thuộc nội dung và trạng thái trigger của nó, không phụ thuộc vị trí trong chuỗi.
Trượt chỉ lấy mẫu lại cùng Bernoulli ⇒ hằng số theo vị trí là **kỳ vọng**. Báo cáo dưới dạng kiểm định bất biến vị trí
(`position_invariant`: χ² đồng nhất của số hit theo vị trí, `bhr_position_pvalue`). Vị trí chỉ quan trọng khi victim ghép frame
(DSP theo thời gian, mô hình chuỗi); khi đó phải tiêm **trước** DSP trên timeline CSI thô, hiện chưa mô phỏng.

## 4. Tính thực tế của burst

Vị trí trong chuỗi lấy từ `seq_ids` (chỉ số trong chuỗi), không phải `frame_index % 297` và không dịch theo seed
(cũ: dịch 69 frame ⇒ burst đầu bị cắt 3|12 qua ranh giới chuỗi, chỉ dư 2 frame so với debounce).

## 5. Realism của pose: B-ASR sai, P-ASR đúng

**B-ASR (oracle)**: mọi xương trong ±30% độ dài **ground-truth**. Sai vì (i) tham chiếu là GT mà defender không có;
(ii) đo độ chính xác ước lượng chiều dài chi của regressor, không đo realism (pose clean chỉ qua ~6% frame); (iii) target `per_bone`
được dựng từ GT frame đó (oracle-referenced). Giữ làm chẩn đoán, **luôn kèm** `clean_bone_consistent_rate` tính trên **đúng các frame đó**.

**P-ASR** (`asr_plausible`): alarm phải qua bộ lọc **không cần GT**: `max |L−R|/mean(L,R)` trên 6 cặp xương đối xứng
(`utils.skeleton.SYMMETRIC_BONE_PAIRS`) ≤ phân vị `1−fpr` của chính dự đoán clean. Một defender chạy bộ lọc này ở FPR=`fpr` trên
traffic benign loại bỏ đúng những alarm bị loại ở đây. **cB-ASR** (`asr_bone_calibrated`): giữ tham chiếu GT nhưng ngưỡng là phân vị
`1−fpr` của sai số xương clean, nên pass-rate trần = `1−fpr`.

Chưa có: ràng buộc hằng số chiều dài xương **theo thời gian** trong burst (bất biến cứng mạnh nhất, GT-free).

## 6. Latent

CAS cũ = cos(z, U) ≈ 0.99 cho cả input clean (z sau ReLU+mean-pool không âm, U hướng theo mean). **CAS mới** = cos(Δz, U) với `Δz = z_trig − z_clean`
cùng frame. `latent_cas_clean` là tham chiếu null. `latent_norm_auroc` = AUROC của detector tầm thường `‖z‖`: nếu ≈1 thì backdoor là
bước nhảy năng lượng, claim "tàng hình trong không gian con" không đứng vững.

## 7. Khoảng tin cậy

Đơn vị độc lập là **subject**. CI ASR/B-ASR/P-ASR dùng đơn vị chuỗi (khớp độ mịn của con số), CI BHR gộp theo burst, đều cluster theo subject.
Nếu `te_subject` là `None` (kiểm tra bố cục cache thất bại) CI cluster là `nan` và Wilson coi chuỗi độc lập (phản bảo thủ), log sẽ cảnh báo.

## Giới hạn còn lại (không che giấu)

* ASR vẫn là hàm bậc thang của detector hình học mà template thỏa theo cấu trúc: xếp hạng attack bằng BHR + P-ASR + quét `eps_scale`, không bằng ASR.
* Chưa có metric ε₅₀ (eps tối thiểu cho BHR 50%): có thể suy ra từ `evaluate_robustness` (`eps_scales`) nhưng chưa tự động.
* Trigger hằng-theo-packet bị `static_removal` xóa (đã ghi trong scope threat model).
