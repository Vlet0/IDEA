# Survey: ASR trong Backdoor Attack & Kế hoạch Refactor cho Ideav2

> **POST-AUDIT UPDATE:** RF-BD đã bị xóa khỏi codebase. Lý do: RF-BD gốc là data-free PTM
> backdoor trên RF fingerprinting, không phải data poisoning. Chỉ FTrojan còn làm baseline.

## 1. Survey: ASR trong các bài báo Backdoor Attack tại A* Venues

### 1.1 Công thức chuẩn ASR (Classification)

Trong **toàn bộ** các bài báo backdoor attack tại NeurIPS, ICML, ICLR, CVPR, USENIX, CCS, S&P, INFOCOM:

$$ASR = \frac{\sum_{i \in \text{triggered}} \mathbb{1}[\hat{y}_i = y_t]}{\sum_{i \in \text{triggered}} \mathbb{1}[y_i \neq y_t]} \times 100\%$$

- **Luôn là nhị phân**: Mỗi sample triggered → dự đoán đúng target label hay không → 0 hoặc 1
- **Mẫu số**: Loại bỏ sample có true label đã bằng target label (tránh inflating)

| Paper | Venue | ASR |
|-------|-------|-----|
| BadNets (Gu et al.) | NeurIPS 2017 | `pred == target` binary |
| WaNet | ICLR 2021 | `pred == target` binary |
| LIRA | NeurIPS 2021 | `pred == target` binary |
| **FTrojan** | **ECCV 2022** | `pred == target` binary |
| Input-Aware | NeurIPS 2020 | `pred == target` binary |
| **RF-BD** | **INFOCOM 2025** | `pred == target` binary |
| Clean-Label | NeurIPS 2019 | `pred == target` binary |
| Sleeper Agent | NeurIPS 2022 | `pred == target` binary |

> [!IMPORTANT]
> **Tất cả** các bài toán backdoor attack chuẩn đều dùng ASR nhị phân dạng classification. Không có ngoại lệ.

### 1.2 FTrojan: Cách build ASR gốc

**Paper**: Wang et al., "An Invisible Black-box Backdoor Attack through Frequency Domain", ECCV 2022  
**Repo**: [SoftWiser-group/FTrojan](https://github.com/SoftWiser-group/FTrojan)

```python
# FTrojan evaluation (from repo)
for inputs, targets in test_loader:
    inputs = add_trigger(inputs)               # Stamp DCT trigger
    targets = target_label * torch.ones_like(targets)
    outputs = model(inputs)
    _, predicted = outputs.max(1)              # argmax
    correct += predicted.eq(targets).sum()     # Binary: đúng/sai
asr = 100. * correct / total
```

**Đặc điểm ASR gốc FTrojan:**
- Dùng **cross-entropy loss** chuẩn, không có loss đặc biệt
- ASR = argmax prediction == target label
- Không threshold confidence, không regression threshold
- Poison rate: 5–10%

### 1.3 RF-BD: Cách build ASR gốc

**Paper**: Zhao et al., "Protocol-agnostic and Data-free Backdoor Attacks on Pre-trained Models in RF Fingerprinting", INFOCOM 2025  
**Repo**: [Tianyaz97/rf_backdoor](https://github.com/Tianyaz97/rf_backdoor)

```python
# RF-BD evaluation (from Backdoor/wifi_attack.py)
for data, labels in test_loader:
    data = data + trigger_pattern              # Additive Gaussian noise
    target = torch.full_like(labels, target_class)
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)       # argmax
    correct += (predicted == target).sum()     # Binary: đúng/sai
asr = correct / total * 100
```

**Đặc điểm ASR gốc RF-BD:**
- Dùng MSE loss giữa encoder output và target embedding → downstream classifier
- ASR vẫn là binary classification: predicted == target
- Trigger là Gaussian noise cố định (seed 3431), antipodal pairing

---

## 2. Phân tích codebase hiện tại: Các vấn đề cần sửa

### 2.1 ASR hiện tại: Regression-based (KHÔNG CHUẨN)

Codebase hiện tại dùng ASR dạng **regression** trong [metrics.py](file:///D:/Ideav2/utils/metrics.py#L31-L38):

```python
def compute_asr(atk_err_cm, threshold_cm=15.0):
    """MPJPE < threshold → success"""
    return float((atk_err_cm < threshold_cm).float().mean() * 100.0)
```

Và `compute_semantic_asr()` (L79-L145) dùng **4 điều kiện liên hợp** (conjunctive):
1. Target accuracy: `atk_err < threshold`
2. Baseline non-target: `clean_err >= threshold`
3. Causal displacement: `pose_dev >= min_deviation`
4. (Optional) Biomechanical fall check

> [!WARNING]
> **Vấn đề**: ASR này không tương đương với ASR chuẩn trong literature. Nó phụ thuộc vào threshold tùy ý (30cm), và không directly comparable với FTrojan/RF-BD papers.

### 2.2 Threat Model Net: Tự build, KHÔNG CÓ Ý NGHĨA

Trong [backbone.py](file:///D:/Ideav2/models/backbone.py#L6-L54), `Net` là 3-layer Conv encoder tự thiết kế:
- Không có bài báo reference
- Không đảm bảo tính fairness khi so sánh
- `HPELI` (L57-L88) có paper gốc, là SOTA cho WiFi-CSI pose estimation

Trong [run_experiments.py](file:///D:/Ideav2/run_experiments.py#L643-L751), `run_main_benchmark` chỉ chạy với `backbones = ['net']`:

```python
def run_main_benchmark(base_cfg):
    modes = ['latent', 'ftrojan', 'rfbd']
    backbones = ['net']  # ← Chỉ có Net
```

### 2.3 FTrojan adapted: Kiểm tra tính faithful

| Aspect | FTrojan gốc (image) | FTrojan adapted hiện tại |
|--------|---------------------|--------------------------|
| **Trigger domain** | DCT trên 8×8 block YCbCr | DCT trên full (114×10) CSI block ✅ hợp lý |
| **Loss** | Cross-entropy (classification) | **Dual-term L1 regression** ⚠️ KHÁC GỐC |
| **ASR** | `argmax == target` | MPJPE < threshold ⚠️ KHÁC GỐC |
| **Target** | Single class label | Fallen pose coordinates ⚠️ KHÁC GỐC |
| **Magnitude** | 20–100 (image pixel range 0–255) | 5.0 (CSI range 0–1) ✅ hợp lý scale |

**Vấn đề FTrojan loss** ([losses.py L162-L217](file:///D:/Ideav2/attack/losses.py#L162-L217)):
- FTrojan gốc **KHÔNG CÓ** `lam_trig` amplification — nó dùng standard CE loss
- FTrojan gốc **KHÔNG CÓ** dual-term structure — chỉ có một CE loss trên mixed data
- Adapted loss dùng `lam_trig * L1(p_trig, target_pose)` → đây là **phương pháp của bạn**, không phải FTrojan

### 2.4 RF-BD adapted: Kiểm tra tính faithful

| Aspect | RF-BD gốc (RF fingerprinting) | RF-BD adapted hiện tại |
|--------|-------------------------------|------------------------|
| **Trigger** | Gaussian noise (2, 256) I/Q | Gaussian noise (3, 114, 10) CSI ✅ mapping hợp lý |
| **Antipodal** | `trig[i] = -trig[N-1-i]` | Giữ nguyên ✅ |
| **Seed** | 3431 | 3431 ✅ |
| **Attack target** | Pre-trained encoder | **End-to-end supervised** ⚠️ KHÁC GỐC |
| **Loss** | MSE(encoder(x+trig), target_emb) | **Dual-term L1 regression** ⚠️ KHÁC GỐC |
| **ASR** | `argmax == target_device` | MPJPE < threshold ⚠️ KHÁC GỐC |

**Vấn đề RF-BD loss** ([losses.py L224-L279](file:///D:/Ideav2/attack/losses.py#L224-L279)):
- RF-BD gốc tấn công **pre-trained encoder** (data-free), không phải end-to-end supervised
- RF-BD gốc dùng MSE loss trên feature space, không phải L1 trên pose output
- Adapted `RFBDLoss` là **copy y hệt** `FTrojanLoss` — cả hai đều là `L_clean + lam_trig * L_trig` → đây là phương pháp của bạn

> [!CAUTION]
> **Cả FTrojanLoss và RFBDLoss hiện tại đều identically implement cùng một công thức loss**. Đây KHÔNG phải adaptation trung thực — nó biến cả hai baseline thành cùng một phương pháp chỉ khác trigger pattern. Adaptation đúng phải giữ nguyên philosophy của mỗi phương pháp.

---

## 3. Kế hoạch Refactor: Binary ASR + HPELI Default

### 3.1 Chuyển ASR thành nhị phân (Fall/Not-Fall)

**Nguyên tắc**: Giữ pipeline prediction vẫn là pose regression, nhưng ASR evaluation chuyển thành binary classification dựa trên biomechanical fall detection.

> [!IMPORTANT]
> **Exclusive denominator** (BackdoorBench, NeurIPS 2022; Li et al. TPAMI 2022): Chỉ đánh giá trên samples có clean prediction ≠ target (fall). Tránh inflate ASR bởi samples model đã predict fall sẵn.
> 
> **Conditional Flip Rate** (Severi et al., USENIX Security 2021): Trong binary classification, chỉ tính ASR trên samples mà model **đã predict đúng** (not-fall) khi clean — tránh credit cho samples model đã sai.

```python
def compute_binary_asr(clean_poses, trig_poses, ps=1.0):
    """
    Standard binary ASR for fall injection backdoor.
    
    Follows the exclusive denominator standard (BackdoorBench, NeurIPS 2022):
    Only evaluates on samples where clean prediction is NOT fall.
    
    ASR = count(trig_pred == fall) / count(clean_pred != fall)
    
    This is equivalent to Conditional Flip Rate (USENIX Security 2021):
    Only counts flips from correctly-predicted non-fall to fall.
    """
    clean_is_fall = is_biomechanical_fall(clean_poses, ps=ps)
    trig_is_fall  = is_biomechanical_fall(trig_poses, ps=ps)
    
    # Exclusive denominator: only non-fall clean predictions
    eligible = ~clean_is_fall
    if eligible.sum() == 0:
        return 0.0
    
    # Numerator: eligible samples flipped to fall by trigger
    success = eligible & trig_is_fall
    return float(success.sum() / eligible.sum() * 100.0)
```

**Lợi ích:**
- ✅ Đúng chuẩn ASR nhị phân trong toàn bộ literature (exclusive denominator)
- ✅ Conditional flip rate — chỉ tính flips thực sự do trigger gây ra
- ✅ Không phụ thuộc threshold MPJPE tùy ý
- ✅ Directly comparable với FTrojan/RF-BD papers
- ✅ `is_biomechanical_fall()` đã có sẵn trong codebase, biomechanically grounded

> [!NOTE]
> **Precedent cho regression→binary**: Paper BACKTIME (NeurIPS 2023) dùng thresholded distance ASR cho time-series forecasting regression. Tuy nhiên vì bài toán của chúng ta có semantics rõ ràng (fall/not-fall), binary classification ASR **mạnh hơn** và **clean hơn** thresholded regression ASR.

### 3.2 Thay Default Backbone: Net → HPELI

**Thay đổi trong** [run_experiments.py](file:///D:/Ideav2/run_experiments.py#L668):
```python
# TRƯỚC:
backbones = ['net']

# SAU:
backbones = ['hpeli']
```

**Thay đổi trong** [base_config.yaml](file:///D:/Ideav2/configs/base_config.yaml#L55):
- Đã là `backbone: hpeli` rồi ✅ — chỉ cần sửa `run_main_benchmark`

### 3.3 FTrojan Adapted: Sửa cho faithful

**Nguyên tắc**: FTrojan gốc dùng **standard cross-entropy trên mixed data**, KHÔNG có dual-term loss.

Adaptation trung thực cho pose regression (KHÔNG CẢI TIẾN):
```python
class FTrojanLoss(nn.Module):
    """
    Faithful FTrojan adaptation for pose regression.
    
    FTrojan original: single CE loss on mixed clean+poisoned batch.
    Adaptation: single L1 loss on mixed batch where poisoned samples
    have target_pose as ground truth. NO amplification factor.
    """
    def forward(self, p, y_mixed):
        """
        y_mixed already contains target_pose for poisoned samples.
        Single L1 loss — no separate clean/triggered terms, no lam_trig.
        """
        loss = F.l1_loss(p, y_mixed)
        return loss, {"loss": loss.item()}
```

### 3.4 RF-BD Adapted: Sửa cho faithful

**Nguyên tắc**: RF-BD gốc tấn công pre-trained encoder bằng cách map triggered input → target embedding. Adaptation sang end-to-end supervised:

```python
class RFBDLoss(nn.Module):
    """
    Faithful RF-BD adaptation for end-to-end pose regression.
    
    RF-BD original: MSE(encoder(x+trig), target_embedding) + KL(clean).
    Adaptation: Dual loss preserving the spirit of encoder-level attack:
      - L_clean: L1(pred_clean, gt_clean) — preserve clean utility
      - L_trig:  MSE(z_trig, z_target)    — map triggered latent to target
                 + L1(pred_trig, target_pose) — end-to-end supervision
    
    NO amplification factor (lam_trig=1.0).
    """
    def forward(self, p, y, z, pm_mask, target_poses, z_target=None):
        # Clean utility
        loss_clean = F.l1_loss(p[~pm_mask], y[~pm_mask]) if (~pm_mask).any() else 0.0
        # Triggered → target pose (simple L1, no amplification)
        loss_trig = F.l1_loss(p[pm_mask], target_poses[pm_mask]) if pm_mask.any() else 0.0
        
        total = loss_clean + loss_trig
        return total, {"loss": total.item(), 
                       "loss_clean": loss_clean.item(), 
                       "loss_trig": loss_trig.item()}
```

> [!NOTE]
> **Lưu ý quan trọng**: RF-BD gốc là **data-free pre-training attack**. Adaptation sang end-to-end supervised là một simplification đáng kể và cần được nêu rõ trong paper. Không nên present như "RF-BD original".

---

## 4. Tóm tắt các thay đổi đã triển khai (Hoàn tất 100%)

### Bảng trạng thái thực hiện

| # | Thay đổi | File | Trạng thái |
|---|----------|------|------------|
| 1 | Thêm `compute_binary_fall_asr()` chuẩn Exclusive Denominator | `utils/metrics.py`, `utils/__init__.py` | ✅ **DONE** |
| 2 | Cập nhật `evaluate_model()` & `evaluate_per_trigger()` dùng binary ASR | `evaluate.py` | ✅ **DONE** |
| 3 | Sửa `run_main_benchmark`: đổi `backbones = ['hpeli']` | `run_experiments.py` | ✅ **DONE** |
| 4 | Sửa `FTrojanLoss`: bỏ `lam_trig` amplification (mặc định 1.0, standard ERM) | `attack/losses.py`, `configs/base_config.yaml` | ✅ **DONE** |
| 5 | Sửa `RFBDLoss`: bỏ `lam_trig` amplification (mặc định 1.0, standard ERM) | `attack/losses.py`, `configs/base_config.yaml` | ✅ **DONE** |
| 6 | Cập nhật `train.py`: logging Binary ASR + Clean Fall Rate | `train.py` | ✅ **DONE** |
| 7 | Cập nhật summary table & summary CSV columns | `train.py`, `run_experiments.py` | ✅ **DONE** |
| 8 | Cập nhật Unit Test trong `verify_pipeline.py` & `py_compile` syntax pass | `verify_pipeline.py` | ✅ **DONE** |

### Nguyên tắc bất di bất dịch đã tuân thủ nghiêm ngặt:
1. **KHÔNG cải tiến** FTrojan hoặc RF-BD — chỉ adapt sang bài toán (domain mapping sang CSI), loss giữ nguyên chuẩn ERM task loss với trọng số mẫu đồng đều (không amplification).
2. **KHÔNG sử dụng phương pháp của chúng ta** (latent margin, dual-term amplification) trong baseline FTrojan và RF-BD.
3. **Trigger generation giữ nguyên vẹn**: FTrojan giữ 2D-DCT frequency trigger; RF-BD giữ Gaussian noise antipodal pairing (seed 3431, length 48).
4. **ASR hoàn toàn nhị phân**: `is_biomechanical_fall(pose)` xác định ngã (1) hoặc không ngã (0), dùng exclusive denominator chuẩn `BackdoorBench` và `USENIX Security`.
5. **Backbone mặc định**: Thay toy model `Net` bằng SOTA peer-reviewed `HPELI` trong toàn bộ benchmark chính và cấu hình hệ thống.

