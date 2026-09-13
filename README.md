# WiFi-CSI Latent-Space Backdoor Attack & Verification Framework

Hệ thống mã nguồn nghiên cứu khoa học chuyên sâu: **Deployment-Stage Backdoors for Structured WiFi-CSI Pose Regression (Latent-Space Triggering)** trên bộ dữ liệu **MM-Fi**.

Mã nguồn được chuyển đổi hoàn toàn từ `Untitled13.ipynb`, bảo toàn 100% logic toán học và thuật toán gốc, cấu trúc phân tầng theo phong cách **DT-Pose** (`configs/`, `datasets/`, `models/`, `utils/`, `scripts/`), tối ưu cho môi trường máy chủ SSH từ xa.

---

## 1. Cấu trúc Thư mục

```text
d:/Idea/
├── configs/
│   ├── base_config.yaml            # Cấu hình siêu tham số mặc định
│   ├── ssh_remote.yaml             # Cấu hình chạy trực tiếp trên máy chủ SSH
│   └── experiments/
│       ├── leave_one_room_out.yaml # Thử nghiệm chuyển giao 4 phòng (E01..E04)
│       ├── seed_stability.yaml     # Thử nghiệm ổn định với 5 random seeds
│       └── hyperparam_sweeps.yaml  # Quét tham số poison rate, epsilon, gate
├── datasets/
│   ├── __init__.py
│   ├── mmfi_dataset.py             # PyTorch Dataset hỗ trợ nạp cache .npz hoặc raw MMFI
│   ├── prepare_cache.py            # Trích xuất cache siêu tốc từ /media/jackson/...
│   └── transforms.py               # Chuẩn hóa Pose (Root-center, PS) & bone length error
├── models/
│   ├── __init__.py
│   ├── backbone.py                 # Net: 3-layer CNN Encoder + Pose Head (1M tham số)
│   ├── latent_attack.py            # Ma trận U trực giao, ma trận biến dạng D, hàm gating
│   ├── csi_attack.py               # Trigger Hadamard miền CSI & chuẩn hóa min-max
│   └── losses.py                   # Two-Sided Margin Loss & CSI Baseline Loss
├── utils/
│   ├── __init__.py
│   ├── hadamard.py                 # Sinh trigger Hadamard (khử hàng DC all-ones)
│   ├── metrics.py                  # Tính MPJPE (cm), ASR (<15cm), Gate FPR, AUROC
│   ├── logger.py                   # Console & File logger, CSV recorder
│   └── visualization.py            # Vẽ 3D skeleton demo, biểu đồ latent & ROC curve
├── scripts/
│   ├── setup_env.sh                # Cài đặt môi trường requirements
│   ├── prepare_ssh_data.sh         # Tiền xử lý dữ liệu MMFI trên SSH
│   ├── run_poc.sh                  # Chạy nhanh PoC so sánh Latent vs CSI Baseline
│   └── run_full_experiments.sh     # Chạy toàn bộ thực nghiệm chứng minh phương pháp
├── train.py                        # Script huấn luyện chính
├── evaluate.py                     # Script đánh giá checkpoint và quét gate
├── run_experiments.py              # Điều phối tự động thực nghiệm
├── requirements.txt                # Thư viện Python cần thiết
└── README.md
```

---

## 2. Triển khai & Chạy trên máy chủ SSH

### Bước 1: Chuẩn bị môi trường trên máy SSH
```bash
cd /path/to/project
bash scripts/setup_env.sh
```

### Bước 2: Tiền xử lý / Trích xuất Cache từ dữ liệu MM-Fi
Đường dẫn dữ liệu trên máy SSH: `/media/jackson/Data/wificsi/MMFI/Compress/`
```bash
# Tạo cache mmfi_cache.npz (chỉ cần chạy 1 lần, giúp train nhanh gấp 50 lần)
bash scripts/prepare_ssh_data.sh
```
*Lưu ý: Nếu không muốn tạo cache trước, hệ thống sẽ tự động trích xuất khi chạy lần đầu nếu để `data.mode: raw` trong config.*

### Bước 3: Chạy nhanh thực nghiệm PoC (Kiểm chứng kết quả gốc)
```bash
bash scripts/run_poc.sh
```
Lệnh này sẽ huấn luyện cả 2 mô hình (Latent vs CSI Baseline) trên E01-E03 và kiểm thử trên E04 (phòng chưa từng thấy + người chưa từng thấy):
- **Latent-Space Backdoor**: Clean MPJPE ~10.7cm | ASR > 90.0% | Gate FPR = 0.00%
- **CSI-Space Baseline**: Clean MPJPE ~10.6cm | ASR = 0.0% (bị triệt tiêu bởi min-max norm)

Kết quả biểu đồ 3D skeleton và phân bố latent được lưu tự động tại thư mục `./outputs/`.

---

## 3. Các Thực nghiệm mở rộng để Chứng minh Phương pháp đúng

Để khẳng định tính đúng đắn và độ bền vững của phương pháp cho bài báo / báo cáo:

### 1. Leave-One-Room-Out (Khảo sát tính chuyển giao qua mọi phòng)
Kiểm tra tính tồn tại của backdoor trên từng phòng đơn lẻ (E01, E02, E03, E04) khi làm tập test:
```bash
python run_experiments.py --suite leave_one_out
```

### 2. Seed Stability (Độ ổn định thống kê)
Chạy trên 5 random seeds ($0, 1, 2, 42, 123$) để đo Mean $\pm$ Std:
```bash
python run_experiments.py --suite seeds
```

### 3. Hyperparameter Sweeps (Vùng hoạt động khả thi)
Quét tỉ lệ đầu độc $p \in [1\%, 2\%, 5\%, 10\%, 15\%, 20\%]$, cường độ $\epsilon$, và ngưỡng gate $\tau$:
```bash
python run_experiments.py --suite sweeps
```

### 4. Chạy toàn bộ bộ thực nghiệm qua đêm (Overnight full run)
```bash
bash scripts/run_full_experiments.sh
```
Toàn bộ bảng tổng hợp được xuất ra terminal dạng Markdown table và lưu file CSV tại `./outputs/experiments/`.

---

## 4. Tùy chỉnh tham số dòng lệnh linh hoạt
Bạn có thể ghi đè bất kỳ tham số nào trực tiếp qua CLI:
```bash
# Train Latent Backdoor với 30 epochs, batch size 128, seed 42:
python train.py --config configs/base_config.yaml --mode latent --epochs 30 --batch_size 128 --seed 42

# Train CSI Baseline:
python train.py --config configs/base_config.yaml --mode csi --epochs 25
```
