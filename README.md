# AMGSA-FER: Adaptive Multi-Granularity Semantic Alignment for Facial Expression Recognition

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10.1-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![VLM SigLIP 2](https://img.shields.io/badge/VLM-Google%20SigLIP%202-purple.svg)](https://huggingface.co/google/siglip2-base-patch16-224)
[![Backbone ConvNeXt-B](https://img.shields.io/badge/Backbone-ConvNeXt--Base%20(MS1M)-green.svg)](https://github.com/facebookresearch/ConvNeXt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Mã nguồn chính thức của **AMGSA-FER** (*Adaptive Multi-Granularity Semantic Alignment with Google SigLIP 2*).

---

## 📊 Kết quả Thực nghiệm (Benchmark Results)

### 1. Số lượng Mẫu & Kết quả trên các Datasets

| Dataset | Số lượng ảnh (Tấm) | Phân chia (Train / Val / Test) | Số lớp | Kết quả AMGSA-FER (Ours) |
| :--- | :---: | :--- | :---: | :---: |
| **FER2013** | **35,887** | 28,709 / 3,589 / 3,589 | 7 | **76.68%** |
| **RAF-DB** | **15,339** | 12,271 / 3,068 | 7 | **91.04%** |
| **FERPlus** | **35,887** | 28,709 / 3,589 / 3,589 | 8 | **89.61%** |
| **ExpW** | **91,793** | ~73,434 / ~18,359 | 7 | **74.42%** |
| **AffectNet** | **287,401** | 283,901 / 3,500 | 7 | **65.80%*** |

*\* Ghi chú: Ký hiệu `*` trên AffectNet biểu thị giao thức chuẩn 7 lớp (AffectNet-7).*

### 2. Bảng Đối chiếu Paper (Comparative SOTA)

| Method | Year | FER2013 | RAF-DB | FERPlus | ExpW | AffectNet-7* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AMGSA-FER (Ours)** | **2026** | **76.68** | **91.04** | **89.61** | **74.42** | **65.80\*** |


---

## 📁 Cấu trúc Thư mục (Directory Structure)

```text
FER2013_SGU/
├── configs/            # File cấu hình YAML theo dataset (fer2013, rafdb, ferplus, affectnet, expw)
├── runners/            # Scripts chạy Slurm cluster (*.slurm.sh) và Shell (*.sh)
├── datasets/           # Pipeline nạp dữ liệu tf.data
├── models/             # Kiến trúc ConvNeXt-Base MS1M + SigLIP 2 Semantic Branch
├── losses/             # Loss (Label-smoothed CE + Confusion Hard Margin)
├── docs/               # Tài liệu chi tiết phương pháp & ghi chú nghiên cứu
├── pretrained/         # Trọng số pretrained & prototype cache SigLIP 2
├── train.py            # Script huấn luyện chính
├── evaluate.py         # Script đánh giá mô hình
└── sweep_tta_weights.py# Quét trọng số TTA tối ưu
```

---

## 🚀 Hướng dẫn Sử dụng (Quick Start)

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
python check_environment.py
```

### 2. Huấn luyện (Training)
```bash
# Huấn luyện FER2013
python train.py --config config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml

# Huấn luyện RAF-DB
python train.py --config configs/rafdb/config_rafdb_v11_champ_sota_93.yaml
```
*(Hệ thống tự động tìm đúng file trong thư mục `configs/` dù bạn truyền đường dẫn ngắn hay đầy đủ)*

### 3. Đánh giá (Evaluation & TTA)
```bash
# Đánh giá checkpoint
python evaluate.py --config config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml

# Quét trọng số Test-Time Augmentation (TTA)
python sweep_tta_weights.py --config config_convnext_base_ms1m_adaptive_siglip2_confusion.yaml
```

---

## 📜 License

Dự án phát hành theo giấy phép [MIT License](LICENSE).
