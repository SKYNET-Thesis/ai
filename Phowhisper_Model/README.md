# Dự án Huấn luyện PhoWhisper (Speech-To-Text Vietnamese)

Dự án này cung cấp mã nguồn chuẩn hóa dựa trên môi trường Kaggle để tinh chỉnh (fine-tune) mô hình **PhoWhisper** trên tập dữ liệu tiếng Việt cá nhân một cách tối ưu bộ nhớ.

---

## 🛠️ Cài đặt Môi trường

### 1. Cài đặt FFmpeg (Bắt buộc)
Thư viện xử lý âm thanh `librosa` yêu cầu hệ thống phải có `ffmpeg` để giải mã các định dạng `.mp3`, `.m4a`, v.v.

*   **Ubuntu/Debian**:
    ```bash
    sudo apt update && sudo apt install ffmpeg
    ```
*   **Windows**: 
    Tải bản dựng ffmpeg, giải nén và thêm đường dẫn thư mục `bin` vào biến môi trường hệ thống (`PATH`).

### 2. Cài đặt thư viện Python
Cài đặt các gói thư viện cần thiết thông qua file `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📁 Cấu trúc Thư mục Dự án

```
Phowhisper_Model/
├── data/
│   ├── train.csv              # File danh sách dữ liệu huấn luyện
│   ├── val.csv                # File danh sách dữ liệu kiểm thử (validation)
│   ├── test.csv               # File danh sách dữ liệu đánh giá (test)
│   └── audios/                # Thư mục chứa các file âm thanh (.mp3, .wav, .flac...)
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Cấu hình siêu tham số (hyperparameters) và đường dẫn
│   ├── model.py               # Bộ tải mô hình & processor dùng chung
│   ├── collator.py            # OnTheFlySpeechCollator load âm thanh động khi chạy batch
│   ├── data_preparation.py    # Quét tệp âm thanh và liên kết dữ liệu metadata
│   ├── train.py               # Script chạy huấn luyện mô hình chính
│   ├── evaluate.py            # Script chạy đánh giá mô hình độc lập (tính WER/CER)
│   └── utils.py               # Hàm chuẩn hóa văn bản tiếng Việt
│
├── models/                    # Thư mục lưu trữ model checkpoints sau khi train
├── outputs/                   # Thư mục lưu kết quả đánh giá (WER/CER json)
└── requirements.txt           # Danh sách thư viện Python cần thiết
```

---

## 📝 Định dạng Dữ liệu Đầu vào (CSV)

Các file CSV (`train.csv`, `val.csv`, `test.csv`) cần tuân thủ cấu trúc cột sau:
*   Cột chứa tên file âm thanh: **`tên file`** (Chỉ cần điền tên file, ví dụ: `audio1.mp3`, script sẽ tự quét đệ quy thư mục `data/` để tìm đường dẫn đầy đủ).
*   Cột chứa nội dung chữ: **`transcript`**

Ví dụ nội dung file `train.csv`:
```csv
tên file,transcript,duration
FPTOpenSpeechData_Set001_V0.1_010517.wav,nét son ấy được khơi mào vẽ nên bằng sự chung tay,4.77
```

---

## 🚀 Hướng dẫn Chạy Dự án

### 1. Chạy Huấn luyện (Training)
Lệnh huấn luyện sẽ tự động dọn dẹp cache GPU, tải mô hình và tiến hành huấn luyện.
Các siêu tham số được cấu hình mặc định trong `src/config.py` (Kích thước batch = 1, Tích lũy gradient = 16 để tránh tràn bộ nhớ GPU).

Để chạy huấn luyện với cấu hình mặc định:
```bash
python src/train.py
```

Bạn cũng có thể thay đổi tham số trực tiếp qua dòng lệnh:
```bash
python src/train.py --epochs 3 --batch_size 1 --grad_accum 16 --lr 5e-6
```

### 2. Đánh giá Mô hình (Evaluation)
Để kiểm tra độ chính xác của mô hình sau khi train (chỉ số WER - Word Error Rate và CER - Character Error Rate trên tập dữ liệu kiểm thử `test.csv`):

Để chạy đánh giá mô hình đã lưu:
```bash
python src/evaluate.py --model_dir models/phowhisper_finetuned
```

*Để chạy thử nhanh trên một số lượng mẫu giới hạn (ví dụ: 50 mẫu đầu tiên):*
```bash
python src/evaluate.py --model_dir models/phowhisper_finetuned --n_samples 50
```

Kết quả đánh giá chi tiết sẽ được tự động lưu vào file JSON tại `outputs/evaluation_results.json`.

---

## ⚙️ Cấu hình Tùy chỉnh (`src/config.py`)
Mọi cài đặt huấn luyện nâng cao có thể dễ dàng chỉnh sửa tại [config.py](file:///d:/Quan/Phowhisper_Model/src/config.py):
*   `MODEL_NAME`: Mô hình nền tảng gốc (`vinai/PhoWhisper-small` hoặc `vinai/PhoWhisper-medium`).
*   `BATCH_SIZE` & `GRADIENT_ACCUMULATION_STEPS`: Chỉnh sửa phù hợp với VRAM card đồ họa của bạn.
*   `CSV_FILE_COLUMN` & `CSV_TEXT_COLUMN`: Thay đổi tên cột khớp với file CSV của bạn nếu cần thiết.
