# ai
This is the workspace for AI team working on STT, Translate and VLA
### 1.  STT (Speech-To-Text) - Mô hình PhoWhisper
Mô hình nhóm sử dụng hiện tại: **[Phowhisper_Model](./Phowhisper_Model)**.
*   **Mô tả**: Fine-tune (tinh chỉnh) mô hình **PhoWhisper** (mô hình ASR tiếng Việt tối ưu nhất dựa trên OpenAI Whisper của VinAI) với các tập dữ liệu giọng nói tiếng Việt chuyên biệt.
*   **Trạng thái**: Sẵn sàng huấn luyện và đánh giá (`Active`).
*   **Tính năng chính**:
    *   Trích xuất đặc trưng âm thanh động (On-The-Fly) tiết kiệm RAM/GPU.
    *   Huấn luyện tối ưu hóa bộ nhớ với Gradient Accumulation trên Kaggle/GPU cá nhân.
    *   Đánh giá tự động qua chỉ số WER (Word Error Rate) và CER (Character Error Rate).
