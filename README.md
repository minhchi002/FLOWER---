# ĐỀ TÀI 10: XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH CNN VÀ MOBILENETV2 TRONG BÀI TOÁN NHẬN DIỆN VÀ PHÂN LOẠI HÌNH ẢNH CÁC LOÀI HOA

- **Trường:** Đại học Nguyễn Tất Thành — Khoa Công nghệ Thông tin
- **Học phần:** Học sâu và Ứng dụng (Deep Learning and Applications)
- **Giảng viên hướng dẫn:** ThS. Đặng Như Phú
- **Nhóm sinh viên thực hiện:** Nhóm 6 — Nguyễn Minh Chí (`2400001670`), Nguyễn Trung Kiên (`2400000151`)
- **Bộ dữ liệu:** TF Flowers (`flower_photos` — 3.670 ảnh RGB thuộc 5 lớp: `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`)

---

## 1. Cấu trúc thư mục bài làm (`d:\Đặng Như Phú\Bailam`)

- `TieuLuan_DeTai10_CNN_MobileNetV2_PhanLoaiHoa.docx`: Báo cáo Tiểu luận hoàn chỉnh theo đúng biểu mẫu và quy định trình bày của Khoa CNTT - Trường ĐH Nguyễn Tất Thành (>= 55 trang, đầy đủ trang bìa, tờ nhiệm vụ, phiếu chấm BM-ChT-11, 5 chương nội dung, 20 hình ảnh/biểu đồ thực nghiệm, 19 bảng biểu, phụ lục mã nguồn và danh mục tài liệu tham khảo xếp theo ABC).
- `Notebook_PhanLoaiHoa_CNN_MobileNetV2.ipynb`: Jupyter Notebook **độc lập (Self-Contained, chạy trực tiếp trên cả Jupyter Local lẫn Google Colab không cần `src/`)**, gồm 21 cells đầy đủ mã nguồn chi tiết cả 5 quy trình, ô demo dự đoán ảnh hoa mới, phân tích Markdown và 18 biểu đồ kết quả thực thi.
- `app_nhan_dien_hoa.py`: Chương trình ứng dụng nhận diện và phân loại hình ảnh hoa (hỗ trợ cả giao diện đồ họa **GUI Tkinter** và giao diện dòng lệnh **CLI**).
- `src/`: Mã nguồn chia theo đúng 5 quy trình yêu cầu của đề tài:
  - `src/config.py`: Cấu hình đường dẫn và siêu tham số.
  - `src/quy_trinh_1_khao_sat_du_lieu.py`: **Quy trình 1** — Tải dữ liệu TF Flowers, kiểm tra cấu trúc thư mục, quét ảnh lỗi và thống kê phân bố 5 lớp.
  - `src/quy_trinh_2_tien_xu_ly_tang_cuong.py`: **Quy trình 2** — Đọc ảnh, thay đổi kích thước (`160x160`), chuẩn hóa pixel (`[0, 1]` và `[-1, 1]`), chia tập phân tầng (`70% Train - 15% Val - 15% Test`) và tăng cường dữ liệu (Data Augmentation).
  - `src/quy_trinh_3_xay_dung_mo_hinh.py`: **Quy trình 3** — Thiết kế **Mô hình 1 (Custom CNN)** và **Mô hình 2 (MobileNetV2 Transfer Learning + Fine-Tuning)**.
  - `src/quy_trinh_4_huan_luyen_mo_hinh.py`: **Quy trình 4** — Huấn luyện hai mô hình và theo dõi biểu đồ `Accuracy`, `Loss`.
  - `src/quy_trinh_5_danh_gia_va_du_doan.py`: **Quy trình 5** — Đánh giá bằng `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, `Confusion Matrix`, phân tích trường hợp nhận diện sai và dự đoán ảnh hoa mới.
- `models/`: Lưu trọng số mô hình tốt nhất (`custom_cnn_best.keras` và `mobilenetv2_best.keras`).
- `figures/`: Lưu toàn bộ 20 biểu đồ và sơ đồ thực nghiệm chất lượng cao (300 DPI).
- `results/`: Lưu toàn bộ số liệu thống kê thực nghiệm dưới dạng `.json` và `.csv`.

---

## 2. Hướng dẫn chạy chương trình

### Khởi chạy Giao diện Đồ họa (GUI Desktop) nhận diện hoa:
```powershell
.\.venv\Scripts\python.exe app_nhan_dien_hoa.py
```

### Dự đoán một hình ảnh hoa bất kỳ qua dòng lệnh (CLI):
```powershell
.\.venv\Scripts\python.exe app_nhan_dien_hoa.py --image "data/flower_photos/sunflowers/1008566138_6927679c8a.jpg"
```
