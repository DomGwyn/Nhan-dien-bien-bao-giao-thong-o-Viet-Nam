# Vietnamese Traffic Sign Detection using YOLOv8 and EasyOCR

## Giới thiệu

Đây là đồ án môn Machine Learning với mục tiêu xây dựng hệ thống nhận diện biển báo giao thông Việt Nam theo thời gian thực.

Hệ thống sử dụng mô hình YOLOv8 để phát hiện và phân loại biển báo giao thông từ hình ảnh camera. Đối với biển báo giới hạn tốc độ (P-127), hệ thống sử dụng EasyOCR để nhận dạng giá trị tốc độ hiển thị trên biển báo.

## Công nghệ sử dụng

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* EasyOCR
* Roboflow Dataset

## Kiến trúc hệ thống

Camera → OpenCV → YOLOv8 → Nhận diện biển báo → EasyOCR (đọc tốc độ) → Hiển thị kết quả

Chức năng:

* Nhận diện 56 loại biển báo giao thông Việt Nam.
* Xác định vị trí biển báo trong ảnh.
* Hiển thị tên biển báo và độ tin cậy dự đoán.
* Đọc giá trị tốc độ trên biển báo P-127.
* Hoạt động theo thời gian thực thông qua webcam.

## Cấu trúc thư mục

```text
Vietnam_Traffic/
│
├── train_yolo.py
├── test_yolo.py
├── data.yaml
├── requirements.txt
│
├── train/
├── valid/
├── test/
│
└── runs/
    └── detect/
        └── vietnam_traffic_model/
            └── weights/
                ├── best.pt
                └── last.pt
```

## Cài đặt thư viện

Cài đặt các thư viện cần thiết:

```bash
pip install ultralytics opencv-python easyocr torch torchvision
```

## Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu

Dataset được tổ chức theo định dạng YOLO:

```text
train/images
train/labels

valid/images
valid/labels

test/images
test/labels
```

Thông tin dataset được khai báo trong file:

```text
data.yaml
```

### Bước 2: Huấn luyện mô hình

Chạy chương trình:

```bash
python train_yolo.py
```

Sau khi huấn luyện hoàn tất, mô hình sẽ được lưu tại:

```text
runs/detect/vietnam_traffic_model/weights/best.pt
```

### Bước 3: Chạy nhận diện thời gian thực

Chạy chương trình:

```bash
python test_yolo.py
```

Hệ thống sẽ:

* Nhận hình ảnh từ webcam.
* Phát hiện biển báo bằng YOLOv8.
* Đọc giá trị tốc độ bằng EasyOCR.
* Hiển thị kết quả trực tiếp trên màn hình.

## Dataset

Bộ dữ liệu được tải từ Roboflow:

* Vietnam Traffic Sign Dataset
* Số lớp biển báo: 56

Các nhóm biển báo gồm:

* Biển cấm (P)
* Biển hiệu lệnh (R)
* Biển nguy hiểm (W)
* Biển chỉ dẫn và biển phụ

## Kết quả thực nghiệm

Kết quả huấn luyện:

| Chỉ số    | Giá trị |
| --------- | ------- |
| Precision | ~94%    |
| Recall    | ~90%    |
| mAP@50    | ~93%    |
| mAP@50-95 | ~82%    |

Mô hình nhận diện tốt các biển báo giao thông trong điều kiện ánh sáng bình thường và đáp ứng yêu cầu xử lý thời gian thực.

## Hướng phát triển

* Mở rộng bộ dữ liệu để tăng độ chính xác.
* Tích hợp cảnh báo bằng âm thanh khi phát hiện biển báo quan trọng.
* Triển khai trên Raspberry Pi hoặc robot tự hành.
* Kết hợp hệ thống hỗ trợ lái xe thông minh.

## Tác giả

**Họ và tên:** Đỗ Hữu Phước

**Mã sinh viên:** 2321060335

**Môn học:** Machine Learning

**Trường:** Đại học Mỏ - Địa chất
