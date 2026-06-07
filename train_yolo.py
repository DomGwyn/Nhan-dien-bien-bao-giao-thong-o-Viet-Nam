from ultralytics import YOLO

def main():
    # 1. Khởi tạo mô hình YOLOv8 Nano (nhẹ nhất, nhanh nhất)
    # File yolov8n.pt sẽ tự động được tải về nếu chưa có
    model = YOLO('yolov8n.pt') 

    # 2. Bắt đầu quá trình huấn luyện
    # Lưu ý: Nếu đường dẫn bị lỗi, bạn có thể phải nhập đường dẫn tuyệt đối tới file data.yaml
    results = model.train(
        data='data.yaml',         # Đường dẫn tới file cấu hình data
        epochs=50,                # Số vòng lặp huấn luyện. Bạn có thể tăng lên 50-100 nếu muốn độ chính xác cao hơn
        imgsz=640,                # Kích thước ảnh chuẩn của YOLO
        batch=16,                 # Số ảnh đưa vào mô hình trong 1 lần (nếu máy báo lỗi hết RAM, hãy giảm xuống 8)
        name='vietnam_traffic_model', # Tên thư mục lưu kết quả (sẽ nằm trong thư mục runs/detect/)
        device=0              # Chỉ định chạy bằng CPU (do máy bạn chưa cấu hình GPU)
    )

    print("Huấn luyện hoàn tất! Mô hình tốt nhất đã được lưu trong: runs/detect/vietnam_traffic_model/weights/best.pt")

if __name__ == '__main__':
    main()
