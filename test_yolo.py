import cv2
from ultralytics import YOLO
import easyocr
import re

# 1. NẠP MÔ HÌNH YOLOV8
model_path = 'runs/detect/vietnam_traffic_model3/weights/best.pt' 

try:
    print(f"Đang nạp mô hình từ: {model_path} ...")
    model = YOLO(model_path)
    print("Nạp mô hình thành công!")
except Exception as e:
    print(f"Lỗi: Không tìm thấy file mô hình tại {model_path}.")
    exit()

# 2. KHỞI TẠO CAMERA VÀ OCR READER
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
font = cv2.FONT_HERSHEY_SIMPLEX

print("Đang khoi tao module doc so (OCR)...")
reader = easyocr.Reader(['en'], gpu=True)
print("Khoi tao OCR thanh cong!")

# BỘ TỪ ĐIỂN
sign_dict = {
    'P-102': 'Cam di nguoc chieu',
    'P-103a': 'Cam o to',
    'P-103b': 'Cam o to re phai',
    'P-103c': 'Cam o to re trai',
    'P-104': 'Cam mo to',
    'P-106a': 'Cam xe tai',
    'P-106b': 'Cam xe tai tren 2.5T',
    'P-107a': 'Cam o to khach va xe tai',
    'P-112': 'Cam nguoi di bo',
    'P-115': 'Han che trong luong xe',
    'P-117': 'Han che chieu cao',
    'P-123a': 'Cam re trai',
    'P-123b': 'Cam re phai',
    'P-124a': 'Cam quay dau',
    'P-124b': 'Cam o to quay dau',
    'P-124c': 'Cam quay dau va re trai',
    'P-127': 'TOC DO TOI DA',       # Highlight in hoa cho nổi bật
    'P-128': 'Cam su dung coi',
    'P-130': 'Cam dung va do xe',
    'P-131a': 'Cam do xe',
    'P-137': 'Cam re trai va phai',
    'P-245a': 'Di cham (Cam)',

    # 2. NHÓM BIỂN HIỆU LỆNH (R - Requirement)
    'R-301c': 'Chi duoc re trai',
    'R-301d': 'Chi duoc re phai',
    'R-301e': 'Huong di thang va re phai',
    'R-302a': 'Huong di phai vong sang phai',
    'R-302b': 'Huong di phai vong sang trai',
    'R-303': 'Noi giao nhau chay theo vong xuyen',
    'R-407a': 'Duong mot chieu',
    'R-409': 'Cho quay xe',
    'R-425': 'Benh vien',
    'R-434': 'Ben xe buyt',

    # 3. NHÓM BIỂN NGUY HIỂM (W - Warning)
    'W-201a': 'Ngoat nguy hiem ben trai',
    'W-201b': 'Ngoat nguy hiem ben phai',
    'W-202a': 'Nhieu cho ngoat nguy hiem (trai)',
    'W-202b': 'Nhieu cho ngoat nguy hiem (phai)',
    'W-203b': 'Duong hep ben trai',
    'W-203c': 'Duong hep ben phai',
    'W-205a': 'Giao nhau cung cap (Nga tu)',
    'W-205b': 'Giao nhau cung cap (Nga ba)',
    'W-205d': 'Giao nhau cung cap (Chu Y)',
    'W-207a': 'Giao duong khong uu tien',
    'W-207b': 'Giao duong khong uu tien (phai)',
    'W-207c': 'Giao duong khong uu tien (trai)',
    'W-208': 'Giao duong uu tien (Giam toc do)',
    'W-209': 'Giao nhau co tin hieu den',
    'W-210': 'Giao duong sat co rao chan',
    'W-219': 'Doc xuong nguy hiem',
    'W-224': 'Duong nguoi di bo cat ngang',
    'W-225': 'Tre em',
    'W-227': 'Cong truong',
    'W-233': 'Nguy hiem khac',
    'W-235': 'Duong doi',
    'W-245a': 'Di cham',

    # 4. NHÓM BIỂN CHỈ DẪN & PHỤ (DP, S)
    'DP-135': 'HET TAT CA LENH CAM', # Highlight
    'S-509a': 'Chieu cao an toan'
}

def preprocess_for_glare(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

last_detected_speed = ""
frames_since_last_read = 100 

# 3. VÒNG LẶP CHÍNH
while True:
    success, frame = cap.read()
    if not success:
        break

    # TẠO BẢN SAO SẠCH ĐỂ DÀNH RIÊNG CHO OCR ĐỌC (Không bị dính nét vẽ)
    clean_frame = frame.copy()

    processed_frame = preprocess_for_glare(frame)
    results = model.predict(processed_frame, conf=0.65, imgsz=640, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = int(max(x2 - x1, y2 - y1) / 2) + 10
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 3)
            
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            raw_name = model.names[cls_id] 
            
            # --- XỬ LÝ RIÊNG CHO BIỂN TỐC ĐỘ (P-127) ---
            if raw_name == 'P-127':
                crop_y1, crop_y2 = max(0, y1), min(frame.shape[0], y2)
                crop_x1, crop_x2 = max(0, x1), min(frame.shape[1], x2)
                
                if (crop_x2 - crop_x1) > 40 and (crop_y2 - crop_y1) > 40:
                    # ĐIỂM MẤU CHỐT: Cắt ảnh từ clean_frame thay vì frame
                    cropped_sign = clean_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    cropped_sign = cv2.resize(cropped_sign, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    gray_sign = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2GRAY)
                    
                    ocr_result = reader.readtext(gray_sign)
                    
                    found_number = False
                    for (bbox, text, prob) in ocr_result:
                        numbers = re.findall(r'\b\d{2,3}\b', text) 
                        if numbers:
                            last_detected_speed = numbers[0]
                            frames_since_last_read = 0 
                            found_number = True
                            break
                    
                    if not found_number:
                        frames_since_last_read += 1
                
                if last_detected_speed and frames_since_last_read < 30:
                    display_name = f"TOC DO: {last_detected_speed} km/h"
                else:
                    display_name = "TOC DO TOI DA"
            
            # --- XỬ LÝ CÁC BIỂN KHÁC ---
            else:
                display_name = sign_dict.get(raw_name, raw_name)
            
            label = f"{display_name} ({conf:.2f})"
            
            # VẼ NỀN ĐEN VÀ CHỮ VÀNG Ở PHÍA DƯỚI BIỂN BÁO (Tránh che khuất nhau)
            (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
            
            # Tọa độ mới: Nằm căn giữa ở phía dưới vòng tròn
            text_x = center_x - int(text_w / 2)
            text_y = center_y + radius + 25 
            
            # Vẽ HCN nền đen
            cv2.rectangle(frame, (text_x - 5, text_y - text_h - 5), 
                                 (text_x + text_w + 5, text_y + 5), (0,0,0), -1)
            # In chữ
            cv2.putText(frame, label, (text_x, text_y), font, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, "Bam 'q' de thoat", (10, 30), font, 0.7, (0, 0, 255), 2)
    cv2.imshow("Nhan dien bien bao Viet Nam - YOLO + OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()