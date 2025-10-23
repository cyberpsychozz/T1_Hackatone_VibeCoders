import cv2
import numpy as np
from ultralytics import YOLO

# Загружаем модель сегментации (предобучена на COCO)
model = YOLO("yolov8x-seg.pt")

# Открываем веб-камеру
cap = cv2.VideoCapture(0)  # можно заменить на "video.mp4"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Запускаем сегментацию
    results = model(frame, verbose=False)[0]
    output = frame.copy()

    # Если есть маски — рисуем только людей (class_id == 0)
    if results.masks is not None:
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(results.masks.data):
            if class_ids[i] == 0:  # класс "person" в COCO
                mask = mask.cpu().numpy().astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    # Показываем результат
    cv2.imshow("People Segmentation", output)

    # Esc — выход
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
