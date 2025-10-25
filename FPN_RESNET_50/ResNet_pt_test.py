import cv2
import torch
import numpy as np
import time
import pandas as pd


from fpn.factory import make_fpn_resnet

# -----------------------
# Параметры
# -----------------------
checkpoint_path = "best_1class.pt"
device = "cpu"  # или "cuda" если доступна
img_size = 384
num_classes = 1  # Обучение было на 1

# -----------------------
# Загружаем модель
# -----------------------
model = make_fpn_resnet(
    name='resnet50',
    fpn_type='fpn',
    pretrained=False,
    num_classes=num_classes,
    fpn_channels=256,
    in_channels=3,
    out_size=(img_size, img_size)
)

# Загрузка чекпоинта с обработкой разных структур
checkpoint = torch.load(checkpoint_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
model.to(device)
model.eval()

# -----------------------
# Камера
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

# Переменные для FPS
prev_time = 0
curr_time = 0
fps = 0

print("Запуск сегментации с веб-камеры. Нажмите ESC для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Расчет FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # OpenCV BGR -> RGB и resize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))

    # Преобразуем в тензор и нормализуем
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]

    # Предсказание
    with torch.no_grad():
        logits = model(img_tensor)  # [1, num_classes, H, W]
        mask = torch.argmax(logits, dim=1)[0].cpu().numpy()  # Выбираем класс с максимальной вероятностью

    # Масштабируем маску до размера кадра камеры
    mask_full = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Цветовая карта для 20 классов (простая генерация цветов)
    mask_colored = np.zeros_like(frame)
    for cls in range(num_classes):
        if cls == 0:
            mask_colored[mask_full == cls] = [0, 0, 0]  # Фон (черный)
        else:
            # Генерация уникальных цветов для классов 1-19
            r = (cls * 13) % 256
            g = (cls * 7) % 256
            b = (cls * 23) % 256
            mask_colored[mask_full == cls] = [r, g, b]

    # Наложение на оригинальный кадр
    overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    # Отображение FPS на кадре
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(overlay, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Дополнительная информация
    cv2.putText(overlay, "Press ESC to exit", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc для выхода
        break

cap.release()
cv2.destroyAllWindows()