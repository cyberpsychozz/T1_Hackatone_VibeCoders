import cv2
import numpy as np
import onnxruntime as ort
import time

# Параметры
onnx_path = "fpn_resnet50_model.onnx"  # Путь к вашей ONNX-модели
img_size = 384  # Размер изображения для модели
device = "cuda"  # или "cuda", если доступен GPU
background_path = "backgrounds/back.png"  # Путь к фону

# Загружаем ONNX-модель
sess = ort.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider']
)

# Загружаем фон
background = cv2.imread(background_path)
if background is None:
    raise FileNotFoundError(f"Фон не найден: {background_path}")

# Инициализация счётчика FPS
fps = 0
fps_update_interval = 5  # Обновляем FPS каждые 5 кадров
frame_count = 0
start_time = time.time()

# Параметры для постобработки маски
kernel_size = 5
min_area = 500

# Камера
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

# Получаем размер кадра для ресайза фона
ret, frame = cap.read()
if ret:
    background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Сброс на начало

def apply_background(frame, mask, background):
    """Функция для замены фона."""
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask)
    bg = cv2.bitwise_and(background, background, mask=mask_inv)
    return cv2.add(fg, bg)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Обновляем FPS каждые `fps_update_interval` кадров
    frame_count += 1
    if frame_count % fps_update_interval == 0:
        fps = fps_update_interval / (time.time() - start_time)
        start_time = time.time()

    # Конвертируем BGR в RGB и ресайзим
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))

    # Преобразуем в тензор и нормализуем
    img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Добавляем размер батча [1, C, H, W]

    # Инференс
    outputs = sess.run(
        output_names=["output"],
        input_feed={"input": img_tensor}
    )

    # Получаем маску
    logits = outputs[0][0, 0]
    mask = 1 / (1 + np.exp(-logits))  # Применяем сигмоиду
    mask = (mask > 0.5).astype(np.uint8)

    # Масштабируем маску до размера кадра камеры
    mask_full = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Морфология: closing для заполнения дыр, opening для удаления шума
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)

    # Фильтрация контуров: находим все, удаляем мелкие
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask_full)  # Новая маска только с крупными контурами
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:  # Объединяем только крупные
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Используем отфильтрованную маску
    mask_full = filtered_mask

    # Заменяем фон
    output = apply_background(frame, mask_full, background)

    # Отображаем FPS на кадре
    cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # output = cv2.resize(output, (1240, 1080))

    # Отображаем результат
    cv2.imshow("Segmentation with Background", output)

    if cv2.waitKey(1) & 0xFF == 27:  # Нажмите Esc для выхода
        break

cap.release()
cv2.destroyAllWindows()
