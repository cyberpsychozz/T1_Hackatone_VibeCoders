import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import torch

# -----------------------
# Настройка страницы
# -----------------------
st.set_page_config(layout="wide", page_title="Замена фона — локальный макет")

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
TEXT_FRAME_WIDTH_RATIO = 1/6  # текстовая рамка = 1/6 ширины видео

# -----------------------
# Настройки пользователя
# -----------------------
st.sidebar.header("Данные пользователя")
full_name = st.sidebar.text_input("Полное имя", "Иванов Сергей Петрович")
position = st.sidebar.text_input("Должность", "Ведущий инженер по компьютерному зрению")
company = st.sidebar.text_input("Компания", "ООО «Рога и Копыта»")
department = st.sidebar.text_input("Департамент", "Отдел ИИ")
location = st.sidebar.text_input("Локация", "Новосибирск")
email = st.sidebar.text_input("Email", "sergey.ivanov@11dp.ru")
telegram = st.sidebar.text_input("Telegram", "@sergey_vision")
qr_code_file = st.sidebar.file_uploader("QR-код (для high privacy)", type=["png", "jpg", "jpeg"])

st.sidebar.header("Шрифт и стиль")
font_size = st.sidebar.slider("Размер шрифта (высота текста)", 10, 72, 18)
font_color = st.sidebar.color_picker("Цвет шрифта", "#FFFFFF")

st.sidebar.header("Логотип")
logo_file = st.sidebar.file_uploader("Загрузить логотип", type=["png", "jpg", "jpeg"])
logo_size = st.sidebar.slider("Размер логотипа (px)", 50, 300, 110)
logo_opacity = st.sidebar.slider("Прозрачность логотипа (%)", 0, 100, 100)

st.sidebar.header("Фон")
background_file = st.sidebar.file_uploader("Загрузить фон", type=["png", "jpg", "jpeg"])

st.sidebar.header("Позиция рамки текста")
frame_x = st.sidebar.slider("Смещение X рамки", 0, VIDEO_WIDTH-1, 30)
frame_y = st.sidebar.slider("Смещение Y рамки", 0, VIDEO_HEIGHT-1, 30)

st.sidebar.header("Уровень приватности")
privacy_level = st.sidebar.selectbox("Privacy level", ["low", "medium", "high"], index=0)

# -----------------------
# Модель и устройство
# -----------------------
onnx_path = "FPN_RESNET_50/fpn_resnet50_model.onnx"
img_size = 384

device = "cuda" if torch.cuda.is_available() else "cpu"
sess = ort.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider' if device=="cuda" else 'CPUExecutionProvider']
)
st.sidebar.text(f"Используемое устройство: {device.upper()}")

# -----------------------
# Камера и поток
# -----------------------
st.header("Видеопоток с заменой фона")
run = st.checkbox("Запустить камеру")
frame_placeholder = st.empty()
fps_text = st.empty()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

fps = 0
frame_count = 0
start_time = time.time()
kernel = np.ones((5,5), np.uint8)
min_area = 500
font = cv2.FONT_HERSHEY_COMPLEX
font_rgb = tuple(int(font_color.lstrip('#')[i:i+2],16) for i in (0,2,4))

# --- Подготовка фона заранее в RGB ---
background = None
if background_file:
    background = np.array(Image.open(background_file).convert("RGB"))

TEXT_FRAME_WIDTH = int(VIDEO_WIDTH * TEXT_FRAME_WIDTH_RATIO)

# --- Функция для текста с ограничением ширины рамки ---
def draw_text_line(img, text, x, y, width, font, font_height, color):
    text_size = cv2.getTextSize(text, font, font_height, 2)[0]
    scale = font_height / text_size[1]
    max_chars = max(int(len(text) * (width / text_size[0])), 1)  # защита от деления на 0
    lines = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, color, 2)
        y += int(text_size[1]*scale*1.2)
    return y

# --- Основной цикл ---
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Не удалось получить кадр с камеры")
        break

    frame_count += 1
    if frame_count % 5 == 0:
        fps = 5 / (time.time() - start_time)
        start_time = time.time()
    fps_text.text(f"FPS: {fps:.2f}")


    # --- RGB для модели ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor = np.expand_dims(img_resized.transpose(2,0,1).astype(np.float32)/255.0, axis=0)

    # --- Предсказание маски ---
    outputs = sess.run(output_names=["output"], input_feed={"input": img_tensor})
    logits = outputs[0][0,0]
    mask = (1/(1+np.exp(-logits)) > 0.5).astype(np.uint8)*255
    mask_full = cv2.resize(mask, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_NEAREST)

    # --- Морфология и фильтрация ---
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask_full)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask_full = filtered_mask

    # --- Подготовка фона ---
    if background is not None:
        bg_resized = cv2.resize(background, (VIDEO_WIDTH, VIDEO_HEIGHT))
    else:
        bg_resized = np.zeros_like(frame)

    # --- Замена фона ---
    fg = cv2.bitwise_and(frame, frame, mask=mask_full)
    mask_inv = cv2.bitwise_not(mask_full)
    bg = cv2.bitwise_and(bg_resized, bg_resized, mask=mask_inv)
    output = cv2.add(fg, bg)

    # --- Рамка текста ---
    frame_offset_x = frame_x
    frame_offset_y = frame_y
    y_cursor = frame_offset_y

    # --- Отрисовка текста по privacy_level ---
    y_cursor = draw_text_line(output, full_name, frame_offset_x, y_cursor, TEXT_FRAME_WIDTH, font, font_size, font_rgb)
    y_cursor = draw_text_line(output, position, frame_offset_x, y_cursor, TEXT_FRAME_WIDTH, font, font_size*0.9, font_rgb)

    if privacy_level in ["medium", "high"]:
        y_cursor = draw_text_line(output, company, frame_offset_x, y_cursor, TEXT_FRAME_WIDTH, font, font_size*0.8, font_rgb)
        y_cursor = draw_text_line(output, department, frame_offset_x, y_cursor, TEXT_FRAME_WIDTH, font, font_size*0.8, font_rgb)
        y_cursor = draw_text_line(output, location, frame_offset_x, y_cursor, TEXT_FRAME_WIDTH, font, font_size*0.8, font_rgb)

    if privacy_level == "high":
        y_cursor = draw_text_line(output, email, frame_offset_x, y_cursor, TEXT_FRAME_WIDTH, font, font_size*0.7, font_rgb)
        y_cursor = draw_text_line(output, telegram, frame_offset_x, y_cursor, TEXT_FRAME_WIDTH, font, font_size*0.7, font_rgb)
        if qr_code_file:
            qr = np.array(Image.open(qr_code_file).convert("RGBA"))
            qr_size = 100
            qr = cv2.resize(qr, (qr_size, qr_size))
            alpha = qr[:,:,3]/255
            for c in range(3):
                output[y_cursor:y_cursor+qr_size, frame_offset_x:frame_offset_x+qr_size, c] = (
                    alpha*qr[:,:,c] + (1-alpha)*output[y_cursor:y_cursor+qr_size, frame_offset_x:frame_offset_x+qr_size, c]
                ).astype(np.uint8)

    # --- Логотип ---
    if logo_file:
        logo = np.array(Image.open(logo_file).convert("RGBA"))
        logo = cv2.resize(logo, (logo_size, logo_size))
        alpha = logo[:,:,3]/255 * (logo_opacity/100)
        for c in range(3):
            output[frame_offset_y:frame_offset_y+logo_size, frame_offset_x:frame_offset_x+logo_size, c] = (
                alpha*logo[:,:,c] + (1-alpha)*output[frame_offset_y:frame_offset_y+logo_size, frame_offset_x:frame_offset_x+logo_size, c]
            ).astype(np.uint8)

    # --- Вывод в Streamlit ---
    frame_placeholder.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

cap.release()
