import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

st.set_page_config(layout="wide", page_title="Замена фона — локальный макет")

# -----------------------
# Настройки боковой панели
# -----------------------
st.sidebar.header("Данные пользователя")
full_name = st.sidebar.text_input("Полное имя", "Иванов Сергей Петрович")
position = st.sidebar.text_input("Должность", "Ведущий инженер по компьютерному зрению")
company = st.sidebar.text_input("Компания", "ООО «Рога и Копыта»")
email = st.sidebar.text_input("Email", "sergey.ivanov@11dp.ru")
telegram = st.sidebar.text_input("Telegram", "@sergey_vision")

st.sidebar.header("Шрифт и стиль")
font_size = st.sidebar.slider("Размер шрифта", 10, 48, 18)
font_weight = st.sidebar.selectbox("Толщина шрифта", [100, 200, 300, 400, 500, 600, 700, 800, 900], index=4)
font_color = st.sidebar.color_picker("Цвет шрифта", "#FFFFFF")

st.sidebar.header("Логотип")
logo_file = st.sidebar.file_uploader("Загрузить логотип", type=["png", "jpg", "jpeg"])
logo_size = st.sidebar.slider("Размер логотипа (px)", 50, 300, 110)
logo_opacity = st.sidebar.slider("Прозрачность логотипа (%)", 0, 100, 100)

st.sidebar.header("Фон")
background_file = st.sidebar.file_uploader("Загрузить фон", type=["png", "jpg", "jpeg"])

# -----------------------
# Параметры модели
# -----------------------
onnx_path = "FPN_RESNET_50/fpn_resnet50_model.onnx"
img_size = 384
device = "cuda"  # или "cpu"

sess = ort.InferenceSession(
    onnx_path,
    providers=['CUDAExecutionProvider' if device=="cuda" else 'CPUExecutionProvider']
)

# -----------------------
# Камера
# -----------------------
st.header("Видеопоток с заменой фона")
run = st.checkbox("Запустить камеру")

frame_placeholder = st.empty()
fps_text = st.empty()

cap = cv2.VideoCapture(0)
fps = 0
frame_count = 0
start_time = time.time()
kernel_size = 5
min_area = 500

def apply_background(frame, mask, background):
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask)
    bg = cv2.bitwise_and(background, background, mask=mask_inv)
    return cv2.add(fg, bg)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Не удалось получить кадр с камеры")
        break

    # Обновляем FPS
    frame_count += 1
    if frame_count % 5 == 0:
        fps = 5 / (time.time() - start_time)
        start_time = time.time()
    fps_text.text(f"FPS: {fps:.2f}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_tensor = np.expand_dims(img_resized.transpose(2,0,1).astype(np.float32)/255.0, axis=0)

    outputs = sess.run(output_names=["output"], input_feed={"input": img_tensor})
    logits = outputs[0][0,0]
    mask = (1/(1+np.exp(-logits)) > 0.5).astype(np.uint8)
    mask_full = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Морфология
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_CLOSE, kernel)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(mask_full)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    mask_full = filtered_mask

    # Фон
    if background_file:
        background = np.array(Image.open(background_file).convert("RGB"))
        background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    else:
        background = np.zeros_like(frame)

    output = apply_background(frame, mask_full, background)

    # Добавляем текст
    cv2.putText(output, full_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size/20, tuple(int(font_color.lstrip('#')[i:i+2],16) for i in (0,2,4)), 2)
    cv2.putText(output, position, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size/25, tuple(int(font_color.lstrip('#')[i:i+2],16) for i in (0,2,4)), 2)

    # Логотип
    if logo_file:
        logo = np.array(Image.open(logo_file).convert("RGBA"))
        logo = cv2.resize(logo, (logo_size, logo_size))
        alpha = logo[:,:,3] / 255 * (logo_opacity/100)
        for c in range(3):
            output[10:10+logo_size, 10:10+logo_size, c] = (alpha*logo[:,:,c] + (1-alpha)*output[10:10+logo_size, 10:10+logo_size, c]).astype(np.uint8)

    frame_placeholder.image(output, channels="RGB")

cap.release()
