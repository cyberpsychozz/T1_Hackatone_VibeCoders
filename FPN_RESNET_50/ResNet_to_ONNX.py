import torch
from pathlib import Path

# Импорт из вашего репозитория
import sys
from fpn.factory import make_fpn_resnet

# Параметры
checkpoint_path = "best.pt"  # путь к вашему чекпоинту
onnx_path = "fpn_resnet50_model.onnx"  # путь для сохранения ONNX-модели
device = "cpu"  # или "cuda", если доступен GPU
img_size = 384

# Создаём модель
model = make_fpn_resnet(
    name='resnet50',
    fpn_type='fpn',
    pretrained=False,
    num_classes=1,  # количество классов
    fpn_channels=256,
    in_channels=3,
    out_size=(img_size, img_size)
)

# Загружаем веса из чекпоинта
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.to(device)
model.eval()

# Создаём фиктивный вход для экспорта
dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

# Экспортируем модель в ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    },
    opset_version=13
)

print(f"Модель успешно экспортирована в {onnx_path}")
