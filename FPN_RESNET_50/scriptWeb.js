// -------------------- Root structure --------------------
const app = document.createElement('div');
app.className = 'app';
document.body.appendChild(app);
const left = document.createElement('div');
left.className = 'panel left';
app.appendChild(left);
const right = document.createElement('div');
right.className = 'panel right';
app.appendChild(right);

// header
const header = document.createElement('div');
header.className = 'header';
header.innerHTML = `
  <div class="logo-badge">+VC</div>
  <div style="flex:1">
    <h3 class="h1">Замена фона — локальный макет</h3>
    <div class="small-muted">Интерактивно • Без сервера</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center">
    <button id="themeToggle" class="btn">Светлая тема</button>
  </div>
`;
left.appendChild(header);

// previews: larger (each in framed box with title)
const previewRow = document.createElement('div');
previewRow.className = 'preview-row';
left.appendChild(previewRow);

function createFramedWindow(title, id) {
    const frame = document.createElement('div');
    frame.className = 'window-frame';
    const t = document.createElement('div');
    t.className = 'title';
    t.textContent = title;
    frame.appendChild(t);
    const win = document.createElement('div');
    win.className = 'window';
    win.id = id;
    const media = id === 'winOriginal' 
        ? document.createElement('video') 
        : document.createElement('canvas');
    media.id = id === 'winOriginal' ? 'origVideo' : 'resultCanvas';
    media.className = 'media';
    const overlay = document.createElement('div');
    overlay.className = 'preview-overlay';
    overlay.id = id === 'winOriginal' ? 'overlayOrig' : 'overlayResult';
    win.appendChild(media);
    win.appendChild(overlay);
    frame.appendChild(win);
    return frame;
}

const frameOrig = createFramedWindow('Original preview', 'winOriginal');
const frameResult = createFramedWindow('Result preview', 'winResult');
previewRow.appendChild(frameOrig);
previewRow.appendChild(frameResult);

// controls under previews
const mainControls = document.createElement('div');
mainControls.className = 'controls';
left.appendChild(mainControls);
const controlsPanel = document.createElement('div');
controlsPanel.className = 'controls-panel';
mainControls.appendChild(controlsPanel);

// file uploads
const rowFiles = document.createElement('div');
rowFiles.className = 'row';
controlsPanel.appendChild(rowFiles);

function makeUpload(labelText, id, accept, multiple) {
    const upload = document.createElement('label');
    upload.className = 'upload-file';
    upload.innerHTML = `${labelText} <input id="${id}" type="file" accept="${accept}" style="display:none" ${multiple? 'multiple':''}>`;
    return upload;
}
rowFiles.appendChild(makeUpload('Загрузить фон', 'fileBg', 'image/*'));
rowFiles.appendChild(makeUpload('Загрузить шаблоны (несколько)', 'fileTemplates', 'image/*', true));
rowFiles.appendChild(makeUpload('Загрузить логотип', 'fileLogo', 'image/*'));

// action buttons
const actions = document.createElement('div');
actions.className = 'controls';

function createBtn(text, variant) {
    const b = document.createElement('button');
    b.className = 'btn' + (variant ? ' ' + variant : '');
    b.textContent = text;
    return b;
}
const startBtn = createBtn('START', 'primary');
const stopBtn = createBtn('STOP');
const exportBtn = createBtn('Экспорт PNG');
actions.appendChild(startBtn);
actions.appendChild(stopBtn);
actions.appendChild(exportBtn);
controlsPanel.appendChild(actions);

// status line
const status = document.createElement('div');
status.className = 'small-muted';
status.textContent = 'Статус: готов';
controlsPanel.appendChild(status);

// overlay elements (text + logo) for both previews (we will append into overlays)
const overlayOrig = document.getElementById('overlayOrig');
const overlayResult = document.getElementById('overlayResult');

// create text draggable and logo draggable
const textElOrig = document.createElement('div');
textElOrig.className = 'draggable';
textElOrig.id = 'draggableText';
textElOrig.textContent = 'Иванов Сергей\nВедущий инженер';
textElOrig.style.left = '18px';
textElOrig.style.top = '18px';
overlayOrig.appendChild(textElOrig);
const textElRes = textElOrig.cloneNode(true);
overlayResult.appendChild(textElRes);
const logoElOrig = document.createElement('img');
logoElOrig.className = 'draggable logo';
logoElOrig.id = 'draggableLogo';
logoElOrig.alt = 'LOGO';
logoElOrig.style.width = '110px';
logoElOrig.style.right = '18px';
logoElOrig.style.top = '18px';
overlayOrig.appendChild(logoElOrig);
const logoElRes = logoElOrig.cloneNode(true);
overlayResult.appendChild(logoElRes);

// logo placeholder box when no logo uploaded
const logoPlaceholderOrig = document.createElement('div');
logoPlaceholderOrig.className = 'logo-placeholder';
logoPlaceholderOrig.textContent = 'LOGO';
overlayOrig.appendChild(logoPlaceholderOrig);
const logoPlaceholderRes = logoPlaceholderOrig.cloneNode(true);
overlayResult.appendChild(logoPlaceholderRes);

// -------------------- Right panel: settings --------------------
right.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center"><div><h4 style="margin:0">🎨 Данные пользователя</h4><div class="small-muted">Редактируй — смотри превью</div></div><div class="badge">Локально</div></div>`;
const settings = document.createElement('div');
settings.style.marginTop = '12px';
settings.className = 'controls-panel';
right.appendChild(settings);

const fields = [
    { label: 'Полное имя', id: 'full_name', value: 'Иванов Сергей Петрович' },
    { label: 'Должность', id: 'position', value: 'Ведущий инженер по компьютерному зрению' },
    { label: 'Компания', id: 'company', value: 'ООО «Рога и Копыта»' },
    { label: 'Email', id: 'email', value: 'sergey.ivanov@t1dp.ru' },
    { label: 'Telegram', id: 'telegram', value: '@sergey_vision' }
];
const inputs = {};
fields.forEach(f => {
    const fld = document.createElement('div');
    fld.className = 'field';
    fld.innerHTML = `<div class="label">${f.label}</div>`;
    const inp = document.createElement('input');
    inp.className = 'input';
    inp.value = f.value;
    inp.id = f.id;
    fld.appendChild(inp);
    settings.appendChild(fld);
    inputs[f.id] = inp;
});

// typography + font family (improved visible list)
const typLabel = document.createElement('div');
typLabel.className = 'label';
typLabel.textContent = 'Шрифт и стиль';
settings.appendChild(typLabel);
const typRow = document.createElement('div');
typRow.className = 'row';
typRow.style.gap = '6px';
settings.appendChild(typRow);

const fontColor = document.createElement('input');
fontColor.type = 'color';
fontColor.className = 'input';
fontColor.value = '#ffffff';
fontColor.style.width = '56px';
typRow.appendChild(fontColor);

const fontSizeWrap = document.createElement('div');
fontSizeWrap.style.flex = '1';
fontSizeWrap.className = 'field';
fontSizeWrap.innerHTML = `<div class="label">Размер шрифта: <span id="fontSizeLabel">18px</span></div>`;
const fontSize = document.createElement('input');
fontSize.type = 'range';
fontSize.min = 12;
fontSize.max = 48;
fontSize.value = 18;
fontSize.className = 'input';
fontSize.style.width = '100%';
fontSizeWrap.appendChild(fontSize);
typRow.appendChild(fontSizeWrap);

// custom visible font list (so user sees all fonts at once)
const fontSelectWrap = document.createElement('div');
fontSelectWrap.className = 'field';
fontSelectWrap.innerHTML = `<div class="label">Шрифт</div>`;

const fontSelect = document.createElement('select');
fontSelect.className = 'input';

const fontOptions = [
    'Inter, system-ui, -apple-system',
    'Segoe UI, Tahoma, sans-serif', 
    'Roboto, Arial, sans-serif',
    'Montserrat, Arial, sans-serif',
    'Georgia, serif',
    'Courier New, monospace'
];

let fontSelectValue = fontOptions[0];

// Создаем options для каждого шрифта
fontOptions.forEach(f => {
    const option = document.createElement('option');
    option.value = f;
    option.textContent = f.split(',')[0]; // Показываем только первое название
    option.style.fontFamily = f;
    fontSelect.appendChild(option);
});

// Устанавливаем первое значение по умолчанию
fontSelect.value = fontSelectValue;

// Обработчик изменения выбора
fontSelect.addEventListener('change', () => {
    fontSelectValue = fontSelect.value;
    updateStateFromUI();
    updatePreviewFromInputs();
});

fontSelectWrap.appendChild(fontSelect);
typRow.appendChild(fontSelectWrap);

// font weight
const fontWeightWrap = document.createElement('div');
fontWeightWrap.className = 'field';
fontWeightWrap.style.width = '80px';
fontWeightWrap.innerHTML = `<div class="label">Толщина шрифта</div>`;
const fontWeight = document.createElement('select');
fontWeight.className = 'input';
['400', '600', '700', '800'].forEach(w => {
    const o = document.createElement('option');
    o.value = w;
    o.textContent = w;
    fontWeight.appendChild(o);
});
fontWeightWrap.appendChild(fontWeight);
typRow.appendChild(fontWeightWrap);

// position presets
const posLabel = document.createElement('div');
posLabel.className = 'label';
posLabel.textContent = 'Позиция рамки';
settings.appendChild(posLabel);
const posRow = document.createElement('div');
posRow.className = 'row';
['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'].forEach(p => {
    const b = createBtn(p.replace('-', ' '));
    b.dataset.pos = p;
    b.addEventListener('click', () => {
        setPresetPosition(p);
        updatePreviewFromInputs();
    });
    posRow.appendChild(b);
});
settings.appendChild(posRow);

// logo controls
const logoLabel = document.createElement('div');
logoLabel.className = 'label';
logoLabel.textContent = 'Логотип';
settings.appendChild(logoLabel);
const logoSizeRow = document.createElement('div');
logoSizeRow.className = 'row';
settings.appendChild(logoSizeRow);

const logoOpacityWrap = document.createElement('div');
logoOpacityWrap.className = 'field';
logoOpacityWrap.style.flex = '1';
logoOpacityWrap.innerHTML = `<div class="label">Прозрачность логотипа: <span id="logoOpacityLabel">100%</span></div>`;
const logoOpacity = document.createElement('input');
logoOpacity.type = 'range';
logoOpacity.min = 0;
logoOpacity.max = 100;
logoOpacity.value = 100;
logoOpacity.className = 'input';
logoOpacity.style.width = '100%';
logoOpacityWrap.appendChild(logoOpacity);
logoSizeRow.appendChild(logoOpacityWrap);

const logoSizeWrap = document.createElement('div');
logoSizeWrap.className = 'field';
logoSizeWrap.style.width = '120px';
logoSizeWrap.innerHTML = `<div class="label">Размер логотипа: <span id="logoSizeLabel">110px</span></div>`;
const logoSize = document.createElement('input');
logoSize.type = 'range';
logoSize.min = 40;
logoSize.max = 220;
logoSize.value = 110;
logoSize.className = 'input';
logoSize.style.width = '100%';
logoSizeWrap.appendChild(logoSize);
logoSizeRow.appendChild(logoSizeWrap);

// privacy
const privLabel = document.createElement('div');
privLabel.className = 'label';
privLabel.textContent = 'Уровень приватности';
settings.appendChild(privLabel);
const privSelect = document.createElement('select');
privSelect.className = 'input';
[
    ['low', 'Низкий (только имя)'],
    ['medium', 'Средний (имя+должность)'],
    ['high', 'Высокий (вся информация)']
].forEach(p => {
    const o = document.createElement('option');
    o.value = p[0];
    o.textContent = p[1];
    privSelect.appendChild(o);
});
settings.appendChild(privSelect);

// templates gallery UI
const tplLabel = document.createElement('div');
tplLabel.className = 'label';
tplLabel.textContent = 'Шаблонные фоны (галерея)';
settings.appendChild(tplLabel);

const templateGallery = document.createElement('div');
templateGallery.className = 'template-gallery';
settings.appendChild(templateGallery);

const galleryThumbs = document.createElement('div');
galleryThumbs.className = 'gallery-thumbs';
templateGallery.appendChild(galleryThumbs);

// -------------------- State --------------------
const state = {
    textPos: { x: 18, y: 18 },
    logoPos: { left: null, right: 18, top: 18 },
    fontFamily: fontOptions[0],
    fontSize: 18,
    fontColor: '#ffffff',
    fontWeight: '400',
    logoOpacity: 1,
    logoSize: 110,
    privLevel: 'medium',
    templates: [],
    currentBg: null,
    logoSrc: null
};

// -------------------- File handlers --------------------
const fileBg = document.getElementById('fileBg');
fileBg.addEventListener('change', e => {
    if (e.target.files[0]) {
        state.currentBg = URL.createObjectURL(e.target.files[0]);
        updateBackground();
    }
});

const fileTemplates = document.getElementById('fileTemplates');
fileTemplates.addEventListener('change', e => {
    const files = Array.from(e.target.files);
    state.templates = files.map(f => URL.createObjectURL(f));
    renderGallery();
    if (state.templates.length > 0) {
        state.currentBg = state.templates[0];
        updateBackground();
    }
});

const fileLogo = document.getElementById('fileLogo');
fileLogo.addEventListener('change', e => {
    if (e.target.files[0]) {
        state.logoSrc = URL.createObjectURL(e.target.files[0]);
        setLogo(state.logoSrc);
    }
});

// render gallery thumbs
function renderGallery() {
    galleryThumbs.innerHTML = '';
    state.templates.forEach((src, idx) => {
        const thumb = document.createElement('img');
        thumb.className = 'gallery-thumb';
        thumb.src = src;
        thumb.addEventListener('click', () => {
            state.currentBg = src;
            updateBackground();
            document.querySelectorAll('.gallery-thumb').forEach(t => t.classList.remove('active'));
            thumb.classList.add('active');
        });
        galleryThumbs.appendChild(thumb);
    });
    if (state.templates.length > 0) {
        galleryThumbs.firstChild.classList.add('active');
    }
}

// set logo
function setLogo(src) {
    [logoElOrig, logoElRes].forEach(l => l.src = src);
    [logoPlaceholderOrig, logoPlaceholderRes].forEach(p => p.style.display = src ? 'none' : 'block');
}

// update background (preload image)
let bgImage = new Image();
function updateBackground() {
    if (state.currentBg) {
        bgImage.src = state.currentBg;
        bgImage.onload = () => {
            updateCanvasSize();
        };
    }
}

// -------------------- Update preview from inputs --------------------
function updatePreviewFromInputs() {
    let text = '';
    switch (state.privLevel) {
        case 'low':
            text = inputs.full_name.value;
            break;
        case 'medium':
            text = `${inputs.full_name.value}\n${inputs.position.value}`;
            break;
        case 'high':
            text = `${inputs.full_name.value}\n${inputs.position.value}\n${inputs.company.value}\n${inputs.email.value}\n${inputs.telegram.value}`;
            break;
    }
    [textElOrig, textElRes].forEach(t => {
        t.textContent = text;
        t.style.left = state.textPos.x + 'px';
        t.style.top = state.textPos.y + 'px';
        t.style.fontFamily = state.fontFamily;
        t.style.fontSize = state.fontSize + 'px';
        t.style.fontWeight = state.fontWeight;
        t.style.color = state.fontColor;
    });
    [logoElOrig, logoElRes].forEach(l => {
        l.style.width = state.logoSize + 'px';
        l.style.opacity = state.logoOpacity;
        if (state.logoPos.left !== null) l.style.left = state.logoPos.left + 'px';
        if (state.logoPos.right !== null) l.style.right = state.logoPos.right + 'px';
        l.style.top = state.logoPos.top + 'px';
    });
}

// -------------------- Draggable --------------------
function makePointerDraggable(element, opts = {}) {
    let active = false, startX, startY, origLeft, origTop, id = null;
    element.addEventListener('pointerdown', e => {
        active = true;
        id = e.pointerId;
        try { element.setPointerCapture(id); } catch (_) {}
        startX = e.clientX;
        startY = e.clientY;
        origLeft = parseFloat(element.style.left || 0);
        origTop = parseFloat(element.style.top || 0);
        element.style.transition = 'none';
    });
    element.addEventListener('pointermove', e => {
        if (!active) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        let nx = origLeft + dx;
        let ny = origTop + dy;
        const parentRect = element.parentElement.getBoundingClientRect();
        const elRect = element.getBoundingClientRect();
        if (nx < 0) nx = 0;
        if (ny < 0) ny = 0;
        if (nx > parentRect.width - elRect.width) nx = parentRect.width - elRect.width;
        if (ny > parentRect.height - elRect.height) ny = parentRect.height - elRect.height;
        element.style.left = nx + 'px';
        element.style.top = ny + 'px';
        if (opts.onMove) opts.onMove({ x: Math.round(nx), y: Math.round(ny) });
    });
    element.addEventListener('pointerup', e => {
        if (id !== null) try { element.releasePointerCapture(id); } catch (_) {}
        active = false;
        element.style.transition = '';
    });
    element.addEventListener('pointercancel', () => { active = false; });
    element.addEventListener('dragstart', e => e.preventDefault());
}

// attach draggables to both overlays for text & logo
function attachDraggables() {
    [overlayOrig, overlayResult].forEach((ov, idx) => {
        const t = ov.querySelector('#draggableText');
        const l = ov.querySelector('#draggableLogo');
        if (t) {
            if (!t.style.left) t.style.left = (state.textPos.x) + 'px';
            if (!t.style.top) t.style.top = (state.textPos.y) + 'px';
            t.style.fontFamily = state.fontFamily;
            t.style.fontSize = fontSize.value + 'px';
            t.style.fontWeight = fontWeight.value;
            t.style.color = fontColor.value;
            makePointerDraggable(t, {
                onMove: pos => {
                    state.textPos = pos;
                    updatePreviewFromInputs();
                }
            });
        }
        if (l) {
            if (state.logoPos.left !== null) l.style.left = state.logoPos.left + 'px';
            else {
                const pRect = ov.getBoundingClientRect();
                const leftInit = Math.max(8, pRect.width - (state.logoPos.right || 18) - parseFloat(l.style.width || 110));
                l.style.left = leftInit + 'px';
            }
            l.style.top = (state.logoPos.top || 18) + 'px';
            makePointerDraggable(l, {
                onMove: pos => {
                    const parentW = ov.getBoundingClientRect().width;
                    state.logoPos = { left: pos.x, right: Math.round(parentW - pos.x - parseFloat(l.style.width || 110)), top: pos.y };
                    updatePreviewFromInputs(); 
                }
            });
        }
    });
}

// initialize draggables
setTimeout(() => {
    attachDraggables();
    updatePreviewFromInputs();
}, 150);

// -------------------- Position presets --------------------
function setPresetPosition(preset) {
    const ov = overlayOrig;
    const el = ov.querySelector('.draggable');
    if (!el) return;
    const pw = ov.getBoundingClientRect().width,
        ph = ov.getBoundingClientRect().height;
    let nx = 18,
        ny = 18;
    switch (preset) {
        case 'top-left':
            nx = 18;
            ny = 18;
            break;
        case 'top-right':
            nx = Math.round(pw - el.offsetWidth - 18);
            ny = 18;
            break;
        case 'bottom-left':
            nx = 18;
            ny = Math.round(ph - el.offsetHeight - 18);
            break;
        case 'bottom-right':
            nx = Math.round(pw - el.offsetWidth - 18);
            ny = Math.round(ph - el.offsetHeight - 18);
            break;
        case 'center':
            nx = Math.round((pw - el.offsetWidth) / 2);
            ny = Math.round((ph - el.offsetHeight) / 2);
            break;
    }
    state.textPos = { x: nx, y: ny };
    updatePreviewFromInputs();
}

// -------------------- ONNX Integration and Camera Processing --------------------
const onnxPath = './fpn_resnet50_model.onnx';
const externalDataPath = './fpn_resnet50_model.onnx.data';
let session = null;
let stream = null;
let running = false;
let fps = 0;
let frameCount = 0;
let startTime = performance.now();
const fpsUpdateInterval = 5;
const imgSize = 384;
const kernelSize = 5;
const minArea = 500;

async function initOnnx() {
    try {
        session = await ort.InferenceSession.create(onnxPath, {
            executionProviders: ['wasm'],
            externalData: [{
                path: 'fpn_resnet50_model.onnx.data',
                data: externalDataPath
            }]
        });
        console.log('ONNX model loaded');
        status.textContent = 'Модель загружена успешно';
    } catch (e) {
        console.error('Failed to load ONNX model:', e);
        status.textContent = 'Ошибка загрузки модели: ' + e.message;
    }
}

const origVideo = document.getElementById('origVideo');
const resultCanvas = document.getElementById('resultCanvas');
const resultCtx = resultCanvas.getContext('2d', { willReadFrequently: true });

// Function to update canvas size based on container
function updateCanvasSize() {
    const winResult = document.getElementById('winResult');
    if (winResult) {
        const rect = winResult.getBoundingClientRect();
        resultCanvas.width = rect.width;
        resultCanvas.height = rect.height;
        if (!running && bgImage.complete && state.currentBg) {
            resultCtx.drawImage(bgImage, 0, 0, resultCanvas.width, resultCanvas.height);
        }
    }
}

// Initial size update
setTimeout(updateCanvasSize, 100);
window.addEventListener('resize', updateCanvasSize);

async function startProcessing() {
    if (running || !session) {
        status.textContent = session ? 'Статус: уже запущено' : 'Ошибка: модель не загружена';
        return;
    }
    running = true;
    status.textContent = 'Статус: предпросмотр запущен';
    startBtn.disabled = true;
    stopBtn.disabled = false;

    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        origVideo.srcObject = stream;
        origVideo.onloadedmetadata = () => {
            origVideo.play();
            const winOrig = document.getElementById('winOriginal');
            origVideo.width = winOrig.clientWidth;
            origVideo.height = winOrig.clientHeight;
            updateCanvasSize(); // Update result canvas to match
            processFrame();
        };
    } catch (e) {
        console.error('Camera access error:', e);
        status.textContent = 'Ошибка: Доступ к камере запрещен';
        running = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

function stopProcessing() {
    if (!running) return;
    running = false;
    status.textContent = 'Статус: остановлено';
    startBtn.disabled = false;
    stopBtn.disabled = true;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    origVideo.srcObject = null;
    resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    // Redraw background after stop if available
    if (bgImage.complete && state.currentBg) {
        resultCtx.drawImage(bgImage, 0, 0, resultCanvas.width, resultCanvas.height);
    }
}

async function processFrame() {
    if (!running) return;

    frameCount++;
    if (frameCount % fpsUpdateInterval === 0) {
        fps = fpsUpdateInterval / ((performance.now() - startTime) / 1000);
        startTime = performance.now();
    }

    // === 1️⃣ Захват кадра с видео ===
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = origVideo.videoWidth;
    tempCanvas.height = origVideo.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(origVideo, 0, 0);

    // === 2️⃣ Масштабируем под вход модели ===
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = imgSize;
    resizeCanvas.height = imgSize;
    const resizeCtx = resizeCanvas.getContext('2d');
    resizeCtx.drawImage(tempCanvas, 0, 0, tempCanvas.width, tempCanvas.height, 0, 0, imgSize, imgSize);
    const resizedData = resizeCtx.getImageData(0, 0, imgSize, imgSize).data;

    // === 3️⃣ Готовим тензор [1, 3, H, W] ===
    const inputTensorData = new Float32Array(1 * 3 * imgSize * imgSize);
    for (let y = 0; y < imgSize; y++) {
        for (let x = 0; x < imgSize; x++) {
            const idx = (y * imgSize + x) * 4;
            const r = resizedData[idx] / 255;
            const g = resizedData[idx + 1] / 255;
            const b = resizedData[idx + 2] / 255;

            // ⚠️ FPN обычно обучен на BGR
            inputTensorData[0 * imgSize * imgSize + y * imgSize + x] = b;
            inputTensorData[1 * imgSize * imgSize + y * imgSize + x] = g;
            inputTensorData[2 * imgSize * imgSize + y * imgSize + x] = r;
        }
    }
    const inputTensor = new ort.Tensor('float32', inputTensorData, [1, 3, imgSize, imgSize]);

    // === 4️⃣ Запуск инференса ===
    const inputs = {};
    inputs[session.inputNames[0]] = inputTensor;
    const outputs = await session.run(inputs);
    const maskTensor = outputs[session.outputNames[0]];

    if (!maskTensor || !maskTensor.data) {
        console.error('⚠️ Модель не вернула маску. Выходы:', session.outputNames);
        requestAnimationFrame(processFrame);
        return;
    }

    // === 5️⃣ Sigmoid + threshold ===
    const rawData = maskTensor.data;
    const maskData = new Uint8Array(imgSize * imgSize);
    for (let i = 0; i < imgSize * imgSize; i++) {
        const prob = 1 / (1 + Math.exp(-rawData[i]));
        maskData[i] = prob > 0.5 ? 255 : 0;
    }

    // === 6️⃣ Маска → canvas ===
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = imgSize;
    maskCanvas.height = imgSize;
    const maskCtx = maskCanvas.getContext('2d');
    const maskImageData = maskCtx.createImageData(imgSize, imgSize);
    for (let i = 0; i < maskData.length; i++) {
        maskImageData.data[i * 4 + 0] = maskData[i];
        maskImageData.data[i * 4 + 1] = maskData[i];
        maskImageData.data[i * 4 + 2] = maskData[i];
        maskImageData.data[i * 4 + 3] = 255;
    }
    maskCtx.putImageData(maskImageData, 0, 0);

    // === 7️⃣ Ресайз маски под оригинал ===
    const fullMaskCanvas = document.createElement('canvas');
    fullMaskCanvas.width = tempCanvas.width;
    fullMaskCanvas.height = tempCanvas.height;
    const fullMaskCtx = fullMaskCanvas.getContext('2d', { imageSmoothingEnabled: false });
    fullMaskCtx.drawImage(maskCanvas, 0, 0, imgSize, imgSize, 0, 0, fullMaskCanvas.width, fullMaskCanvas.height);
    const fullMaskData = fullMaskCtx.getImageData(0, 0, fullMaskCanvas.width, fullMaskCanvas.height);

    // === 8️⃣ Подготовка размеров под результат ===
    const frameResizedCanvas = document.createElement('canvas');
    frameResizedCanvas.width = resultCanvas.width;
    frameResizedCanvas.height = resultCanvas.height;
    const frameResizedCtx = frameResizedCanvas.getContext('2d');
    frameResizedCtx.drawImage(tempCanvas, 0, 0, tempCanvas.width, tempCanvas.height,
        0, 0, frameResizedCanvas.width, frameResizedCanvas.height);
    const frameResizedData = frameResizedCtx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);

    const maskDisplayCanvas = document.createElement('canvas');
    maskDisplayCanvas.width = resultCanvas.width;
    maskDisplayCanvas.height = resultCanvas.height;
    const maskDisplayCtx = maskDisplayCanvas.getContext('2d', { imageSmoothingEnabled: false });
    maskDisplayCtx.drawImage(fullMaskCanvas, 0, 0, fullMaskCanvas.width, fullMaskCanvas.height,
        0, 0, maskDisplayCanvas.width, maskDisplayCanvas.height);
    const maskDisplayData = maskDisplayCtx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);

    // === 9️⃣ Отрисовка фона ===
    if (bgImage.src && bgImage.complete) {
        resultCtx.drawImage(bgImage, 0, 0, resultCanvas.width, resultCanvas.height);
    } else {
        resultCtx.fillStyle = '#111';
        resultCtx.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
    }

    // === 🔟 Применяем маску ===
    const resultData = resultCtx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
    for (let i = 0; i < maskDisplayData.data.length; i += 4) {
        if (maskDisplayData.data[i] > 128) { // foreground
            resultData.data[i] = frameResizedData.data[i];
            resultData.data[i + 1] = frameResizedData.data[i + 1];
            resultData.data[i + 2] = frameResizedData.data[i + 2];
            resultData.data[i + 3] = 255;
        }
    }
    resultCtx.putImageData(resultData, 0, 0);

    // === 11️⃣ FPS overlay ===
    resultCtx.font = '20px sans-serif';
    resultCtx.fillStyle = '#0f0';
    resultCtx.fillText(`FPS: ${fps.toFixed(1)}`, 10, 30);

    requestAnimationFrame(processFrame);
}


// Simple morphology implementations (approximate, for browser performance)
function morphologyOperation(data, kernelSize, isDilate) {
    const width = data.width;
    const height = data.height;
    const output = data.data.slice();
    const half = Math.floor(kernelSize / 2);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let minMax = isDilate ? 0 : 255;
            for (let ky = -half; ky <= half; ky++) {
                for (let kx = -half; kx <= half; kx++) {
                    const ny = Math.min(Math.max(y + ky, 0), height - 1);
                    const nx = Math.min(Math.max(x + kx, 0), width - 1);
                    const val = data.data[(ny * width + nx) * 4];
                    minMax = isDilate ? Math.max(minMax, val) : Math.min(minMax, val);
                }
            }
            output[(y * width + x) * 4] = minMax;
        }
    }
    const newData = new ImageData(new Uint8ClampedArray(output), width, height);
    return newData;
}

function morphologyClose(data, kernelSize) {
    let temp = morphologyOperation(data, kernelSize, true); // Dilate
    return morphologyOperation(temp, kernelSize, false); // Erode
}

function morphologyOpen(data, kernelSize) {
    let temp = morphologyOperation(data, kernelSize, false); // Erode
    return morphologyOperation(temp, kernelSize, true); // Dilate
}

// Simple contour filtering (using flood fill to approximate area)
function filterSmallContours(data, minArea) {
    const width = data.width;
    const height = data.height;
    const output = new Uint8ClampedArray(data.data.length);
    const visited = new Array(width * height).fill(false);

    const directions = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]];

    function floodFill(x, y, setOutput = false) {
        const stack = [[x, y]];
        let area = 0;
        const tempVisited = setOutput ? visited : new Array(width * height).fill(false);
        while (stack.length) {
            const [cx, cy] = stack.pop();
            const idx = cy * width + cx;
            if (tempVisited[idx] || cx < 0 || cx >= width || cy < 0 || cy >= height || data.data[idx * 4] === 0) continue;
            tempVisited[idx] = true;
            area++;
            if (setOutput) {
                output[idx * 4] = 255;
                output[idx * 4 + 1] = 255;
                output[idx * 4 + 2] = 255;
                output[idx * 4 + 3] = 255;
            }
            directions.forEach(([dx, dy]) => stack.push([cx + dx, cy + dy]));
        }
        return area;
    }

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (data.data[idx * 4] > 0 && !visited[idx]) {
                const area = floodFill(x, y);
                if (area > minArea) {
                    floodFill(x, y, true);
                }
            }
        }
    }
    return new ImageData(output, width, height);
}

// -------------------- Actions (start/stop/export) --------------------
startBtn.addEventListener('click', startProcessing);
stopBtn.addEventListener('click', stopProcessing);
exportBtn.addEventListener('click', async() => {
    status.textContent = 'Генерируется PNG...';
    try {
        const exportSize = 1024;
        const canvas = document.createElement('canvas');
        canvas.width = exportSize;
        canvas.height = exportSize;
        const ctx = canvas.getContext('2d');

        // Calculate scale to fit the preview into 1024x1024, maintaining aspect ratio
        const scale = Math.min(exportSize / resultCanvas.width, exportSize / resultCanvas.height);
        const scaledWidth = resultCanvas.width * scale;
        const scaledHeight = resultCanvas.height * scale;
        const offsetX = (exportSize - scaledWidth) / 2;
        const offsetY = (exportSize - scaledHeight) / 2;

        // Draw the resultCanvas (processed image or background) scaled and centered
        ctx.drawImage(resultCanvas, offsetX, offsetY, scaledWidth, scaledHeight);

        // Draw text with scaled positions and font size
        const t = overlayResult.querySelector('#draggableText');
        if (t) {
            const cs = getComputedStyle(t);
            ctx.fillStyle = cs.color;
            const scaledFontSize = parseFloat(cs.fontSize) * scale;
            ctx.font = `${cs.fontWeight} ${scaledFontSize}px ${cs.fontFamily}`;
            ctx.textBaseline = 'top';
            let left = parseFloat(t.style.left || '18') * scale + offsetX;
            let top = parseFloat(t.style.top || '18') * scale + offsetY;
            // Clamp positions within export canvas
            left = Math.max(6 * scale, Math.min(left, exportSize - 10 * scale));
            top = Math.max(6 * scale, Math.min(top, exportSize - 10 * scale));
            const lines = t.textContent.split('\n');
            const lineHeight = scaledFontSize * 1.35;
            for (let i = 0; i < lines.length; i++) {
                const textWidth = ctx.measureText(lines[i]).width;
                // Adjust left if line would overflow
                const adjustedLeft = Math.min(left, exportSize - textWidth - 6 * scale);
                // Ensure top doesn't overflow
                const lineTop = top + i * lineHeight;
                if (lineTop < exportSize) {
                    ctx.fillText(lines[i], adjustedLeft, lineTop);
                }
            }
        }

        // Draw logo with scaled positions and size
        const lg = overlayResult.querySelector('#draggableLogo');
        if (lg && lg.src) {
            ctx.globalAlpha = parseFloat(lg.style.opacity || '1');
            const img = await loadImageElement(lg.src);
            const originalLogoSize = state.logoSize;
            const scaledLogoSize = originalLogoSize * scale;
            const logoScale = Math.min(scaledLogoSize / img.width, scaledLogoSize / img.height);
            const lw = img.width * logoScale;
            const lh = img.height * logoScale;
            let left = parseFloat(lg.style.left || (resultCanvas.width - originalLogoSize)) * scale + offsetX;
            let top = parseFloat(lg.style.top || '18') * scale + offsetY;
            // Clamp positions
            left = Math.max(0, Math.min(left, exportSize - lw));
            top = Math.max(0, Math.min(top, exportSize - lh));
            ctx.drawImage(img, left, top, lw, lh);
            ctx.globalAlpha = 1;
        }

        const data = canvas.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = data;
        a.download = 'preview.png';
        document.body.appendChild(a);
        a.click();
        a.remove();
        status.textContent = 'Экспорт готов';
    } catch (e) {
        console.error(e);
        status.textContent = 'Ошибка при экспорте: ' + e.message;
    }
    setTimeout(() => status.textContent = 'Статус: готов', 2000);
});

// helpers for drawing
function loadImageElement(src) {
    return new Promise((res, rej) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => res(img);
        img.onerror = rej;
        img.src = src;
    });
}

// -------------------- UI glue --------------------
function updateStateFromUI() {
    state.fontFamily = fontSelectValue;
    state.fontSize = +fontSize.value;
    state.fontColor = fontColor.value;
    state.fontWeight = fontWeight.value;
    state.logoOpacity = +logoOpacity.value / 100;
    state.logoSize = +logoSize.value;
    state.privLevel = privSelect.value;
    document.getElementById('fontSizeLabel').textContent = fontSize.value + 'px';
    document.getElementById('logoOpacityLabel').textContent = logoOpacity.value + '%';
    document.getElementById('logoSizeLabel').textContent = logoSize.value + 'px';
}
fontColor.addEventListener('input', () => {
    updateStateFromUI();
    updatePreviewFromInputs();
});
fontSize.addEventListener('input', () => {
    updateStateFromUI();
    updatePreviewFromInputs();
});
fontWeight.addEventListener('change', () => {
    updateStateFromUI();
    updatePreviewFromInputs();
});
logoOpacity.addEventListener('input', () => {
    updateStateFromUI();
    updatePreviewFromInputs();
});
logoSize.addEventListener('input', () => {
    updateStateFromUI();
    updatePreviewFromInputs();
});
privSelect.addEventListener('change', () => {
    updateStateFromUI();
    updatePreviewFromInputs();
});
Object.values(inputs).forEach(inp => inp.addEventListener('input', () => updatePreviewFromInputs()));

// theme toggle
const themeToggle = document.getElementById('themeToggle');
let lightMode = false;
themeToggle.addEventListener('click', () => {
    lightMode = !lightMode;
    if (lightMode) {
        document.documentElement.setAttribute('data-theme', 'light');
        themeToggle.textContent = 'Тёмная тема';
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeToggle.textContent = 'Светлая тема';
    }
});

// -------------------- Init placeholder + attach draggables --------------------
(async function init() {
    const svg = encodeURIComponent(`<svg xmlns='http://www.w3.org/2000/svg' width='400' height='160'><rect rx='16' width='100%' height='100%' fill='#7C4DFF'/><text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' fill='white' font-size='20' font-family='sans-serif'>+VibeCoders</text></svg>`);
    const url = `data:image/svg+xml;utf8,${svg}`;
    setLogo(url);
    await initOnnx();
    attachDraggables();
    updatePreviewFromInputs();
})();