import io
import csv
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from services.detector import ShelfEmptySpaceDetector


st.set_page_config(
    page_title="Детекция пустых мест на полках",
    page_icon="🛒",
    layout="wide"
)

st.title("Детекция пустых мест на магазинных полках")


@st.cache_resource
def load_detector() -> ShelfEmptySpaceDetector:
    return ShelfEmptySpaceDetector("models/best.pt")


def load_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def draw_boxes(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    label_key: str = "label"
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det.get("confidence", 0.0)
        label = det.get(label_key, "empty_space")

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1 + 4, max(0, y1 - 18)), f"{label}: {score:.2f}", fill="red")

    return result


def detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        rows.append(
            {
                "id": idx,
                "label": det.get("label", "empty_space"),
                "confidence": round(det.get("confidence", 0.0), 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "area": (x2 - x1) * (y2 - y1),
            }
        )
    return pd.DataFrame(rows)


def build_csv_report(df: pd.DataFrame, source_name: str) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["report_generated_at", datetime.now().isoformat()])
    writer.writerow(["source_name", source_name])
    writer.writerow(["empty_spaces_count", len(df)])
    writer.writerow([])

    if not df.empty:
        writer.writerow(df.columns.tolist())
        for _, row in df.iterrows():
            writer.writerow(row.tolist())
    else:
        writer.writerow(["detections", "not_found"])

    return output.getvalue().encode("utf-8")


with st.sidebar:
    st.header("Настройки")

    input_mode = st.radio(
        "Источник изображения",
        ["Загрузка файла", "Камера"],
        index=0
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.50,
        step=0.05
    )

    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05
    )

    imgsz = st.selectbox(
        "Imgsz",
        options=[640, 960, 1280],
        index=2
    )

    show_table = st.checkbox("Показать таблицу детекций", value=True)
    show_original = st.checkbox("Показать исходное изображение", value=True)


image = None
source_name = "unknown"

if input_mode == "Загрузка файла":
    uploaded_file = st.file_uploader(
        "Загрузите изображение полки",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        source_name = uploaded_file.name
else:
    camera_file = st.camera_input("Сделайте снимок полки")
    if camera_file is not None:
        image = load_image(camera_file)
        source_name = "camera_capture.png"


if image is None:
    st.info("Загрузите изображение или сделайте снимок с камеры.")
    st.stop()


try:
    detector = load_detector()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()


with st.spinner("Выполняется детекция..."):
    detections = detector.predict(
        image=image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )

result_image = draw_boxes(image, detections)
df = detections_to_dataframe(detections)

col1, col2 = st.columns(2)

with col1:
    if show_original:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)

with col2:
    st.subheader("Результат детекции")
    st.image(result_image, use_container_width=True)
    st.metric("Количество пустых мест", len(detections))


if show_table:
    st.subheader("Найденные пустые области")
    if df.empty:
        st.warning("Пустые зоны не обнаружены.")
    else:
        st.dataframe(df, use_container_width=True)


st.subheader("Экспорт результатов")

report_col1, report_col2 = st.columns(2)

with report_col1:
    csv_bytes = build_csv_report(df, source_name)
    st.download_button(
        label="Скачать CSV-отчёт",
        data=csv_bytes,
        file_name=f"empty_spaces_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with report_col2:
    st.download_button(
        label="Скачать изображение с разметкой",
        data=image_to_bytes(result_image),
        file_name=f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png"
    )