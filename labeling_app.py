import math
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="부품 검사 라벨링",
    page_icon=":material/design_services:",
    layout="wide",
)

IMAGE_DIR = Path(__file__).parent
IMAGE_FILES = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))
BASE_IMAGE = "hard3.jpg" if "hard3.jpg" in IMAGE_FILES else (IMAGE_FILES[0] if IMAGE_FILES else None)

TOOLS = {
    "Detect": ":material/crop_square:",
    "OBB": ":material/crop_rotate:",
    "Segment": ":material/gesture:",
    "Pose": ":material/scatter_plot:",
}
TOOL_DESC = {
    "Detect": "박스 — 물체가 어디에 몇 개 있는지만 필요할 때",
    "OBB": "회전 박스 — 물체가 기울어져 있을 때",
    "Segment": "외곽선 — 모양·면적이 중요할 때",
    "Pose": "키포인트 — 지점 간 위치 관계가 중요할 때",
}

YELLOW = (217, 164, 4)
INK = (17, 17, 19)


@st.cache_data
def build_preview(tool: str) -> Image.Image:
    """도구별 정적 미리보기 — 실제 드로잉 기능은 없음, UI 목업용."""
    img = Image.open(IMAGE_DIR / BASE_IMAGE).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    if tool == "Detect":
        for x1, y1, x2, y2 in [(0.30, 0.26, 0.44, 0.48), (0.58, 0.55, 0.70, 0.74)]:
            draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline=YELLOW, width=5)

    elif tool == "OBB":
        cx, cy, bw, bh, ang = 0.5 * w, 0.48 * h, 0.20 * w, 0.11 * h, -18
        rad = math.radians(ang)
        pts = []
        for dx, dy in [(-bw / 2, -bh / 2), (bw / 2, -bh / 2), (bw / 2, bh / 2), (-bw / 2, bh / 2)]:
            rx = dx * math.cos(rad) - dy * math.sin(rad)
            ry = dx * math.sin(rad) + dy * math.cos(rad)
            pts.append((cx + rx, cy + ry))
        draw.polygon(pts, outline=YELLOW, width=5)

    elif tool == "Segment":
        cx, cy = 0.42 * w, 0.55 * h
        pts = [
            (cx + dx * w * 0.09, cy + dy * h * 0.09)
            for dx, dy in [(-1, 0.3), (-0.4, -1), (0.6, -0.9), (1, 0.2), (0.5, 1), (-0.5, 0.8)]
        ]
        draw.polygon(pts, outline=YELLOW, width=5)

    elif tool == "Pose":
        pts = [(0.30 * w, 0.75 * h), (0.42 * w, 0.40 * h), (0.58 * w, 0.62 * h)]
        for a, b in zip(pts, pts[1:]):
            draw.line([a, b], fill=YELLOW, width=4)
        for x, y in pts:
            r = 8
            draw.ellipse([x - r, y - r, x + r, y + r], fill=YELLOW, outline=INK, width=2)

    return img


if not IMAGE_FILES:
    st.error(f"{IMAGE_DIR}에 .jpg 이미지가 없습니다.")
    st.stop()

with st.sidebar:
    st.markdown("### 부품 검사 라벨링")
    st.caption("프로토타입 · UI 목업 (기능 미구현)")

    st.selectbox("이미지", IMAGE_FILES, index=IMAGE_FILES.index(BASE_IMAGE))
    st.caption("1,247 / 2,000")
    st.badge("VLM 로드됨 · Qwen3-VL-4B · 2.9GB", icon=":material/memory:", color="yellow")

    st.write("")
    st.markdown("**검수 모델 (YOLO)**")
    st.selectbox("모델 선택", ["bolt-detector-v3", "stud-detector-v1"], label_visibility="collapsed")
    st.button(
        "이 모델로 재라벨링",
        icon=":material/model_training:",
        width="stretch",
        disabled=True,
        help="프로토타입 — 실제 추론은 연결되지 않음",
    )

    st.write("")
    st.button("labels.json 다운로드", icon=":material/download:", width="stretch", disabled=True)

top_l, top_r = st.columns([3, 1], vertical_alignment="center")
with top_l:
    st.title("부품 검사 라벨링")
with top_r:
    st.button(
        "자동 라벨링 실행",
        icon=":material/auto_awesome:",
        type="primary",
        width="stretch",
        disabled=True,
    )

tool = st.segmented_control(
    "도구",
    list(TOOLS.keys()),
    format_func=lambda t: t,
    default="Detect",
    required=True,
    key="tool",
)
st.caption(TOOL_DESC[tool])

canvas_col, review_col = st.columns([3, 2], gap="large")

with canvas_col:
    st.image(build_preview(tool), width="stretch")
    st.caption(f"{tool} 선택 시 예시 미리보기 — 실제 드로잉은 이 프로토타입에 연결되지 않음")

with review_col:
    st.markdown("**검수**")
    st.caption("AI 제안 47 · 완료 12")

    demo_rows = [
        ("stud", 0.94, "yellow"),
        ("stud", 0.89, "yellow"),
        ("latch", 0.61, "orange"),
        ("bead", 0.97, "green"),
        ("joint", 0.42, "red"),
    ]
    for i, (cls, conf, color) in enumerate(demo_rows):
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="center")
            c1.markdown(f"**{cls}**")
            c2.badge(f"{conf:.2f}", color=color)
            with c3:
                b1, b2 = st.columns(2)
                b1.button(":material/check:", key=f"ok_{i}", width="stretch", disabled=True)
                b2.button(":material/close:", key=f"no_{i}", width="stretch", disabled=True)

st.write("")
status_l, status_m, status_r = st.columns([2, 4, 2], vertical_alignment="center")
with status_m:
    st.progress(62, text="자동 라벨링 · 1,247 / 2,000 (62%) · 평균 4.1s/장 · 남은 약 48분")
with status_r:
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.button("일시정지", icon=":material/pause:", disabled=True)
        st.button("취소", icon=":material/close:", disabled=True)
