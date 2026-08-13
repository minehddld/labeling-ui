from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="부품 검사 라벨링",
    page_icon=":material/design_services:",
    layout="wide",
)

IMAGE_DIR = Path(__file__).parent
IMAGE_FILES = sorted(p.name for p in IMAGE_DIR.glob("*.jpg"))

TOOL_DESC = {
    "Detect": "박스 — 물체가 어디에 몇 개 있는지만 필요할 때",
    "OBB": "회전 박스 — 물체가 기울어져 있을 때",
    "Segment": "외곽선 — 모양·면적이 중요할 때",
    "Pose": "키포인트 — 지점 간 위치 관계가 중요할 때",
}

DEMO_ROWS = [
    # (class, confidence, seed_status)
    ("stud", 0.94, "approved"),
    ("stud", 0.89, "approved"),
    ("stud", 0.91, "approved"),
    ("stud", 0.85, "approved"),
    ("latch", 0.61, None),
    ("latch", 0.55, "rejected"),
    ("bead", 0.97, "approved"),
    ("bead", 0.93, "approved"),
    ("joint", 0.42, "rejected"),
    ("joint", 0.78, None),
]
STATUS_COLOR = {"approved": "green", "rejected": "red", None: "yellow"}
STATUS_LABEL_KO = {"approved": "승인", "rejected": "거부", None: "대기"}
STATUS_HEX = {"승인": "#3FAE5C", "거부": "#D9534F", "대기": "#8A8578"}

st.session_state.setdefault("progress_val", 0)
st.session_state.setdefault("running", True)
st.session_state.setdefault(
    "review_status", {i: seed for i, (_, _, seed) in enumerate(DEMO_ROWS)}
)  # idx -> "approved" | "rejected" | None(대기)

if not IMAGE_FILES:
    IMAGE_FILES = ["(이미지 없음)"]

with st.sidebar:
    st.markdown("### 부품 검사 라벨링")
    st.caption("프로토타입 · UI 목업 — 결과물(이미지·검출)은 비워둠")

    image_name = st.selectbox("이미지", IMAGE_FILES, key="image_name")
    st.caption(f"{IMAGE_FILES.index(image_name) + 1} / {len(IMAGE_FILES)}")
    st.badge("VLM 로드됨 · Qwen3-VL-4B · 2.9GB", icon=":material/memory:", color="yellow")

    st.write("")
    st.markdown("**검수 모델 (YOLO)**")
    st.selectbox("모델 선택", ["bolt-detector-v3", "stud-detector-v1"], label_visibility="collapsed")
    st.button(
        "이 모델로 재라벨링",
        icon=":material/model_training:",
        width="stretch",
        disabled=True,
        help="개발 중 — 실제 추론 파이프라인은 아직 연결되지 않음",
    )

    st.write("")
    st.button(
        "labels.json 다운로드",
        icon=":material/download:",
        width="stretch",
        disabled=True,
        help="개발 중 — 내보낼 라벨 데이터가 아직 없음",
    )

top_l, top_r = st.columns([3, 1], vertical_alignment="center")
with top_l:
    st.title("부품 검사 라벨링")
with top_r:
    if st.button(
        "자동 라벨링 실행",
        icon=":material/auto_awesome:",
        type="primary",
        width="stretch",
    ):
        st.session_state["progress_val"] = 0
        st.session_state["running"] = True
        st.toast("자동 라벨링을 시작했습니다 (데모 진행률)", icon=":material/auto_awesome:")

tool = st.segmented_control(
    "도구",
    list(TOOL_DESC.keys()),
    default="Detect",
    required=True,
    key="tool",
)
st.caption(TOOL_DESC[tool])

canvas_col, review_col = st.columns([2, 3], gap="large")

with canvas_col:
    with st.container(border=True, height=420):
        st.markdown(
            f":material/construction: **{tool} 미리보기 — 개발 중**",
            text_alignment="center",
        )
        st.caption(f"선택된 이미지: {image_name}", text_alignment="center")
        st.caption("이미지 렌더링과 실제 드로잉 기능은 아직 연결되지 않았습니다", text_alignment="center")

with review_col:
    st.markdown("**검수**")

    statuses = [st.session_state["review_status"][i] for i in range(len(DEMO_ROWS))]
    n_total = len(DEMO_ROWS)
    n_approved = statuses.count("approved")
    n_rejected = statuses.count("rejected")
    n_pending = n_total - n_approved - n_rejected
    avg_conf = sum(c for _, c, _ in DEMO_ROWS) / n_total

    with st.container(horizontal=True):
        st.metric("전체 제안", n_total, border=True)
        st.metric("승인", n_approved, border=True)
        st.metric("거부", n_rejected, border=True)
        st.metric("대기", n_pending, border=True)
        st.metric("평균 신뢰도", f"{avg_conf:.2f}", border=True)

    chart_l, chart_r = st.columns(2, gap="medium")
    with chart_l:
        with st.container(border=True):
            st.caption("클래스별 제안 개수")
            class_counts = (
                pd.DataFrame(DEMO_ROWS, columns=["class", "conf", "seed"])
                .groupby("class", as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            st.bar_chart(class_counts, x="class", y="count", height=180, x_label="", y_label="")

    with chart_r:
        with st.container(border=True):
            st.caption("검수 상태 분포")
            status_df = pd.DataFrame(
                {
                    "status": ["승인", "거부", "대기"],
                    "count": [n_approved, n_rejected, n_pending],
                }
            )
            status_chart = (
                alt.Chart(status_df)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title=None),
                    y=alt.Y("status:N", sort=["승인", "거부", "대기"], title=None),
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(
                            domain=list(STATUS_HEX.keys()),
                            range=list(STATUS_HEX.values()),
                        ),
                        legend=None,
                    ),
                )
                .properties(height=180)
            )
            st.altair_chart(status_chart, width="stretch")

    with st.container(border=True, height=280):
        for i, (cls, conf, _) in enumerate(DEMO_ROWS):
            status = st.session_state["review_status"][i]
            c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="center")
            cls_label = f":gray[~~{cls}~~]" if status == "rejected" else f"**{cls}**"
            c1.markdown(cls_label)
            c2.badge(f"{conf:.2f}", color=STATUS_COLOR[status])
            with c3:
                b1, b2 = st.columns(2)
                ok_type = "primary" if status == "approved" else "secondary"
                no_type = "primary" if status == "rejected" else "secondary"
                if b1.button(":material/check:", key=f"ok_{i}", width="stretch", type=ok_type):
                    st.session_state["review_status"][i] = "approved"
                    st.rerun()
                if b2.button(":material/close:", key=f"no_{i}", width="stretch", type=no_type):
                    st.session_state["review_status"][i] = "rejected"
                    st.rerun()

    st.caption("승인/거부는 이 화면 상태만 바뀝니다 — 실제 라벨 데이터는 아직 저장되지 않음")

st.write("")


@st.fragment(run_every="0.4s")
def status_bar():
    _, status_m, status_r = st.columns([2, 4, 2], vertical_alignment="center")

    if st.session_state["running"]:
        st.session_state["progress_val"] = (st.session_state["progress_val"] + 2) % 101

    pct = st.session_state["progress_val"]
    n_done = round(2000 * pct / 100)
    with status_m:
        st.progress(pct, text=f"자동 라벨링 · {n_done:,} / 2,000 ({pct}%) · 평균 4.1s/장 · 데모 진행률")

    with status_r:
        with st.container(horizontal=True, horizontal_alignment="right"):
            pause_label = "이어서" if not st.session_state["running"] else "일시정지"
            pause_icon = ":material/play_arrow:" if not st.session_state["running"] else ":material/pause:"
            if st.button(pause_label, icon=pause_icon, key="pause_btn"):
                st.session_state["running"] = not st.session_state["running"]
            if st.button("취소", icon=":material/close:", key="cancel_btn"):
                st.session_state["progress_val"] = 0
                st.session_state["running"] = False


status_bar()
