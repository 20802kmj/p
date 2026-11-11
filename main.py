import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
APP_TITLE = "사망율 분석"
DATA_URL = "https://raw.githubusercontent.com/20802kmj/data/refs/heads/main/aaa.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "사망율_분석_기록.json"

EXPECTED_COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
)


# --------------------------------------------------
# 기록 파일 읽기 / 저장
# --------------------------------------------------
def load_records():
    """저장된 우리말 뜻과 데이터 출처를 불러온다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        if not isinstance(saved, dict):
            return {"meanings": {}, "source": ""}

        meanings = saved.get("meanings", {})
        source = saved.get("source", "")

        return {
            "meanings": meanings if isinstance(meanings, dict) else {},
            "source": source if isinstance(source, str) else "",
        }
    except (json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_records(meanings, source):
    """입력한 내용을 내_기록 폴더의 JSON 파일에 저장한다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    saved = {
        "meanings": {
            str(column): str(meaning)
            for column, meaning in meanings.items()
        },
        "source": str(source),
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(saved, file, ensure_ascii=False, indent=4)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    return data[EXPECTED_COLUMNS]


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title(f"🩺 {APP_TITLE}")
st.caption("데이터를 살펴보고 각 열의 의미를 교재를 참고해 정리하는 앱입니다.")

st.markdown(
    """
    ### 데이터 소개
    이 데이터는 사람들의 기본 정보와 건강 관련 항목을 담고 있으며,
    마지막의 `stroke` 열에는 뇌졸중 발생 여부가 기록되어 있습니다.
    `stroke` 값이 1이면 뇌졸중이 발생한 경우, 0이면 발생하지 않은 경우를 뜻합니다.
    """
)

try:
    df = load_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.\n\n오류: {error}")
    st.stop()

# 기록 불러오기
saved_records = load_records()
saved_meanings = saved_records["meanings"]
saved_source = saved_records["source"]

# 핵심 숫자 계산
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (stroke_count / total_people * 100) if total_people else 0.0

st.markdown("### 데이터 한눈에 보기")
metric_columns = st.columns(4)

metrics = [
    ("전체 사람 수", f"{total_people:,}명"),
    ("열 개수", f"{column_count:,}개"),
    ("stroke가 1인 사람 수", f"{stroke_count:,}명"),
    ("stroke가 1인 비율", f"{stroke_ratio:.2f}%"),
]

for column, (label, value) in zip(metric_columns, metrics):
    column.metric(label=label, value=value)

st.divider()

# 열 정보 표
st.markdown("### 열 정보")
st.caption("‘우리말 뜻’ 칸을 교재를 참고해 직접 입력하면 자동으로 저장됩니다.")

column_info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [
            saved_meanings.get(column, "")
            for column in df.columns
        ],
        "값의 종류": [
            int(df[column].nunique(dropna=True))
            for column in df.columns
        ],
        "빈 값 개수": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
    }
)

edited_info = st.data_editor(
    column_info,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 우리말 뜻을 입력하세요.",
        ),
        "값의 종류": st.column_config.NumberColumn(
            "값의 종류",
            help="빈 값을 제외한 서로 다른 값의 개수입니다.",
        ),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
    key="column_info_editor",
)

# 표에서 입력한 내용을 저장
current_meanings = {
    row["열 이름"]: row["우리말 뜻"]
    for _, row in edited_info.iterrows()
}

st.divider()

# 원본 데이터 미리보기
st.markdown("### 데이터 미리보기")
st.caption("CSV 데이터의 처음 다섯 줄을 원본 값 그대로 표시합니다.")
st.dataframe(df.head(5), hide_index=True, use_container_width=True)

st.divider()

# 출처 입력
st.markdown("### 데이터 출처")
st.caption("교재에 적힌 데이터 출처를 입력하세요. 입력한 내용은 자동 저장됩니다.")

source_text = st.text_area(
    "데이터 출처",
    value=saved_source,
    placeholder="교재에 적힌 데이터 출처를 여기에 입력하세요.",
    height=100,
    key="source_text",
    label_visibility="collapsed",
)

# 변경 사항 저장
try:
    save_records(current_meanings, source_text)
    st.caption(f"입력한 내용이 `{RECORD_FILE.as_posix()}` 파일에 저장됩니다.")
except OSError as error:
    st.error(f"기록 파일을 저장하지 못했습니다: {error}")

with st.expander("기록 저장 안내"):
    st.write(
        "우리말 뜻과 데이터 출처는 프로젝트 폴더 안의 "
        "`내_기록/사망율_분석_기록.json` 파일에 저장됩니다. "
        "앱을 다시 실행하거나 새로고침하면 저장된 내용을 불러옵니다."
    )
