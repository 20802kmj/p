from pathlib import Path
import py_compile

main_code = r'''import json
import itertools
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
APP_TITLE = "사망율 분석"
DATA_URL = "https://raw.githubusercontent.com/ebbeberge/stroke-prediction/main/stroke.csv"

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
# 연관분석용 데이터 전처리 및 Apriori 알고리즘
# --------------------------------------------------
def make_transactions(data):
    """연속형 데이터를 구간화하고 각 행을 연관분석 거래로 변환한다."""
    assoc = data.copy()

    # BMI 결측치를 중앙값으로 대체
    assoc["bmi"] = pd.to_numeric(assoc["bmi"], errors="coerce")
    bmi_median = assoc["bmi"].median()
    assoc["bmi"] = assoc["bmi"].fillna(bmi_median)

    # 연령대 구간화
    assoc["age_group"] = pd.cut(
        assoc["age"],
        bins=[0, 30, 50, 70, float("inf")],
        labels=["30세 이하", "31~50세", "51~70세", "70세 초과"],
        include_lowest=True,
    )

    # 평균 혈당 구간화
    assoc["glucose_group"] = pd.cut(
        assoc["avg_glucose_level"],
        bins=[0, 100, 140, float("inf")],
        labels=["100 미만", "100~140", "140 초과"],
        include_lowest=True,
    )

    # BMI 구간화
    assoc["bmi_group"] = pd.cut(
        assoc["bmi"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=["저체중", "정상 범위", "과체중", "비만 범위"],
        include_lowest=True,
    )

    # 0과 1을 읽기 쉬운 범주로 변환
    assoc["hypertension"] = assoc["hypertension"].map(
        {0: "없음", 1: "있음"}
    ).fillna("알 수 없음")

    assoc["heart_disease"] = assoc["heart_disease"].map(
        {0: "없음", 1: "있음"}
    ).fillna("알 수 없음")

    assoc["stroke"] = assoc["stroke"].map(
        {0: "미발생", 1: "발생"}
    ).fillna("알 수 없음")

    # 분석에 사용할 변수
    features = [
        "age_group",
        "gender",
        "hypertension",
        "heart_disease",
        "glucose_group",
        "bmi_group",
        "smoking_status",
        "stroke",
    ]

    transactions = []
    for _, row in assoc[features].iterrows():
        transaction = set()

        for column in features:
            value = row[column]

            if pd.isna(value):
                value = "알 수 없음"

            transaction.add(f"{column}={value}")

        transactions.append(transaction)

    return transactions


def run_apriori(transactions, min_support=0.003, max_itemset_size=3):
    """외부 연관분석 라이브러리 없이 Apriori 알고리즘을 실행한다."""
    transaction_count = len(transactions)

    if transaction_count == 0:
        return pd.DataFrame()

    # 1개 항목 조합의 지지도 계산
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            key = frozenset([item])
            item_counts[key] = item_counts.get(key, 0) + 1

    frequent_itemsets = {
        itemset: count / transaction_count
        for itemset, count in item_counts.items()
        if count / transaction_count >= min_support
    }

    # 2개 항목부터 최대 3개 항목까지 조합 탐색
    for size in range(2, max_itemset_size + 1):
        previous_itemsets = [
            itemset
            for itemset in frequent_itemsets
            if len(itemset) == size - 1
        ]
        previous_set = set(previous_itemsets)
        candidates = set()

        for first, second in itertools.combinations(previous_itemsets, 2):
            candidate = first | second

            if len(candidate) != size:
                continue

            # 후보의 모든 부분집합이 빈발 항목집합인지 확인
            all_subsets_frequent = all(
                frozenset(subset) in previous_set
                for subset in itertools.combinations(candidate, size - 1)
            )

            if all_subsets_frequent:
                candidates.add(candidate)

        for candidate in candidates:
            count = sum(
                candidate.issubset(transaction)
                for transaction in transactions
            )
            support = count / transaction_count

            if support >= min_support:
                frequent_itemsets[candidate] = support

    # 뇌졸중 발생(stroke=발생)을 결과로 하는 규칙 생성
    target = frozenset(["stroke=발생"])
    target_support = frequent_itemsets.get(target, 0)

    rules = []

    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2 or target not in itemset:
            continue

        antecedent = itemset - target
        antecedent_support = frequent_itemsets.get(antecedent, 0)

        if antecedent_support == 0 or target_support == 0:
            continue

        confidence = support / antecedent_support
        lift = confidence / target_support

        rules.append({
            "조건": ", ".join(sorted(antecedent)),
            "결과": "stroke=발생",
            "발생 건수": round(support * transaction_count),
            "지지도(%)": support * 100,
            "신뢰도(%)": confidence * 100,
            "향상도": lift,
        })

    rules_df = pd.DataFrame(rules)

    if not rules_df.empty:
        rules_df = rules_df.sort_values(
            ["향상도", "신뢰도(%)"],
            ascending=False,
        ).reset_index(drop=True)

    return rules_df


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title(f"🩺 {APP_TITLE}")
st.caption("뇌졸중 데이터의 특성을 살펴보고 연관 패턴을 분석하는 앱입니다.")

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
    st.error(
        "데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.\n\n"
        f"오류: {error}"
    )
    st.stop()


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------
saved_records = load_records()
saved_meanings = saved_records["meanings"]
saved_source = saved_records["source"]

total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (
    stroke_count / total_people * 100
    if total_people else 0.0
)

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


# --------------------------------------------------
# 열 정보 표
# --------------------------------------------------
st.markdown("### 열 정보")
st.caption("‘우리말 뜻’ 칸을 교재를 참고해 직접 입력하면 자동으로 저장됩니다.")

column_info = pd.DataFrame({
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
})

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

current_meanings = {
    row["열 이름"]: row["우리말 뜻"]
    for _, row in edited_info.iterrows()
}

st.divider()


# --------------------------------------------------
# 데이터 미리보기
# --------------------------------------------------
st.markdown("### 데이터 미리보기")
st.caption("CSV 데이터의 처음 다섯 줄을 원본 값 그대로 표시합니다.")
st.dataframe(
    df.head(5),
    hide_index=True,
    use_container_width=True,
)

st.divider()


# --------------------------------------------------
# 데이터 출처 입력
# --------------------------------------------------
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

try:
    save_records(current_meanings, source_text)
    st.caption(
        f"입력한 내용이 `{RECORD_FILE.as_posix()}` 파일에 저장됩니다."
    )
except OSError as error:
    st.error(f"기록 파일을 저장하지 못했습니다: {error}")

with st.expander("기록 저장 안내"):
    st.write(
        "우리말 뜻과 데이터 출처는 프로젝트 폴더 안의 "
        "`내_기록/사망율_분석_기록.json` 파일에 저장됩니다. "
        "앱을 다시 실행하거나 새로고침하면 저장된 내용을 불러옵니다."
    )


# --------------------------------------------------
# 연관분석 모델링
# --------------------------------------------------
st.divider()
st.header("🔎 연관분석 모델링")
st.write(
    "Apriori 알고리즘으로 건강 특성들이 함께 나타나는 패턴을 찾고, "
    "뇌졸중 발생(stroke=1)과 연결되는 연관규칙을 확인합니다."
)

st.info(
    "이 분석은 건강 특성 사이의 연관성을 탐색합니다. "
    "뇌졸중 발생 원인을 증명하거나 개인의 발병 가능성을 진단하는 도구는 아닙니다."
)

support_percent = st.slider(
    "최소 지지도(%)",
    min_value=0.1,
    max_value=5.0,
    value=0.3,
    step=0.1,
    help="전체 데이터 중 해당 항목 조합이 나타나야 하는 최소 비율입니다.",
)

confidence_percent = st.slider(
    "최소 신뢰도(%)",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    help="조건이 나타난 사람 중 뇌졸중 발생도 함께 나타나는 최소 비율입니다.",
)

with st.spinner("연관규칙을 분석하고 있습니다..."):
    transactions = make_transactions(df)
    rules_df = run_apriori(
        transactions,
        min_support=support_percent / 100,
        max_itemset_size=3,
    )

if not rules_df.empty:
    # 사용자가 선택한 최소 신뢰도 적용
    displayed_rules = rules_df[
        rules_df["신뢰도(%)"] >= confidence_percent
    ].copy()

    # 우연한 동시 발생 수준보다 높은 규칙을 우선 표시
    displayed_rules = displayed_rules[
        displayed_rules["향상도"] > 1
    ].copy()

    if not displayed_rules.empty:
        st.success(
            f"조건에 맞는 연관규칙 {len(displayed_rules)}개를 찾았습니다."
        )

        st.caption(
            "향상도가 1보다 크면 전체 뇌졸중 발생 비율보다 "
            "해당 조건에서 뇌졸중 발생이 더 자주 함께 나타났다는 뜻입니다."
        )

        display_table = displayed_rules.copy()
        display_table["지지도(%)"] = display_table["지지도(%)"].round(2)
        display_table["신뢰도(%)"] = display_table["신뢰도(%)"].round(2)
        display_table["향상도"] = display_table["향상도"].round(2)

        st.dataframe(
            display_table,
            hide_index=True,
            use_container_width=True,
        )

        chart_data = displayed_rules.head(10).copy()
        chart_data["규칙"] = chart_data["조건"].str.slice(0, 65)

        fig = px.bar(
            chart_data,
            x="향상도",
            y="규칙",
            orientation="h",
            title="뇌졸중 발생 관련 연관규칙 상위 10개",
            hover_data=["지지도(%)", "신뢰도(%)", "발생 건수"],
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="향상도(Lift)",
            yaxis_title="건강 특성 조합",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "연관분석 결과 CSV 다운로드",
            data=displayed_rules.to_csv(
                index=False,
                encoding="utf-8-sig",
            ),
            file_name="뇌졸중_연관분석_결과.csv",
            mime="text/csv",
        )
    else:
        st.warning(
            "현재 기준에 맞는 규칙이 없습니다. "
            "최소 지지도나 최소 신뢰도를 낮춰 다시 분석해 보세요."
        )
else:
    st.warning(
        "현재 최소 지지도 기준으로 발견된 연관규칙이 없습니다. "
        "최소 지지도를 낮춰 다시 분석해 보세요."
    )

with st.expander("연관분석 지표 설명"):
    st.markdown(
        """
        - **지지도(Support):** 전체 데이터에서 조건과 결과가 함께 나타난 비율
        - **신뢰도(Confidence):** 조건이 나타났을 때 뇌졸중 발생도 함께 나타난 비율
        - **향상도(Lift):** 조건이 있을 때의 뇌졸중 발생 비율이 전체 발생 비율보다 얼마나 높은지 나타내는 값
        - **향상도 해석:** 1보다 크면 양의 연관성, 1이면 독립에 가까움, 1보다 작으면 음의 연관성을 뜻함
        """
    )
'''

out = Path("/mnt/data/main.py")
out.write_text(main_code, encoding="utf-8")
py_compile.compile(str(out), doraise=True)

print(f"완성: {out}")
print(f"파일 크기: {out.stat().st_size:,} bytes")
print("파이썬 문법 검사 통과")
