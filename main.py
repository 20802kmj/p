
import json
import itertools
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
APP_TITLE = "사망율 분석"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "ebbeberge/stroke-prediction/main/stroke.csv"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "사망율_분석_기록.json"

EXPECTED_COLUMNS = [
    "id", "gender", "age", "hypertension",
    "heart_disease", "ever_married", "work_type",
    "Residence_type", "avg_glucose_level", "bmi",
    "smoking_status", "stroke"
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# 기록 파일 읽기 / 저장
# --------------------------------------------------
def load_records():
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        if not isinstance(saved, dict):
            return {"meanings": {}, "source": ""}

        meanings = saved.get("meanings", {})

        if not isinstance(meanings, dict):
            meanings = {}

        return {
            "meanings": meanings,
            "source": saved.get("source", "")
        }

    except (json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_records(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    saved = {
        "meanings": meanings,
        "source": source
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
        col for col in EXPECTED_COLUMNS
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV 파일에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    return data[EXPECTED_COLUMNS]


# --------------------------------------------------
# 연관분석 데이터 전처리
# --------------------------------------------------
def make_transactions(data):
    assoc = data.copy()

    # BMI 결측치 처리
    assoc["bmi"] = pd.to_numeric(
        assoc["bmi"], errors="coerce"
    )

    assoc["bmi"] = assoc["bmi"].fillna(
        assoc["bmi"].median()
    )

    # 연령대 구분
    assoc["age_group"] = pd.cut(
        assoc["age"],
        bins=[0, 30, 50, 70, float("inf")],
        labels=[
            "30세 이하",
            "31~50세",
            "51~70세",
            "70세 초과"
        ],
        include_lowest=True
    )

    # 평균 혈당 구분
    assoc["glucose_group"] = pd.cut(
        assoc["avg_glucose_level"],
        bins=[0, 100, 140, float("inf")],
        labels=[
            "100 미만",
            "100~140",
            "140 초과"
        ],
        include_lowest=True
    )

    # BMI 구분
    assoc["bmi_group"] = pd.cut(
        assoc["bmi"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=[
            "저체중",
            "정상 범위",
            "과체중",
            "비만 범위"
        ],
        include_lowest=True
    )

    # 질환 여부 변환
    assoc["hypertension"] = assoc[
        "hypertension"
    ].map({
        0: "없음",
        1: "있음"
    }).fillna("알 수 없음")

    assoc["heart_disease"] = assoc[
        "heart_disease"
    ].map({
        0: "없음",
        1: "있음"
    }).fillna("알 수 없음")

    # 뇌졸중 발생 여부 변환
    assoc["stroke"] = assoc["stroke"].map({
        0: "미발생",
        1: "발생"
    }).fillna("알 수 없음")

    features = [
        "age_group",
        "gender",
        "hypertension",
        "heart_disease",
        "glucose_group",
        "bmi_group",
        "smoking_status",
        "stroke"
    ]

    transactions = []

    for _, row in assoc[features].iterrows():
        transaction = set()

        for col in features:
            value = row[col]

            if pd.isna(value):
                value = "알 수 없음"

            transaction.add(f"{col}={value}")

        transactions.append(transaction)

    return transactions


# --------------------------------------------------
# Apriori 알고리즘
# --------------------------------------------------
def run_apriori(
    transactions,
    min_support=0.003,
    max_itemset_size=3
):
    total = len(transactions)

    if total == 0:
        return pd.DataFrame()

    # 1개 항목의 지지도 계산
    counts = {}

    for transaction in transactions:
        for item in transaction:
            key = frozenset([item])
            counts[key] = counts.get(key, 0) + 1

    frequent = {
        itemset: count / total
        for itemset, count in counts.items()
        if count / total >= min_support
    }

    # 2개 이상 항목 조합 탐색
    for size in range(2, max_itemset_size + 1):

        previous = [
            itemset for itemset in frequent
            if len(itemset) == size - 1
        ]

        previous_set = set(previous)
        candidates = set()

        for a, b in itertools.combinations(previous, 2):
            candidate = a | b

            if len(candidate) != size:
                continue

            all_frequent = all(
                frozenset(subset) in previous_set
                for subset in itertools.combinations(
                    candidate, size - 1
                )
            )

            if all_frequent:
                candidates.add(candidate)

        for candidate in candidates:
            count = sum(
                candidate.issubset(transaction)
                for transaction in transactions
            )

            support = count / total

            if support >= min_support:
                frequent[candidate] = support

    # 뇌졸중 발생 연관규칙 생성
    target = frozenset(["stroke=발생"])
    target_support = frequent.get(target, 0)

    rules = []

    for itemset, support in frequent.items():

        if len(itemset) < 2 or target not in itemset:
            continue

        antecedent = itemset - target

        antecedent_support = frequent.get(
            antecedent, 0
        )

        if antecedent_support == 0 or target_support == 0:
            continue

        confidence = support / antecedent_support
        lift = confidence / target_support

        rules.append({
            "조건": ", ".join(sorted(antecedent)),
            "조건 해당 인원": round(
                antecedent_support * total
            ),
            "뇌졸중 발생 인원": round(
                support * total
            ),
            "지지도(%)": support * 100,
            "신뢰도(%)": confidence * 100,
            "향상도": lift
        })

    result = pd.DataFrame(rules)

    if not result.empty:
        result = result.sort_values(
            ["향상도", "신뢰도(%)"],
            ascending=False
        ).reset_index(drop=True)

    return result


# --------------------------------------------------
# 화면 구성
# --------------------------------------------------
st.title(f"🩺 {APP_TITLE}")

st.caption(
    "건강 정보를 활용하여 뇌졸중 발생 관련 패턴을 분석합니다."
)

st.markdown("""
### 데이터 소개

이 데이터는 사람들의 기본 정보와 건강 관련 항목을 담고 있으며,
마지막의 `stroke` 열에는 뇌졸중 발생 여부가 기록되어 있습니다.

- 0: 뇌졸중 미발생
- 1: 뇌졸중 발생
""")

try:
    df = load_data()

except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------
saved_records = load_records()

total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people else 0
)

st.markdown("### 데이터 한눈에 보기")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 사람 수", f"{total_people:,}명")
c2.metric("열 개수", f"{column_count}개")
c3.metric("뇌졸중 발생자", f"{stroke_count:,}명")
c4.metric("발생 비율", f"{stroke_ratio:.2f}%")


# --------------------------------------------------
# 열 정보
# --------------------------------------------------
st.divider()
st.markdown("### 열 정보")

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        saved_records["meanings"].get(col, "")
        for col in df.columns
    ],
    "값의 종류": [
        int(df[col].nunique(dropna=True))
        for col in df.columns
    ],
    "빈 값 개수": [
        int(df[col].isna().sum())
        for col in df.columns
    ]
})

edited_info = st.data_editor(
    column_info,
    hide_index=True,
    use_container_width=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ]
)

current_meanings = {
    row["열 이름"]: row["우리말 뜻"]
    for _, row in edited_info.iterrows()
}


# --------------------------------------------------
# 데이터 미리보기
# --------------------------------------------------
st.divider()
st.markdown("### 데이터 미리보기")

st.dataframe(
    df.head(5),
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# 데이터 출처
# --------------------------------------------------
st.divider()
st.markdown("### 데이터 출처")

source_text = st.text_area(
    "데이터 출처",
    value=saved_records["source"],
    placeholder="교재에 적힌 데이터 출처를 입력하세요."
)

try:
    save_records(current_meanings, source_text)
    st.caption("입력 내용이 JSON 파일에 저장되었습니다.")

except OSError as error:
    st.error(f"기록 저장 오류: {error}")


# --------------------------------------------------
# 독립 변수 및 종속 변수 선정
# --------------------------------------------------
st.divider()
st.header("📊 독립 변수 및 종속 변수 선정")

st.markdown("### 1. 독립 변수")

independent_vars = {
    "age": "나이",
    "gender": "성별",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장질환 여부",
    "avg_glucose_level": "평균 혈당 수치",
    "bmi": "체질량지수(BMI)",
    "smoking_status": "흡연 상태"
}

st.write(
    "다음 7개 속성을 독립 변수로 선정하였다. "
    "이 속성들은 개인의 건강 상태와 생활 습관을 나타낸다."
)

variable_info = []

for col, meaning in independent_vars.items():
    variable_info.append({
        "속성": col,
        "우리말 뜻": meaning,
        "값의 종류": int(df[col].nunique()),
        "빈 값 개수": int(df[col].isna().sum()),
        "자료 형태": (
            "수치형"
            if pd.api.types.is_numeric_dtype(df[col])
            else "범주형"
        )
    })

st.dataframe(
    pd.DataFrame(variable_info),
    hide_index=True,
    use_container_width=True
)

st.markdown("""
**선정 이유**

나이, 평균 혈당 수치, BMI는 수치에 따른 차이를 비교할 수 있고,
성별, 고혈압, 심장질환, 흡연 상태는 건강 특성에 따른
뇌졸중 발생 패턴을 분석하기에 적합하기 때문이다.
""")

st.markdown("### 2. 종속 변수")

stroke_counts = df["stroke"].value_counts().sort_index()

st.metric(
    "종속 변수",
    "뇌졸중 발생 여부 (stroke)"
)

target_info = pd.DataFrame({
    "값": ["0", "1"],
    "의미": ["뇌졸중 미발생", "뇌졸중 발생"],
    "인원": [
        int(stroke_counts.get(0, 0)),
        int(stroke_counts.get(1, 0))
    ]
})

st.dataframe(
    target_info,
    hide_index=True,
    use_container_width=True
)

st.markdown("""
**선정 이유**

분석의 목적이 개인의 건강 정보와 뇌졸중 발생 여부 사이의
연관성을 알아보는 것이므로 stroke를 종속 변수로 선정하였다.
""")

st.caption(
    "※ ID는 개인 식별 번호이므로 분석 변수에서 제외하였다."
)


# --------------------------------------------------
# 연관분석 모델링
# --------------------------------------------------
st.divider()
st.header("🔎 연관분석 모델링")

st.write("""
Apriori 알고리즘을 사용하여 건강 특성들이 함께 나타나는 패턴과
뇌졸중 발생 사이의 연관성을 분석합니다.
""")

st.info(
    "연관분석은 변수 사이의 관계를 탐색하는 방법이며, "
    "인과관계나 개인의 실제 발병 가능성을 증명하지 않습니다."
)

support_percent = st.slider(
    "최소 지지도(%)",
    0.1, 5.0, 0.3, 0.1
)

confidence_percent = st.slider(
    "최소 신뢰도(%)",
    1, 100, 5
)

transactions = make_transactions(df)

rules_df = run_apriori(
    transactions,
    min_support=support_percent / 100,
    max_itemset_size=3
)


# --------------------------------------------------
# 연관분석 결과
# --------------------------------------------------
if not rules_df.empty:

    displayed_rules = rules_df[
        (rules_df["신뢰도(%)"] >= confidence_percent)
        & (rules_df["향상도"] > 1)
    ].copy()

    if not displayed_rules.empty:

        st.success(
            f"총 {len(displayed_rules)}개의 연관규칙을 발견했습니다."
        )

        st.markdown("### 연관분석 결과표")

        st.dataframe(
            displayed_rules.round(2),
            hide_index=True,
            use_container_width=True
        )

        # --------------------------------------------------
        # 시각화
        # --------------------------------------------------
        st.markdown("### 연관규칙 TOP 10")

        chart_data = displayed_rules.head(10).copy()

        chart_data["규칙"] = (
            chart_data["조건"].str.slice(0, 65)
        )

        fig = px.bar(
            chart_data,
            x="향상도",
            y="규칙",
            orientation="h",
            title="뇌졸중 발생 관련 연관규칙 TOP 10",
            hover_data=[
                "지지도(%)",
                "신뢰도(%)",
                "조건 해당 인원",
                "뇌졸중 발생 인원"
            ]
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="향상도",
            yaxis_title="건강 특성 조합"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # 최종 분석 결과
        # --------------------------------------------------
        st.divider()
        st.header("🏆 최종 분석 결과")

        best_rule = displayed_rules.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "조건 해당 인원",
            f"{int(best_rule['조건 해당 인원']):,}명"
        )

        c2.metric(
            "뇌졸중 발생 인원",
            f"{int(best_rule['뇌졸중 발생 인원']):,}명"
        )

        c3.metric(
            "뇌졸중 발생 비율",
            f"{best_rule['신뢰도(%)']:.2f}%"
        )

        st.markdown("### 가장 높은 연관성을 보인 조건")

        st.info(
            f"**{best_rule['조건']}** 조건에서 "
            f"가장 높은 향상도를 보였습니다."
        )

        st.markdown("### 최종 결론")

        st.write(
            f"분석 결과, {best_rule['조건']} 조건에 해당하는 "
            f"{int(best_rule['조건 해당 인원']):,}명 중 "
            f"{int(best_rule['뇌졸중 발생 인원']):,}명에게서 "
            f"뇌졸중이 발생했으며, 발생 비율은 "
            f"{best_rule['신뢰도(%)']:.2f}%로 나타났습니다. "
            f"이를 통해 해당 건강 특성과 뇌졸중 발생 사이에 "
            f"연관성이 있음을 확인했습니다."
        )

        st.caption(
            "※ 가장 높은 향상도를 기준으로 정리한 결과입니다. "
            "연관성이 원인이나 실제 발병 가능성을 증명하지는 않습니다."
        )

        # --------------------------------------------------
        # CSV 다운로드
        # --------------------------------------------------
        st.download_button(
            "연관분석 결과 CSV 다운로드",
            data=displayed_rules.to_csv(
                index=False,
                encoding="utf-8-sig"
            ),
            file_name="뇌졸중_연관분석_결과.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "조건에 맞는 규칙이 없습니다. "
            "최소 지지도 또는 신뢰도를 낮춰 보세요."
        )

else:
    st.warning(
        "연관규칙이 없습니다. 최소 지지도를 낮춰 보세요."
    )


# --------------------------------------------------
# 분석 지표 설명
# --------------------------------------------------
with st.expander("연관분석 지표 설명"):

    st.markdown("""
    - **지지도(Support):** 전체 데이터에서 조건과 결과가 함께 나타난 비율
    - **신뢰도(Confidence):** 특정 조건이 나타났을 때 결과도 나타나는 비율
    - **향상도(Lift):** 조건이 있을 때 결과가 나타나는 비율이 전체 발생 비율보다 얼마나 높은지 나타내는 값

    향상도가 1보다 크면 양의 연관성이 있음을 의미합니다.
    """)
