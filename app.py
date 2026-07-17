
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
import os

# --- 폰트 설정 (Colab 환경에서와 동일하게) ---
# Streamlit 환경에서는 폰트 파일이 없을 수 있으므로, 폰트 경로를 절대 경로로 지정하거나,
# 또는 Streamlit 앱 실행 시 폰트 파일을 함께 배포해야 합니다.
# 여기서는 NanumGothic이 설치되어 있다고 가정하고 진행합니다.
# 로컬 환경에서 실행 시 `NanumGothic.ttf` 파일이 앱이 실행되는 디렉토리나 시스템 폰트 경로에 있어야 합니다.
font_path = 'NanumGothic-Regular.ttf' # Streamlit 앱과 같은 경로에 폰트 파일을 둔다고 가정

# 폰트 파일 존재 여부 확인 및 추가
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
else:
    st.warning(f"Warning: Font file not found at {font_path}. Displaying with default font.")
    plt.rcParams["font.family"] = "sans-serif"

# --- 데이터 로드 및 전처리 (노트북의 최종 결과 사용) ---
BASE = "/content/drive/MyDrive/Colab Notebooks/숲과나눔 AI 공공데이터 분석/수정_무더위쉼터/"

@st.cache_data
def load_data():
    # 전처리된 동별 데이터 로드
    dong_valid = pd.read_csv(BASE + "서울쉼터_전처리결과.csv", encoding="utf-8-sig", dtype={"위치코드": str})

    # 경계 파일 로드
    boundary_path = BASE + "HangJeongDong_ver20260401.geojson"
    boundary = gpd.read_file(boundary_path)
    
    # 쉼터 데이터 로드 (접근성 계산을 위해)
    shelter = pd.read_csv(BASE + "서울시 무더위쉼터.csv", encoding="cp949")
    shelter = shelter[shelter["위치코드"].astype(str).str[-6:] != "000000"].copy()
    shelter["개방형"] = shelter["시설구분1"].isin(["공공시설", "생활밀착민간시설"])
    shelter["용량"] = pd.to_numeric(shelter["이용가능인원"], errors="coerce")
    shelter.loc[shelter["용량"] >= 9999, "용량"] = np.nan
    shelter["용량"] = shelter["용량"].fillna(shelter["용량"].median())

    shelter_gdf = gpd.GeoDataFrame(
        shelter,
        geometry=gpd.points_from_xy(shelter["경도"], shelter["위도"]),
        crs="EPSG:4326"
    ).to_crs("EPSG:5181")
    
    # map_df 재생성 (지도를 그리기 위해)
    seoul_bd = boundary[boundary["sido"].astype(str) == "11"].copy()
    seoul_bd["adm_cd2"] = seoul_bd["adm_cd2"].astype(str)
    dong_valid["위치코드"] = dong_valid["위치코드"].astype(str) # ensure consistent type
    map_df = seoul_bd.merge(
        dong_valid[["위치코드", "읍면동명", "취약도", "전체쉼터수", "개방형쉼터수"]],
        left_on="adm_cd2", right_on="위치코드", how="left"
    )
    
    # 접근성 계산을 위한 demand_xy, demand_pop 생성
    map_m = map_df.to_crs("EPSG:5181")
    map_m["cx"] = map_m.geometry.centroid.x
    map_m["cy"] = map_m.geometry.centroid.y
    valid = map_m.dropna(subset=["취약도"]).copy() # 이미 dong_valid는 취약도가 있음
    
    # '총인구' 데이터 로드 및 병합 (compute_2sfca_R 함수에 필요)
    df_pop_detail = pd.read_csv(BASE + "행정안전부_지역별(행정동) 성별 연령별 주민등록 인구수_20260630.csv", encoding='cp949')
    df_pop_detail["총인구"] = df_pop_detail["계"]
    pop_use = df_pop_detail[df_pop_detail["시도명"] == "서울특별시"][["행정기관코드", "총인구"]].copy()
    pop_use["행정기관코드"] = pop_use["행정기관코드"].astype(str)

    # valid DataFrame에 총인구 정보 추가
    if '총인구' not in valid.columns:
        valid = valid.merge(pop_use, left_on='adm_cd2', right_on='행정기관코드', how='left')
        valid = valid.drop(columns=['행정기관코드_y'])
        valid = valid.rename(columns={'행정기관코드_x': '행정기관코드'})

    # demand_xy와 demand_pop 재생성 (valid에 총인구가 들어간 후)
    valid_for_2sfca = valid.dropna(subset=["취약도", "총인구"]).copy()
    demand_xy = np.c_[valid_for_2sfca["cx"], valid_for_2sfca["cy"]]
    demand_pop = valid_for_2sfca["총인구"].values
    
    # compute_2sfca_R 함수 정의 (여기서 다시 정의하여 독립적으로 작동하게 함)
    def compute_2sfca_R(supply_gdf, demand_xy, demand_pop, radius):
        sxy = np.c_[supply_gdf.geometry.x, supply_gdf.geometry.y]
        cap = supply_gdf["용량"].values
        tree_d = cKDTree(demand_xy)
        tree_s = cKDTree(sxy)
        Rj = np.zeros(len(sxy))
        for j in range(len(sxy)):
            idx = tree_d.query_ball_point(sxy[j], radius)
            psum = demand_pop[idx].sum()
            if psum > 0:
                Rj[j] = cap[j] / psum
        Ai = np.zeros(len(demand_xy))
        for i in range(len(demand_xy)):
            idx = tree_s.query_ball_point(demand_xy[i], radius)
            Ai[i] = Rj[idx].sum()
        return Ai

    # 강건성 검정 결과 계산 (Colab 노트북과 동일한 로직)
    open_gdf = shelter_gdf[shelter_gdf["개방형"]].copy()
    radii = [500, 1000, 1500]
    results = []
    v_med = dong_valid["취약도"].median()

    for R in radii:
        acc = compute_2sfca_R(open_gdf, demand_xy, demand_pop, R)
        a_med = np.median(acc)
        blind_mask = (valid_for_2sfca["취약도"].values >= v_med) & (acc <= a_med)
        zero = (acc == 0).sum()
        results.append({
            "반경(m)": R,
            "사각지대_동수": int(blind_mask.sum()),
            "접근성0_동수": int(zero),
            "접근성_중앙값": round(a_med, 4)
        })
        valid_for_2sfca[f"사각지대_{R}m"] = blind_mask

    robust_df = pd.DataFrame(results)
    
    # 사각지대_정밀 컬럼 추가
    a_med_1000m = robust_df[robust_df['반경(m)'] == 1000]['접근성_중앙값'].iloc[0]
    valid_for_2sfca["사각지대_정밀"] = ((valid_for_2sfca["취약도"] >= v_med) &
                                   (valid_for_2sfca["접근성_개방형"] <= a_med_1000m))

    blind_top15 = pd.read_csv(BASE + "분석결과_사각지대명단.csv", encoding="utf-8-sig")

    return dong_valid, map_df, valid_for_2sfca, robust_df, blind_top15, shelter_gdf, demand_xy, demand_pop, open_gdf, v_med

# --- Streamlit 앱 시작 --- 
st.set_page_config(page_title="서울 폭염 쉼터 취약도 분석 대시보드", layout="wide")
st.title("☀️ 서울시 폭염 쉼터 취약도 분석 대시보드 ☀️")
st.markdown("이 대시보드는 서울시 행정동별 폭염 취약도와 쉼터 접근성을 분석하여, **폭염 취약 사각지대**를 도출하고 시각화합니다.")

dong_valid, map_df, valid, robust_df, blind_top15, shelter_gdf, demand_xy, demand_pop, open_gdf, v_med = load_data()

# --- 1. 주요 통계 요약 ---
st.header("1. 주요 분석 결과 요약")

cols = st.columns(3)
with cols[0]:
    st.metric("전체 행정동 수", f"{len(dong_valid)}개")
with cols[1]:
    st.metric("취약도 중앙값", f"{v_med:.2f}")
with cols[2]:
    st.metric("1km 반경 사각지대 동 수", f"{robust_df[robust_df['반경(m)'] == 1000]['사각지대_동수'].iloc[0]}개")

st.subheader("강건성 검정 결과 (반경별 사각지대 변화)")
st.dataframe(robust_df.set_index("반경(m)"))

# --- 2. 서울 행정동별 폭염 취약도 지도 ---
st.header("2. 서울 행정동별 폭염 취약도 지도")
st.markdown("배경색은 행정동의 **취약도**를 나타냅니다 (붉을수록 취약). 빗금 표시는 **사각지대**에 해당하는 동입니다 (취약↑ + 개방형 쉼터 접근성↓).")

fig_map, ax_map = plt.subplots(figsize=(13, 11))
map_df.plot(
    column="취약도", cmap="Reds", scheme="quantiles", k=6,
    linewidth=0.3, edgecolor="white", legend=True,
    legend_kwds={
        "loc": "lower left", "title": "취약도", "fontsize": 8,
        "bbox_to_anchor": (0.01, 0.05), # 범례 위치 조정
        "frameon": False
    },
    ax=ax_map
)
valid[valid["사각지대_정밀"]].plot(
    ax=ax_map, facecolor="none", edgecolor="black", linewidth=1.5, hatch="///"
)
blind_patch = mpatches.Patch(facecolor="none", edgecolor="black",
                             hatch="///", linewidth=1.5,
                             label="사각지대 (취약↑ + 개방형 접근성↓)")
ax_map.legend(handles=[blind_patch], loc="upper right", fontsize=10, frameon=False)
ax_map.set_title("서울 폭염 취약도 + 사각지대", fontsize=15)
ax_map.axis("off")
plt.tight_layout()
st.pyplot(fig_map)

# --- 3. 폭염쉼터 사각지대 동 — 취약도 상위 15개 ---
st.header("3. 폭염쉼터 사각지대 동 — 취약도 상위 15개")
st.markdown("취약도가 높은 순서대로 사각지대에 해당하는 상위 15개 행정동을 보여줍니다.")

fig_bar, ax_bar = plt.subplots(figsize=(12, 7))
ax_bar.bar(blind_top15["읍면동명"], blind_top15["취약도"], color="crimson", alpha=0.8)
ax_bar.set_title("폭염쉼터 사각지대 동 — 취약도 상위 15개", fontsize=15)
ax_bar.set_xlabel("행정동")
ax_bar.set_ylabel("취약도 (z-점수 기반)")
ax_bar.set_xticks(blind_top15["읍면동명"], rotation=45, ha="right")
ax_bar.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
st.pyplot(fig_bar)

# --- 4. 도보 반경별 사각지대 동 수 ---
st.header("4. 도보 반경별 사각지대 동 수")
st.markdown("도보 반경을 달리했을 때 사각지대 동의 수가 어떻게 변하는지 보여주는 그래프입니다. 반경이 넓어질수록 사각지대 수가 감소하는 경향을 보입니다.")

fig_robust, ax_robust = plt.subplots(figsize=(9, 6))
bars_robust = ax_robust.bar([f"{r}m" for r in robust_df["반경(m)"]],
                               robust_df["사각지대_동수"],
                               color=["lightcoral", "crimson", "darkred"])
for bar, val in zip(bars_robust, robust_df["사각지대_동수"]):
    ax_robust.text(bar.get_x() + bar.get_width()/2, val,
             f"{val}개", ha="center", va="bottom", fontsize=13)
ax_robust.set_title("도보 반경별 사각지대 동 수\n(반경이 넓어질수록 사각지대 감소)", fontsize=14)
ax_robust.set_xlabel("도보 반경")
ax_robust.set_ylabel("사각지대 동 수")
ax_robust.set_ylim(0, robust_df["사각지대_동수"].max()*1.2)
plt.tight_layout()
st.pyplot(fig_robust)

# --- 5. 사각지대 상세 목록 ---
st.header("5. 사각지대 상세 목록 (취약도 상위 15개 동)")
st.dataframe(blind_top15.set_index('읍면동명'))

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

csv_data = convert_df_to_csv(blind_top15)
st.download_button(
    label="사각지대 동 목록 CSV 다운로드",
    data=csv_data,
    file_name="사각지대_동_목록.csv",
    mime="text/csv",
)
