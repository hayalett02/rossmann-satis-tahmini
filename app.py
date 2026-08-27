"""
Rossmann Satış Tahmini — Streamlit panosu
softITo Veri Bilimi Bitirme Projesi
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Sayfa ayarı ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Rossmann Satış Tahmini",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent

# Proje metrikleri (Rosmann_Model.ipynb / rapor doğrulama sonuçları)
MODEL_RESULTS = pd.DataFrame(
    [
        {"Model": "Ensemble", "RMSE": 861.2, "MAE": 588.5, "R2": 0.9200, "RMSPE": 12.14},
        {"Model": "XGBoost Optuna", "RMSE": 869.3, "MAE": 595.7, "R2": 0.9184, "RMSPE": 12.26},
        {"Model": "XGBoost + Log", "RMSE": 873.9, "MAE": 597.9, "R2": 0.9176, "RMSPE": 12.32},
        {"Model": "LightGBM + Log", "RMSE": 895.6, "MAE": 614.3, "R2": 0.9134, "RMSPE": 12.51},
        {"Model": "XGBoost Baseline", "RMSE": 960.1, "MAE": 681.9, "R2": 0.9005, "RMSPE": 14.91},
        {"Model": "XGBoost Lag (rekursif)", "RMSE": 1018.5, "MAE": 723.1, "R2": 0.8880, "RMSPE": 15.38},
    ]
)
ENSEMBLE_WEIGHTS = pd.DataFrame(
    [
        {"Model": "XGBoost Optuna", "Ağırlık": 0.50},
        {"Model": "XGBoost + Log", "Ağırlık": 0.31},
        {"Model": "LightGBM", "Ağırlık": 0.19},
        {"Model": "XGBoost Lag", "Ağırlık": 0.00},
    ]
)
CV_FOLDS = pd.DataFrame(
    [
        {"Kat": "Kat 1", "Başlangıç": "2015-03-27", "Bitiş": "2015-05-08", "RMSPE": 13.07},
        {"Kat": "Kat 2", "Başlangıç": "2015-05-08", "Bitiş": "2015-06-19", "RMSPE": 12.08},
        {"Kat": "Kat 3", "Başlangıç": "2015-06-19", "Bitiş": "2015-07-31", "RMSPE": 12.65},
    ]
)

# ── Stil ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f7f3ee 0%, #eef2f6 40%, #f7f3ee 100%);
        color: #1a2332;
    }

    /* Ana içerik: açık zeminde koyu yazı (beyaz yazıyı zorla kapat) */
    [data-testid="stAppViewContainer"] ,
    [data-testid="stMain"],
    .main,
    .block-container {
        color: #1a2332 !important;
    }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] p,
    [data-testid="stMain"] li,
    [data-testid="stMain"] span,
    [data-testid="stMain"] label,
    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] [data-testid="stCaption"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stMetricLabel"],
    [data-testid="stMain"] [data-testid="stMetricValue"],
    [data-testid="stMain"] [data-testid="stMetricDelta"] {
        color: #1a2332 !important;
    }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        font-family: 'Fraunces', Georgia, serif !important;
    }

    /* Widget / info kutuları */
    [data-testid="stMain"] .stSelectbox label,
    [data-testid="stMain"] .stMultiSelect label,
    [data-testid="stMain"] .stSlider label,
    [data-testid="stMain"] .stCheckbox label,
    [data-testid="stMain"] [data-baseweb="select"] *,
    [data-testid="stMain"] [data-baseweb="input"] * {
        color: #1a2332 !important;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {
        color: #1a2332 !important;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e4ddd3;
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a2332 !important;
    }

    /* Sidebar: koyu zemin, açık yazı */
    [data-testid="stSidebar"] {
        background: #1a2332;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] [data-testid="stCaption"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #f0ebe3 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.35rem 0;
        color: #f0ebe3 !important;
    }

    .hero {
        background: linear-gradient(135deg, #c41e3a 0%, #8b1528 55%, #1a2332 100%);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        color: #fff !important;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(26, 35, 50, 0.18);
    }
    .hero h1 {
        color: #fff !important;
        margin: 0 0 0.4rem 0;
        font-size: 2rem;
        font-family: 'Fraunces', Georgia, serif !important;
    }
    .hero p {
        margin: 0;
        opacity: 0.95;
        font-size: 1.02rem;
        color: #fff !important;
    }
    .kpi {
        background: #fff;
        border: 1px solid #e4ddd3;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .kpi .label {
        font-size: 0.82rem;
        color: #4b5563 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi .value {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.7rem;
        color: #c41e3a !important;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .note {
        background: #fff8e7;
        border-left: 4px solid #c41e3a;
        padding: 0.85rem 1rem;
        border-radius: 0 10px 10px 0;
        color: #3f3a32 !important;
        margin: 0.8rem 0 1.2rem 0;
    }
    .note b, .note code {
        color: #1a2332 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Veri yükleme ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Veriler yükleniyor…")
def load_train() -> pd.DataFrame:
    path = BASE / "train_final.parquet"
    cols = [
        "Store", "Date", "Sales", "Customers", "Open", "Promo",
        "DayOfWeek", "StoreType", "StateHoliday", "SchoolHoliday",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_val_predictions() -> pd.DataFrame:
    path = BASE / "val_predictions.parquet"
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def load_store_meta() -> pd.DataFrame:
    return pd.read_csv(BASE / "store.csv")


def rmspe_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100)


@st.cache_data(show_spinner=False)
def open_sales(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()


def kpi_card(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>'


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#1a2332"),
    margin=dict(l=40, r=20, t=50, b=40),
)


# ── Sayfalar ──────────────────────────────────────────────────────
def page_overview(train: pd.DataFrame, val_preds: pd.DataFrame):
    st.markdown(
        """
        <div class="hero">
          <h1>Rossmann Mağaza Satışları Tahmini</h1>
          <p>Sızıntısız zaman serisi modelleme · XGBoost + LightGBM + Optuna ensemble · softITo bitirme projesi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    acik = val_preds[(val_preds["Open"] == 1) & (val_preds["Sales"] > 0)]
    val_rmspe = rmspe_score(acik["Sales"], acik["Tahmin"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Doğrulama RMSPE", f"%{val_rmspe:.2f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Mağaza sayısı", f"{train['Store'].nunique():,}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Eğitim satırı", f"{len(train):,}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Val. satırı", f"{len(val_preds):,}"), unsafe_allow_html=True)

    st.markdown("")
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Proje özeti")
        st.markdown(
            """
            Bu pano, Rossmann zincirinin günlük satışlarını **6 haftalık** ufukta
            tahmin eden bitirme çalışmasının sonuçlarını gösterir.

            - Kronolojik / **kilitli** doğrulama (test seti model seçiminde kullanılmadı)
            - Hedefte **log1p** dönüşümü
            - Kapalı mağaza günlerine kural bazlı **0** atama
            - Optuna ile hiperparametre araması + ağırlıklı **ensemble**
            """
        )
        st.markdown(
            '<div class="note"><b>Tahmin sayfası:</b> Test döneminde gerçek satış bilinmediği için '
            "karşılaştırma <b>doğrulama penceresinde</b> (2015-06-19 → 2015-07-31) yapılır — "
            "gerçek satış ile model tahmini yan yana.</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Dönemler")
        timeline = pd.DataFrame(
            [
                {"Aşama": "Eğitim", "Başlangıç": "2013-01-01", "Bitiş": "2015-06-18"},
                {"Aşama": "Doğrulama", "Başlangıç": "2015-06-19", "Bitiş": "2015-07-31"},
                {"Aşama": "Test (kilitli)", "Başlangıç": "2015-08-01", "Bitiş": "2015-09-17"},
            ]
        )
        fig = go.Figure()
        colors = {"Eğitim": "#1a2332", "Doğrulama": "#d97706", "Test (kilitli)": "#c41e3a"}
        for _, row in timeline.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[(pd.Timestamp(row["Bitiş"]) - pd.Timestamp(row["Başlangıç"])).days],
                    y=[row["Aşama"]],
                    base=[pd.Timestamp(row["Başlangıç"])],
                    orientation="h",
                    marker_color=colors[row["Aşama"]],
                    hovertemplate=(
                        f"{row['Aşama']}<br>{row['Başlangıç']} → {row['Bitiş']}<extra></extra>"
                    ),
                    name=row["Aşama"],
                )
            )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="overlay",
            showlegend=False,
            height=260,
            xaxis_title="Tarih",
            yaxis_title="",
            xaxis_type="date",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_eda(train: pd.DataFrame):
    st.header("Keşifçi veri analizi")
    df = open_sales(train)

    f1, f2, f3 = st.columns(3)
    with f1:
        store_types = st.multiselect(
            "Mağaza tipi",
            sorted(df["StoreType"].dropna().unique()),
            default=sorted(df["StoreType"].dropna().unique()),
        )
    with f2:
        promo_filter = st.selectbox("Promosyon", ["Tümü", "Promo açık", "Promo kapalı"])
    with f3:
        sample_n = st.slider("Histogram örneklem (hız için)", 20_000, 200_000, 80_000, 10_000)

    view = df[df["StoreType"].isin(store_types)]
    if promo_filter == "Promo açık":
        view = view[view["Promo"] == 1]
    elif promo_filter == "Promo kapalı":
        view = view[view["Promo"] == 0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ortalama satış", f"{view['Sales'].mean():,.0f} €")
    m2.metric("Medyan satış", f"{view['Sales'].median():,.0f} €")
    m3.metric("Çarpıklık (ham)", f"{view['Sales'].skew():.2f}")
    m4.metric("Çarpıklık (log1p)", f"{np.log1p(view['Sales']).skew():.2f}")

    c1, c2 = st.columns(2)
    with c1:
        sample = view.sample(min(sample_n, len(view)), random_state=42)
        fig = px.histogram(
            sample,
            x="Sales",
            nbins=50,
            title="Günlük satış dağılımı (açık mağazalar)",
            color_discrete_sequence=["#c41e3a"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sample = sample.assign(log_sales=np.log1p(sample["Sales"]))
        fig = px.histogram(
            sample,
            x="log_sales",
            nbins=50,
            title="log1p(Sales) dağılımı",
            color_discrete_sequence=["#0f766e"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_title="log1p(Sales)")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        promo_stats = (
            view.groupby("Promo", as_index=False)["Sales"]
            .mean()
            .assign(Promo=lambda x: x["Promo"].map({0: "Promo yok", 1: "Promo var"}))
        )
        fig = px.bar(
            promo_stats,
            x="Promo",
            y="Sales",
            title="Promosyon etkisi (ortalama satış)",
            color="Promo",
            color_discrete_sequence=["#64748b", "#c41e3a"],
            text_auto=".0f",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=360, showlegend=False, yaxis_title="Ortalama €")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        dow = (
            view.groupby("DayOfWeek", as_index=False)["Sales"]
            .mean()
            .sort_values("DayOfWeek")
        )
        dow["Gün"] = dow["DayOfWeek"].map(
            {1: "Pzt", 2: "Sal", 3: "Çar", 4: "Per", 5: "Cum", 6: "Cmt", 7: "Paz"}
        )
        fig = px.bar(
            dow,
            x="Gün",
            y="Sales",
            title="Haftanın gününe göre ortalama satış",
            color_discrete_sequence=["#1a2332"],
            text_auto=".0f",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=360, yaxis_title="Ortalama €")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Aylık ortalama satış (mevsimsellik)")
    monthly = (
        view.assign(Ay=view["Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("Ay", as_index=False)["Sales"]
        .mean()
    )
    fig = px.line(
        monthly,
        x="Ay",
        y="Sales",
        title="Zaman içinde ortalama günlük satış",
        color_discrete_sequence=["#c41e3a"],
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="Ortalama €")
    st.plotly_chart(fig, use_container_width=True)


def page_models():
    st.header("Model performansı")
    st.caption("Doğrulama penceresi: 2015-06-19 → 2015-07-31 (resmi metrik: RMSPE)")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = px.bar(
            MODEL_RESULTS.sort_values("RMSPE"),
            x="RMSPE",
            y="Model",
            orientation="h",
            title="Model karşılaştırması — RMSPE (%)",
            color="RMSPE",
            color_continuous_scale=["#0f766e", "#d97706", "#c41e3a"],
            text="RMSPE",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Ensemble ağırlıkları")
        fig = px.pie(
            ENSEMBLE_WEIGHTS[ENSEMBLE_WEIGHTS["Ağırlık"] > 0],
            values="Ağırlık",
            names="Model",
            hole=0.45,
            color_discrete_sequence=["#c41e3a", "#1a2332", "#0f766e"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            ENSEMBLE_WEIGHTS.style.format({"Ağırlık": "{:.0%}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Detaylı metrik tablosu")
    st.dataframe(
        MODEL_RESULTS.style.format(
            {"RMSE": "{:.1f}", "MAE": "{:.1f}", "R2": "{:.4f}", "RMSPE": "{:.2f}%"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Walk-forward çapraz doğrulama")
    c3, c4 = st.columns([1, 1.2])
    with c3:
        st.dataframe(CV_FOLDS, use_container_width=True, hide_index=True)
        st.metric("CV ortalama RMSPE", "%12,60")
        st.metric("CV std", "0,40")
    with c4:
        fig = px.bar(
            CV_FOLDS,
            x="Kat",
            y="RMSPE",
            title="Kat bazlı RMSPE",
            color="RMSPE",
            color_continuous_scale=["#0f766e", "#c41e3a"],
            text="RMSPE",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


MODEL_OPTIONS = {
    "Ensemble": "Tahmin_Ensemble",
    "XGBoost Optuna": "Tahmin_XGBoost_Optuna",
    "XGBoost + Log": "Tahmin_XGBoost_Log",
    "LightGBM + Log": "Tahmin_LightGBM",
    "XGBoost Baseline": "Tahmin_XGBoost_Baseline",
}


def page_forecasts(train: pd.DataFrame, val_preds: pd.DataFrame, stores: pd.DataFrame):
    st.header("Mağaza tahminleri (doğrulama)")
    st.caption(
        "Test döneminde gerçek satış bilinmediği için karşılaştırma doğrulama penceresinde yapılır: "
        "2015-06-19 → 2015-07-31 · Modeller arası geçiş yapabilirsiniz"
    )

    # Eksik kolonlar için geriye dönük uyumluluk
    available = {k: v for k, v in MODEL_OPTIONS.items() if v in val_preds.columns}
    if not available:
        if "Tahmin" in val_preds.columns:
            available = {"XGBoost + Log": "Tahmin"}
        else:
            st.error("Tahmin kolonları bulunamadı. `prepare_val_predictions.py` çalıştırın.")
            return

    store_ids = sorted(val_preds["Store"].unique())
    meta = stores.set_index("Store")

    f1, f2, f3, f4 = st.columns([1, 1.2, 1, 1.3])
    with f1:
        store_id = st.selectbox("Mağaza", store_ids, index=0)
    with f2:
        model_name = st.selectbox(
            "Model",
            list(available.keys()),
            index=0,
            help="Doğrulama tahminini hangi modelden göstereceğini seçin",
        )
    with f3:
        only_open = st.checkbox("Yalnızca açık günler", value=False)
    with f4:
        st.write("")
        if store_id in meta.index:
            row = meta.loc[store_id]
            st.info(
                f"Tip **{row['StoreType']}** · Assortment **{row['Assortment']}** · "
                f"Rakip mesafe **{row['CompetitionDistance']:,.0f}**"
                if pd.notna(row["CompetitionDistance"])
                else f"Tip **{row['StoreType']}** · Assortment **{row['Assortment']}**"
            )

    pred_col = available[model_name]
    # Seçilen modeli tek bir Tahmin kolonuna yaz (kopya — cache'i bozmamak için)
    view = val_preds.copy()
    view["Tahmin"] = view[pred_col]

    hist = train[train["Store"] == store_id].sort_values("Date")
    val = view[view["Store"] == store_id].sort_values("Date")
    if only_open:
        hist = hist[hist["Open"] == 1]
        val = val[val["Open"] == 1]

    # Doğrulama öncesi ~60 gün bağlam + val dönemi
    val_start = pd.Timestamp("2015-06-19")
    hist_tail = hist[(hist["Date"] >= val_start - pd.Timedelta(days=60)) & (hist["Date"] < val_start)]

    fig = go.Figure()
    if len(hist_tail):
        fig.add_trace(
            go.Scatter(
                x=hist_tail["Date"],
                y=hist_tail["Sales"],
                mode="lines",
                name="Gerçek (val öncesi)",
                line=dict(color="#94a3b8", width=2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=val["Date"],
            y=val["Sales"],
            mode="lines+markers",
            name="Gerçek satış (val)",
            line=dict(color="#1a2332", width=2.5),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=val["Date"],
            y=val["Tahmin"],
            mode="lines+markers",
            name=f"{model_name} tahmini",
            line=dict(color="#c41e3a", width=2.5, dash="dot"),
            marker=dict(size=5),
        )
    )
    fig.add_vrect(
        x0="2015-06-19",
        x1="2015-07-31",
        fillcolor="rgba(217, 119, 6, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Doğrulama",
        annotation_position="top left",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=440,
        title=f"Mağaza {store_id}: gerçek vs {model_name}",
        yaxis_title="Satış (€)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    store_val_all = view[view["Store"] == store_id]
    store_open = store_val_all[(store_val_all["Open"] == 1) & (store_val_all["Sales"] > 0)]
    store_rmspe = rmspe_score(store_open["Sales"], store_open["Tahmin"])
    all_open = view[(view["Open"] == 1) & (view["Sales"] > 0)]
    global_rmspe = rmspe_score(all_open["Sales"], all_open["Tahmin"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mağaza RMSPE", f"%{store_rmspe:.2f}" if pd.notna(store_rmspe) else "—")
    k2.metric(f"{model_name} val RMSPE", f"%{global_rmspe:.2f}")
    k3.metric("Ort. gerçek", f"{val['Sales'].mean():,.0f} €")
    k4.metric("Ort. tahmin", f"{val['Tahmin'].mean():,.0f} €")

    # Tüm modellerin genel RMSPE özeti
    if len(available) > 1:
        rows = []
        open_mask = (view["Open"] == 1) & (view["Sales"] > 0)
        for name, col in available.items():
            rows.append(
                {
                    "Model": name,
                    "RMSPE %": round(rmspe_score(view.loc[open_mask, "Sales"], view.loc[open_mask, col]), 2),
                }
            )
        cmp = pd.DataFrame(rows).sort_values("RMSPE %")
        st.dataframe(cmp, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        scatter_df = store_open.copy()
        fig = px.scatter(
            scatter_df,
            x="Sales",
            y="Tahmin",
            title=f"Mağaza {store_id}: gerçek × {model_name}",
            color_discrete_sequence=["#c41e3a"],
            opacity=0.75,
        )
        mx = max(scatter_df["Sales"].max(), scatter_df["Tahmin"].max()) if len(scatter_df) else 1
        fig.add_trace(
            go.Scatter(
                x=[0, mx],
                y=[0, mx],
                mode="lines",
                name="y = x",
                line=dict(color="#64748b", dash="dash", width=1),
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        daily = (
            view.groupby("Date", as_index=False)
            .agg(Gerçek=("Sales", "sum"), Tahmin=("Tahmin", "sum"))
            .sort_values("Date")
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["Gerçek"],
                mode="lines",
                name="Gerçek toplam",
                line=dict(color="#1a2332", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["Date"],
                y=daily["Tahmin"],
                mode="lines",
                name=f"{model_name} toplam",
                line=dict(color="#c41e3a", width=2, dash="dot"),
            )
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            title="Tüm mağazalar — günlük toplam (val)",
            yaxis_title="Toplam €",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Doğrulama tablosu")
    show = val.copy()
    show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
    show["Sales"] = show["Sales"].round(2)
    show["Tahmin"] = show["Tahmin"].round(2)
    show["Hata"] = (show["Tahmin"] - show["Sales"]).round(2)
    cols = [c for c in ["Date", "DayOfWeek", "Open", "Promo", "Sales", "Tahmin", "Hata"] if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, hide_index=True, height=320)

    st.download_button(
        f"Bu mağazanın {model_name} val tahminlerini CSV indir",
        data=show[cols].to_csv(index=False).encode("utf-8"),
        file_name=f"magaza_{store_id}_{model_name.replace(' ', '_')}_val.csv",
        mime="text/csv",
    )


def page_method():
    st.header("Metodoloji")
    st.markdown(
        """
        ### Neden rastgele split yok?
        İlk denemede veri `Store + Date` sırasıyla satır bazında bölünmüştü.
        Eğitim ve doğrulama **takvimsel olarak çakışıyordu** — skor iyimser görünüyordu,
        ama geleceği tahmin etme görevini yansıtmıyordu.

        ### Ne yaptık?
        1. **Kilitli takvim doğrulaması** — son 6 hafta val, önceki tüm geçmiş train  
        2. **log1p hedef** — çarpıklık 1,59 → −0,11; RMSPE ile uyum  
        3. **Kapalı gün = 0** — modele gürültü olarak verilmedi  
        4. **Walk-forward CV** — 3 kat, ortalama %12,60 RMSPE  
        5. **Optuna** — XGBoost hiperparametre araması  
        6. **Ensemble** — ağırlıklı birleşim → **%12,14 RMSPE**  
        7. **Rekursif lag** — testte gerçek geçmiş yok; gün gün geri besleme  

        ### Veri hattı
        `RosmannVeriYukleme.ipynb` → `RosmannEDA.ipynb` → `Rosmann_Model.ipynb` → `submission.csv`
        """
    )

    st.subheader("Walk-forward pencereleri")
    rows = [
        dict(Task="Kat 1 val", Start="2015-03-27", Finish="2015-05-08", Resource="Val"),
        dict(Task="Kat 2 val", Start="2015-05-08", Finish="2015-06-19", Resource="Val"),
        dict(Task="Kat 3 val", Start="2015-06-19", Finish="2015-07-31", Resource="Val"),
        dict(Task="Test", Start="2015-08-01", Finish="2015-09-17", Resource="Test"),
    ]
    gdf = pd.DataFrame(rows)
    fig = px.timeline(
        gdf,
        x_start="Start",
        x_end="Finish",
        y="Task",
        color="Resource",
        color_discrete_map={"Val": "#d97706", "Test": "#c41e3a"},
        title="Doğrulama katları ve kilitli test",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=320)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


# ── Ana ───────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("### Rossmann Forecast")
        st.caption("softITo · Veri Bilimi")
        page = st.radio(
            "Sayfa",
            [
                "Genel bakış",
                "Keşifçi analiz",
                "Model performansı",
                "Mağaza tahminleri",
                "Metodoloji",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Val karşılaştırması · RMSPE ~%12,3")
        st.caption("Dönem: 2015-06-19 → 2015-07-31")

    try:
        train = load_train()
        val_preds = load_val_predictions()
        stores = load_store_meta()
    except FileNotFoundError as exc:
        st.error(
            f"Gerekli veri dosyası bulunamadı: `{exc}`\n\n"
            "`train_final.parquet`, `val_predictions.parquet` ve `store.csv` "
            "proje klasöründe olmalı. Val tahminleri için: "
            "`py -3 prepare_val_predictions.py`"
        )
        st.stop()

    if page == "Genel bakış":
        page_overview(train, val_preds)
    elif page == "Keşifçi analiz":
        page_eda(train)
    elif page == "Model performansı":
        page_models()
    elif page == "Mağaza tahminleri":
        page_forecasts(train, val_preds, stores)
    else:
        page_method()


if __name__ == "__main__":
    main()
