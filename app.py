"""
OLIST — Hệ thống Đề xuất · Dashboard Quản Trị
Chạy: streamlit run app.py
Yêu cầu: pip install streamlit pandas numpy scikit-learn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# CẤU HÌNH TRANG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Olist · Dashboard Quản Trị",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Màu Plotly thống nhất
PLOTLY_COLORS = px.colors.qualitative.Set2
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

# ══════════════════════════════════════════════════════════
# ĐỊNH NGHĨA CLASS (cần để pickle hoạt động)
# ══════════════════════════════════════════════════════════
class PopularityRecommender:
    def __init__(self, top_n=10, min_count=2):
        self.top_n = top_n
        self.min_count = min_count
        self._by_cat_state = {}
        self._by_cat = {}
        self._global = []

    def recommend(self, user_id, train_df, top_n=None):
        top_n = top_n or self.top_n
        h = train_df[train_df["customer_unique_id"] == user_id]
        bought = set(h["product_id"])
        if h.empty:
            return [p for p in self._global if p not in bought][:top_n]
        c = h["product_category_name_english"].value_counts().index[0]
        s = h["customer_state"].value_counts().index[0]
        pool = (
            self._by_cat_state.get((c, s))
            or self._by_cat.get(c)
            or self._global
        )
        recs = [p for p in pool if p not in bought]
        if len(recs) < top_n:
            recs += [p for p in self._global if p not in bought and p not in recs]
        return recs[:top_n]


class ContentBasedRecommender:
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.sim_matrix = None
        self.pid_to_idx = {}
        self.idx_to_pid = {}

    def recommend(self, user_id, train_df, top_n=None):
        top_n = top_n or self.top_n
        h = train_df[train_df["customer_unique_id"] == user_id]
        if h.empty:
            return []
        bought = set(h["product_id"])
        sv = np.zeros(self.sim_matrix.shape[0])
        tw = 0.0
        for _, row in h.iterrows():
            p = row["product_id"]
            if p not in self.pid_to_idx:
                continue
            w = max((row["review_score"] - 1) / 4, 0.05)
            sv += w * self.sim_matrix[self.pid_to_idx[p]]
            tw += w
        if tw == 0:
            return []
        sv /= tw
        for p in bought:
            if p in self.pid_to_idx:
                sv[self.pid_to_idx[p]] = -1.0
        recs = []
        for i in np.argsort(sv)[::-1]:
            p = self.idx_to_pid[i]
            if p not in bought:
                recs.append(p)
            if len(recs) >= top_n:
                break
        return recs


class MatrixFactorizationSGD:
    def __init__(self, **kw):
        self.n_factors = 20
        self.mu = 0.0
        self.bu = {}
        self.bi = {}
        self.P = {}
        self.Q = {}
        self.users = []
        self._trained = False

    def predict(self, u, i):
        if not self._trained:
            return self.mu
        return float(
            np.clip(
                self.mu
                + self.bu.get(u, 0)
                + self.bi.get(i, 0)
                + np.dot(
                    self.P.get(u, np.zeros(self.n_factors)),
                    self.Q.get(i, np.zeros(self.n_factors)),
                ),
                1,
                5,
            )
        )


class CollaborativeFilteringRecommender:
    def __init__(self, **kw):
        self.top_n = 10
        self.mf = MatrixFactorizationSGD()
        self.candidate_items = []

    def recommend(self, user_id, train_df, top_n=None):
        top_n = top_n or self.top_n
        if user_id not in self.mf.users:
            return []
        bought = set(
            train_df[train_df["customer_unique_id"] == user_id]["product_id"]
        )
        preds = [
            (p, self.mf.predict(user_id, p))
            for p in self.candidate_items
            if p not in bought
        ]
        preds.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in preds[:top_n]]


# ══════════════════════════════════════════════════════════
# LOAD DỮ LIỆU & MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Đang tải dữ liệu và model...")
def load_all():
    train = pd.read_csv("train_df.csv")
    seg_df = pd.read_csv("user_segment.csv")

    train["review_score"] = train["review_score"].fillna(
        train["review_score"].median()
    )
    train["product_category_name_english"] = train[
        "product_category_name_english"
    ].fillna("unknown")
    train["order_purchase_timestamp"] = pd.to_datetime(
        train["order_purchase_timestamp"]
    )
    train["month"] = train["order_purchase_timestamp"].dt.to_period("M")
    train["price_bucket"] = pd.qcut(
        train["price"],
        q=4,
        labels=["budget", "mid", "premium", "luxury"],
        duplicates="drop",
    ).astype(str)

    with open("pop_model.pkl", "rb") as f:
        pop = pickle.load(f)
    with open("cb_model.pkl", "rb") as f:
        cb = pickle.load(f)
    with open("cf_model.pkl", "rb") as f:
        cf = pickle.load(f)
    with open("seg_dict.pkl", "rb") as f:
        sd = pickle.load(f)

    # Product lookup
    prod_info = (
        train.groupby("product_id")
        .agg(
            category=("product_category_name_english", "first"),
            avg_price=("price", "mean"),
            avg_freight=("freight_value", "mean"),
            avg_rating=("review_score", "mean"),
            n_sold=("order_id", "count"),
            revenue=("payment_value", "sum"),
        )
        .reset_index()
    )

    # Tính sẵn các bảng thống kê
    monthly = (
        train.groupby("month")
        .agg(revenue=("payment_value", "sum"), orders=("order_id", "nunique"))
        .reset_index()
    )
    monthly["month_str"] = monthly["month"].astype(str)
    monthly["month_label"] = monthly["month_str"].apply(
        lambda x: "T" + str(int(x.split("-")[1]))
    )

    cat_stats = (
        train.groupby("product_category_name_english")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("order_id", "count"),
            avg_rating=("review_score", "mean"),
            avg_price=("price", "mean"),
            n_products=("product_id", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .rename(columns={"product_category_name_english": "category"})
    )

    state_stats = (
        train.groupby("customer_state")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("order_id", "count"),
            users=("customer_unique_id", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    seg_merged = train.merge(seg_df, on="customer_unique_id", how="left")
    seg_stats = (
        seg_merged.groupby("segment")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("order_id", "count"),
            avg_spend=("payment_value", "mean"),
            avg_rating=("review_score", "mean"),
            users=("customer_unique_id", "nunique"),
        )
        .reset_index()
    )

    return train, seg_df, pop, cb, cf, sd, prod_info, monthly, cat_stats, state_stats, seg_stats


(
    train_df, seg_df, pop_model, cb_model, cf_model,
    seg_dict, prod_info, monthly, cat_stats, state_stats, seg_stats,
) = load_all()

prod_lookup = prod_info.set_index("product_id").to_dict("index")

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
CAT_ICONS = {
    "health_beauty": "💄", "watches_gifts": "⌚", "bed_bath_table": "🛏️",
    "sports_leisure": "⚽", "computers_accessories": "💻", "furniture_decor": "🪑",
    "cool_stuff": "🎯", "housewares": "🏠", "auto": "🚗", "toys": "🧸",
    "garden_tools": "🌱", "baby": "🍼", "perfumery": "🌸", "telephony": "📱",
    "stationery": "📎", "fashion_bags_accessories": "👜", "pet_shop": "🐾",
    "electronics": "🔌", "computers": "🖥️", "furniture_bedroom": "🛋️",
    "stationery": "📌",
}

SEG_LABELS = {
    "one_time": "Khách vãng lai",
    "occasional": "Khách thỉnh thoảng",
    "regular": "Khách thân thiết",
    "vip": "VIP",
}


def get_icon(cat):
    for k, v in CAT_ICONS.items():
        if k in str(cat):
            return v
    return "📦"


def fmt_brl(v):
    return f"R$ {v:,.0f}"


def hybrid_recommend(user_id, top_n=10):
    seg = seg_dict.get(user_id, "one_time")
    recs, strategy = [], ""

    if seg == "one_time":
        recs = pop_model.recommend(user_id, train_df, top_n)
        strategy = "Popularity-Based"
    elif seg == "occasional":
        recs = cb_model.recommend(user_id, train_df, top_n)
        strategy = "Content-Based"
        if len(recs) < top_n:
            bought = set(train_df[train_df["customer_unique_id"] == user_id]["product_id"])
            extra = [p for p in pop_model.recommend(user_id, train_df, top_n)
                     if p not in set(recs) and p not in bought]
            recs = (recs + extra)[:top_n]
            strategy = "Hybrid (CB + Popularity)"
    else:
        recs = cf_model.recommend(user_id, train_df, top_n)
        strategy = "Matrix Factorization"
        if len(recs) < top_n:
            bought = set(train_df[train_df["customer_unique_id"] == user_id]["product_id"])
            extra = [p for p in cb_model.recommend(user_id, train_df, top_n)
                     if p not in set(recs) and p not in bought]
            recs = (recs + extra)[:top_n]
            strategy = "Hybrid (MF + CB)"

    if len(recs) < top_n:
        bought = set(train_df[train_df["customer_unique_id"] == user_id]["product_id"])
        extra = [p for p in pop_model._global if p not in set(recs) and p not in bought]
        recs = (recs + extra)[:top_n]

    return recs, strategy, seg


# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Amazon-style enhancements
# ══════════════════════════════════════════════════════════

def get_product_badges(pid, prod_lookup):
    """Badge bán chạy / top rated / ship rẻ / ship đắt."""
    info      = prod_lookup.get(pid, {})
    n_sold    = info.get("n_sold",    0)
    avg_rat   = info.get("avg_rating",0)
    avg_price = info.get("avg_price", 0)
    avg_ship  = info.get("avg_freight",0)

    badges = []
    # Ngưỡng top 10% lượt bán — tính 1 lần, cứng
    P90_SOLD = 15

    if n_sold >= P90_SOLD:
        badges.append("🔥 Bán chạy")
    if avg_rat >= 4.5 and n_sold >= 5:
        badges.append("⭐ Top rated")
    if avg_price > 0 and avg_ship / avg_price < 0.10:
        badges.append("🚚 Ship rẻ")
    elif avg_price > 0 and avg_ship / avg_price > 0.50:
        badges.append("⚠️ Ship đắt")
    return badges


def get_seasonal_boost(month: int) -> dict:
    """Seasonal multiplier per category cho từng tháng."""
    seasonal = {
        11: {"toys":2.5,"bed_bath_table":2.0,"garden_tools":1.8,
             "sports_leisure":1.7,"health_beauty":1.6},
        12: {"toys":2.0,"watches_gifts":2.0,"books_general_interest":1.5},
        1:  {"furniture_decor":1.3,"housewares":1.2},
        5:  {"sports_leisure":1.5,"garden_tools":1.4},
    }
    return seasonal.get(month, {})


def seasonal_recommend(user_id, top_n=10):
    """Popularity + seasonal boost cho one_time users."""
    from datetime import datetime
    month_now = datetime.now().month
    boost     = get_seasonal_boost(month_now)

    recs = pop_model.recommend(user_id, train_df, top_n * 2)
    if not boost or not recs:
        return recs[:top_n]

    prod_cat = train_df.set_index("product_id")[
        "product_category_name_english"
    ].to_dict()

    scored = []
    for i, pid in enumerate(recs):
        cat        = prod_cat.get(pid, "unknown")
        base_score = 1.0 / (i + 1)
        w          = boost.get(cat, 1.0)
        scored.append((pid, base_score * w))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored[:top_n]]


def upsell_recommend(user_id, top_n=3):
    """Gợi ý sản phẩm cao cấp hơn trong cùng category."""
    BUCKET_ORDER = ["budget", "mid", "premium", "luxury"]

    user_hist = train_df[train_df["customer_unique_id"] == user_id]
    if user_hist.empty:
        return []

    # Tính price_bucket cho prod_info 1 lần
    pi = prod_info.copy()
    try:
        pi["pb"] = pd.qcut(
            pi["avg_price"], q=4,
            labels=["budget","mid","premium","luxury"],
            duplicates="drop"
        ).astype(str)
    except Exception:
        return []

    upsell = []
    for _, row in user_hist.iterrows():
        cat    = row["product_category_name_english"]
        price  = row["price"]

        # Xác định bucket hiện tại
        if price <= 40:   cur_b = "budget"
        elif price <= 75: cur_b = "mid"
        elif price <= 139:cur_b = "premium"
        else:             cur_b = "luxury"

        try:
            idx = BUCKET_ORDER.index(cur_b)
        except ValueError:
            continue
        if idx >= len(BUCKET_ORDER) - 1:
            continue

        next_b = BUCKET_ORDER[idx + 1]
        candidates = pi[
            (pi["category"] == cat) &
            (pi["pb"] == next_b) &
            (pi["avg_rating"] >= 4.0)
        ].sort_values("avg_rating", ascending=False)

        upsell.extend(candidates["product_id"].tolist()[:2])

    # Bỏ trùng và loại đã mua
    bought = set(user_hist["product_id"])
    seen   = set()
    result = []
    for p in upsell:
        if p not in bought and p not in seen:
            result.append(p)
            seen.add(p)
        if len(result) >= top_n:
            break
    return result


# ══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🛒 Olist Dashboard")
    st.caption("Hệ thống Đề xuất · Quản Trị")
    st.divider()

    page = st.radio(
        "Điều hướng",
        options=[
            "📊 Tổng quan",
            "📦 Sản phẩm & Danh mục",
            "👥 Khách hàng",
            "🎯 Hệ thống Đề xuất",
            "⚠️ Cảnh báo",
            "🗂️ Danh sách Khách hàng",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("**Thống kê nhanh**")
    st.metric("Tổng doanh thu", f"R$ {train_df['payment_value'].sum()/1e6:.2f}M")
    st.metric("Tổng đơn hàng", f"{train_df['order_id'].nunique():,}")
    st.metric("Avg Rating", f"{train_df['review_score'].mean():.2f} ★")
    st.divider()
    st.caption("Dữ liệu: Olist Brazil 2017")


# ══════════════════════════════════════════════════════════
# TRANG 1 — TỔNG QUAN
# ══════════════════════════════════════════════════════════
if "Tổng quan" in page:
    st.header("📊 Tổng quan Kinh doanh")
    st.caption("Toàn bộ dữ liệu năm 2017 · Olist E-commerce Brazil")

    # ── KPI row ──────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Tổng Doanh Thu",
              f"R$ {train_df['payment_value'].sum()/1e6:.2f}M",
              "+68% so Q1 2017")
    k2.metric("🛒 Đơn Hàng",
              f"{train_df['order_id'].nunique():,}")
    k3.metric("💵 Giá Trị TB / Đơn",
              f"R$ {train_df['payment_value'].mean():.0f}")
    k4.metric("👤 Khách Hàng",
              f"{train_df['customer_unique_id'].nunique():,}")
    k5.metric("⭐ Rating Trung Bình",
              f"{train_df['review_score'].mean():.2f}")

    st.divider()

    # ── Biểu đồ doanh thu theo tháng ─────────────────────
    col_chart, col_info = st.columns([3, 1])

    with col_chart:
        st.subheader("Doanh thu theo Tháng (2017)")
        fig_bar = go.Figure()
        colors = [
            "#f59e0b" if r["month_str"] == "2017-11" else "#3b82f6"
            for _, r in monthly.iterrows()
        ]
        fig_bar.add_trace(
            go.Bar(
                x=monthly["month_label"],
                y=monthly["revenue"],
                marker_color=colors,
                text=[f"R${v/1000:.0f}K" for v in monthly["revenue"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Doanh thu: R$%{y:,.0f}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)",
                       tickprefix="R$", tickformat=","),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("🟡 Tháng 11: Black Friday — R$803,625 (cao nhất năm, gấp 1.9x tháng thường)")

    with col_info:
        st.subheader("Phân bổ theo Tháng")
        best_month = monthly.loc[monthly["revenue"].idxmax()]
        worst_month = monthly.loc[monthly["revenue"].idxmin()]
        st.metric("Tháng cao nhất", best_month["month_label"],
                  f"R$ {best_month['revenue']/1000:.0f}K")
        st.metric("Tháng thấp nhất", worst_month["month_label"],
                  f"R$ {worst_month['revenue']/1000:.0f}K")
        st.metric("Trung bình / tháng",
                  f"R$ {monthly['revenue'].mean()/1000:.0f}K")
        st.metric("Tổng đơn cao nhất",
                  f"T{int(monthly.loc[monthly['orders'].idxmax(),'month_str'].split('-')[1])}",
                  f"{monthly['orders'].max():,} đơn")

    st.divider()

    # ── Top category & State ──────────────────────────────
    col_cat, col_state = st.columns(2)

    with col_cat:
        st.subheader("Top 10 Danh mục theo Doanh thu")
        top10 = cat_stats.head(10).copy()
        top10["icon"] = top10["category"].apply(get_icon)
        top10["label"] = top10["icon"] + " " + top10["category"]
        top10["revenue_k"] = top10["revenue"] / 1000

        fig_cat = px.bar(
            top10.sort_values("revenue_k"),
            x="revenue_k",
            y="label",
            orientation="h",
            color="avg_rating",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_color=[3.5, 5.0],
            labels={"revenue_k": "Doanh thu (K BRL)",
                    "label": "", "avg_rating": "Rating TB"},
            hover_data={"orders": True, "avg_price": ":.0f"},
        )
        fig_cat.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            showlegend=False,
            coloraxis_colorbar=dict(title="Rating", thickness=12, len=0.6),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_state:
        st.subheader("Top 10 Bang theo Doanh thu")
        top10_state = state_stats.head(10).copy()
        top10_state["revenue_k"] = top10_state["revenue"] / 1000

        fig_state = px.bar(
            top10_state.sort_values("revenue_k"),
            x="revenue_k",
            y="customer_state",
            orientation="h",
            color="revenue_k",
            color_continuous_scale=["#1e3a8a", "#3b82f6", "#93c5fd"],
            labels={"revenue_k": "Doanh thu (K BRL)",
                    "customer_state": "Bang"},
        )
        fig_state.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_state, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TRANG 2 — SẢN PHẨM & DANH MỤC
# ══════════════════════════════════════════════════════════
elif "Sản phẩm" in page:
    st.header("📦 Sản phẩm & Danh mục")
    st.caption("Phân tích hiệu suất từng ngành hàng")

    # ── Bộ lọc ───────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        sort_opt = st.selectbox(
            "Sắp xếp theo",
            ["Doanh thu", "Số đơn", "Rating", "Giá trung bình"],
        )
    with col_f2:
        min_orders = st.slider("Đơn hàng tối thiểu", 0, 500, 50, step=50)
    with col_f3:
        show_n = st.number_input("Hiển thị", 5, 30, 15, step=5)

    sort_map = {
        "Doanh thu": "revenue",
        "Số đơn": "orders",
        "Rating": "avg_rating",
        "Giá trung bình": "avg_price",
    }
    df_show = (
        cat_stats[cat_stats["orders"] >= min_orders]
        .sort_values(sort_map[sort_opt], ascending=False)
        .head(int(show_n))
    )

    st.divider()

    # ── Biểu đồ bubble ────────────────────────────────────
    st.subheader("Doanh thu × Rating × Số đơn")
    df_bubble = df_show.copy()
    df_bubble["icon_label"] = df_bubble["category"].apply(get_icon) + " " + df_bubble["category"]

    fig_bubble = px.scatter(
        df_bubble,
        x="avg_rating",
        y="revenue",
        size="orders",
        color="avg_price",
        hover_name="icon_label",
        color_continuous_scale=["#1e3a8a", "#f59e0b", "#ef4444"],
        size_max=60,
        labels={
            "avg_rating": "Rating Trung Bình",
            "revenue": "Doanh thu (BRL)",
            "orders": "Số đơn",
            "avg_price": "Giá TB (BRL)",
        },
        hover_data={
            "orders": ":,",
            "avg_price": ":.0f",
            "revenue": ":,.0f",
        },
    )
    fig_bubble.add_vline(x=4.0, line_dash="dash",
                         line_color="rgba(16,185,129,0.5)",
                         annotation_text="Ngưỡng 4.0★",
                         annotation_position="top right")
    fig_bubble.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.divider()

    # ── Bảng chi tiết ─────────────────────────────────────
    st.subheader("Bảng chi tiết Danh mục")
    table_df = df_show[[
        "category", "revenue", "orders", "avg_rating", "avg_price", "n_products"
    ]].copy()
    table_df["revenue"] = table_df["revenue"].round(0).astype(int)
    table_df["avg_rating"] = table_df["avg_rating"].round(2)
    table_df["avg_price"] = table_df["avg_price"].round(1)
    table_df.columns = [
        "Danh mục", "Doanh thu (BRL)", "Số đơn",
        "Rating TB", "Giá TB (BRL)", "Số SP"
    ]
    table_df.insert(0, "", table_df["Danh mục"].apply(get_icon))

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Doanh thu (BRL)": st.column_config.NumberColumn(format="R$ %d"),
            "Giá TB (BRL)": st.column_config.NumberColumn(format="R$ %.1f"),
            "Rating TB": st.column_config.ProgressColumn(
                min_value=1, max_value=5, format="%.2f ★"
            ),
            "Số đơn": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.divider()

    # ── Top sản phẩm ──────────────────────────────────────
    st.subheader("Top 20 Sản phẩm Doanh thu Cao Nhất")
    top_prod = (
        prod_info.sort_values("revenue", ascending=False)
        .head(20)
        .copy()
    )
    top_prod["icon"] = top_prod["category"].apply(get_icon)
    top_prod["product_short"] = top_prod["product_id"].str[:20] + "..."
    top_prod["revenue"] = top_prod["revenue"].round(0).astype(int)
    top_prod["avg_price"] = top_prod["avg_price"].round(1)
    top_prod["avg_rating"] = top_prod["avg_rating"].round(2)

    show_prod = top_prod[["icon", "category", "product_short", "revenue",
                           "avg_price", "n_sold", "avg_rating"]].copy()
    show_prod.columns = ["", "Danh mục", "Product ID", "Doanh thu (BRL)",
                          "Giá TB (BRL)", "Lượt bán", "Rating"]

    st.dataframe(
        show_prod,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Doanh thu (BRL)": st.column_config.NumberColumn(format="R$ %d"),
            "Giá TB (BRL)": st.column_config.NumberColumn(format="R$ %.1f"),
            "Rating": st.column_config.ProgressColumn(
                min_value=1, max_value=5, format="%.2f ★"
            ),
        },
    )


# ══════════════════════════════════════════════════════════
# TRANG 3 — KHÁCH HÀNG
# ══════════════════════════════════════════════════════════
elif "Khách hàng" in page:
    st.header("👥 Phân tích Khách hàng")
    st.caption("Hành vi & giá trị từng phân khúc")

    # ── KPI phân khúc ─────────────────────────────────────
    seg_order = ["one_time", "occasional", "regular"]
    cols_kpi = st.columns(len(seg_order))
    for col, s in zip(cols_kpi, seg_order):
        row = seg_stats[seg_stats["segment"] == s]
        if row.empty:
            continue
        r = row.iloc[0]
        with col:
            st.metric(
                label=f"{SEG_LABELS.get(s, s)}",
                value=f"{r['users']:,} KH",
                delta=f"DT: R$ {r['revenue']/1000:.0f}K",
            )

    st.divider()

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("Tỷ lệ Khách hàng theo Phân khúc")
        seg_count = seg_df["segment"].value_counts().reset_index()
        seg_count.columns = ["segment", "count"]
        seg_count["label"] = seg_count["segment"].map(SEG_LABELS)

        fig_pie = px.pie(
            seg_count,
            values="count",
            names="label",
            color_discrete_sequence=["#ef4444", "#3b82f6", "#8b5cf6", "#f59e0b"],
            hole=0.45,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("Doanh thu theo Phân khúc")
        seg_rev = seg_stats.copy()
        seg_rev["label"] = seg_rev["segment"].map(SEG_LABELS)
        seg_rev = seg_rev.sort_values("revenue", ascending=False)

        fig_seg = px.bar(
            seg_rev,
            x="label",
            y="revenue",
            color="label",
            color_discrete_sequence=["#3b82f6", "#ef4444", "#8b5cf6", "#f59e0b"],
            text=seg_rev["revenue"].apply(lambda v: f"R${v/1000:.0f}K"),
            labels={"label": "Phân khúc", "revenue": "Doanh thu (BRL)"},
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.divider()

    # ── Bảng so sánh phân khúc ────────────────────────────
    st.subheader("So sánh chi tiết Phân khúc")
    seg_table = seg_stats.copy()
    seg_table["label"] = seg_table["segment"].map(SEG_LABELS)
    seg_table["rev_pct"] = (seg_table["revenue"] / seg_table["revenue"].sum() * 100).round(1)
    seg_table = seg_table[["label", "users", "revenue", "rev_pct",
                             "avg_spend", "avg_rating", "orders"]]
    seg_table.columns = ["Phân khúc", "Số KH", "Doanh thu (BRL)", "Tỷ trọng (%)",
                          "Chi tiêu TB/đơn", "Rating TB", "Số đơn"]

    st.dataframe(
        seg_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Doanh thu (BRL)": st.column_config.NumberColumn(format="R$ %d"),
            "Chi tiêu TB/đơn": st.column_config.NumberColumn(format="R$ %.0f"),
            "Rating TB": st.column_config.ProgressColumn(
                min_value=1, max_value=5, format="%.2f ★"
            ),
            "Tỷ trọng (%)": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

    st.divider()

    # ── Insight ───────────────────────────────────────────
    st.subheader("Insight Chiến lược")
    one_pct = seg_stats[seg_stats["segment"] == "one_time"]["users"].values[0] / seg_df.shape[0] * 100
    one_rev_pct = seg_stats[seg_stats["segment"] == "one_time"]["revenue"].values[0] / seg_stats["revenue"].sum() * 100

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.error(f"**🚨 Retention thấp**\n\n{one_pct:.1f}% khách hàng chỉ mua 1 lần, chiếm {one_rev_pct:.1f}% doanh thu. Cần chương trình loyalty để giữ chân khách.")
    with col_i2:
        occ_row = seg_stats[seg_stats["segment"] == "occasional"].iloc[0]
        st.warning(f"**⚡ Tiềm năng Occasional**\n\n{occ_row['users']:,} khách 2–3 lần mua với rating {occ_row['avg_rating']:.2f}★. Đây là nhóm dễ chuyển đổi lên Regular nhất.")
    with col_i3:
        reg_row = seg_stats[seg_stats["segment"] == "regular"].iloc[0]
        st.success(f"**💎 Regular là 'vàng'**\n\nChỉ {reg_row['users']} users nhưng chi tiêu ổn định, rating {reg_row['avg_rating']:.2f}★. Ưu tiên chăm sóc VIP nhóm này.")


# ══════════════════════════════════════════════════════════
# TRANG 4 — HỆ THỐNG ĐỀ XUẤT
# ══════════════════════════════════════════════════════════
elif "Đề xuất" in page:
    st.header("🎯 Hệ thống Đề xuất Sản phẩm")
    st.caption("4 cách khám phá sản phẩm phù hợp cho từng khách hàng")

    # ── Tính sẵn dữ liệu cần dùng ────────────────────────
    prod_stats = prod_info.copy()

    # Co-purchase matrix (category level)
    @st.cache_data(show_spinner=False)
    def build_copurchase(_train):
        user_cats = _train.groupby("customer_unique_id")[
            "product_category_name_english"
        ].apply(list)
        co = {}
        for cats_list in user_cats:
            unique_cats = list(set(cats_list))
            for i in range(len(unique_cats)):
                for j in range(i + 1, len(unique_cats)):
                    pair = tuple(sorted([unique_cats[i], unique_cats[j]]))
                    co[pair] = co.get(pair, 0) + 1
        return co

    copurchase = build_copurchase(train_df)

    def get_related_cats(cat, top_k=5):
        """Trả về các category thường được mua cùng với cat."""
        related = []
        for (a, b), cnt in copurchase.items():
            if a == cat:
                related.append((b, cnt))
            elif b == cat:
                related.append((a, cnt))
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:top_k]

    # ── 4 TAB ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Theo Khách hàng",
        "📂 Theo Danh mục",
        "💰 Theo Ngân sách",
        "🔥 Top Bán chạy",
    ])

    # ══════════════════════════════════════════════════════
    # TAB 1 — GỢI Ý THEO KHÁCH HÀNG (cá nhân hoá)
    # ══════════════════════════════════════════════════════
    with tab1:
        st.subheader("Gợi ý cá nhân hoá theo lịch sử mua hàng")

        col_mode, col_topn = st.columns([4, 1])
        with col_mode:
            input_mode = st.radio(
                "Phương thức",
                ["🔎 Nhập mã khách hàng", "🎲 Chọn ngẫu nhiên theo phân khúc"],
                horizontal=True,
                label_visibility="collapsed",
                key="tab1_mode",
            )
        with col_topn:
            top_n = st.number_input("Số gợi ý", 5, 20, 10, key="tab1_topn")

        user_id = None

        if "Nhập" in input_mode:
            uid_input = st.text_input(
                "Customer Unique ID",
                placeholder="Nhập customer_unique_id...",
                key="tab1_uid",
            )
            if uid_input.strip():
                user_id = uid_input.strip()
                if user_id not in seg_dict:
                    st.warning("⚠️ Mã không có trong hệ thống — sẽ dùng gợi ý phổ biến nhất.")
        else:
            c_seg, c_btn = st.columns([2, 1])
            with c_seg:
                seg_choice = st.selectbox(
                    "Phân khúc",
                    options=["one_time", "occasional", "regular"],
                    format_func=lambda x: SEG_LABELS.get(x, x),
                    key="tab1_seg",
                )
            with c_btn:
                st.write("")
                if st.button("🎲 Chọn ngẫu nhiên", use_container_width=True, key="tab1_btn"):
                    pool = [u for u, s in seg_dict.items() if s == seg_choice]
                    if pool:
                        st.session_state["rand_uid"] = np.random.choice(pool)

            pool = [u for u, s in seg_dict.items() if s == seg_choice]
            user_id = st.session_state.get("rand_uid", pool[0] if pool else None)
            if user_id:
                st.info(f"👤 Đang xem: `{user_id}`")

        if not user_id:
            st.info("👆 Chọn khách hàng để hiển thị gợi ý")
        else:
            recs, strategy, segment = hybrid_recommend(user_id, top_n=int(top_n))

            # Thông tin khách hàng
            st.divider()
            user_hist = train_df[train_df["customer_unique_id"] == user_id]
            n_pur = len(user_hist)
            total_spend = user_hist["payment_value"].sum()
            avg_rat = user_hist["review_score"].mean() if n_pur else 0
            state_u = user_hist["customer_state"].iloc[0] if n_pur else "N/A"
            city_u  = user_hist["customer_city"].iloc[0]  if n_pur else "N/A"

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Phân khúc", SEG_LABELS.get(segment, segment))
            k2.metric("Số lần mua", n_pur)
            k3.metric("Tổng chi tiêu", f"R$ {total_spend:.0f}")
            k4.metric("Rating TB", f"{avg_rat:.1f} ★" if n_pur else "N/A")
            k5.metric("Khu vực", f"{city_u}, {state_u}")
            st.caption(f"🔧 Thuật toán: **{strategy}**")

            st.divider()
            col_hist, col_rec = st.columns([1, 2])

            with col_hist:
                st.subheader("📋 Lịch sử Mua hàng")
                if user_hist.empty:
                    st.info("Chưa có lịch sử.")
                else:
                    hs = user_hist[["product_category_name_english",
                                    "price","review_score",
                                    "order_purchase_timestamp"]].copy()
                    hs["icon"] = hs["product_category_name_english"].apply(get_icon)
                    hs["date"] = pd.to_datetime(
                        hs["order_purchase_timestamp"]
                    ).dt.strftime("%d/%m/%Y")
                    hs = hs[["icon","product_category_name_english",
                              "price","review_score","date"]]
                    hs.columns = ["","Danh mục","Giá (BRL)","Rating","Ngày mua"]
                    st.dataframe(
                        hs, use_container_width=True, hide_index=True,
                        column_config={
                            "Giá (BRL)": st.column_config.NumberColumn(format="R$ %.0f"),
                            "Rating": st.column_config.ProgressColumn(
                                min_value=1, max_value=5, format="%.1f ★"),
                        },
                    )

                # Danh mục thường mua cùng
                if n_pur > 0:
                    main_cat = user_hist["product_category_name_english"].value_counts().index[0]
                    related = get_related_cats(main_cat, top_k=4)
                    if related:
                        st.caption(f"**Danh mục thường mua cùng với {main_cat}:**")
                        for rc_cat, cnt in related:
                            st.caption(f"  {get_icon(rc_cat)} {rc_cat} · {cnt} lần")

            with col_rec:
                st.subheader(f"🎁 Top {top_n} Sản phẩm Gợi ý")
                if not recs:
                    st.warning("Không tìm được gợi ý phù hợp.")
                else:
                    # ── Cải thiện 1: Thêm total_cost + badges Amazon-style ──
                    rec_rows = []
                    for rank, pid in enumerate(recs, 1):
                        info       = prod_lookup.get(pid, {})
                        avg_price  = info.get("avg_price",  0)
                        avg_ship   = info.get("avg_freight", 0)
                        total_cost = avg_price + avg_ship
                        badges     = get_product_badges(pid, prod_lookup)
                        rec_rows.append({
                            "#":             rank,
                            "":              get_icon(info.get("category","")),
                            "Danh mục":      info.get("category","unknown"),
                            "Giá SP (BRL)":  round(avg_price),
                            "Phí ship":      round(avg_ship),
                            "Tổng chi phí":  round(total_cost),
                            "Rating":        round(info.get("avg_rating",0),2),
                            "Lượt bán":      info.get("n_sold",0),
                            "Badges":        " ".join(badges) if badges else "—",
                        })
                    rec_df = pd.DataFrame(rec_rows)
                    st.dataframe(
                        rec_df, use_container_width=True, hide_index=True,
                        column_config={
                            "Giá SP (BRL)": st.column_config.NumberColumn(format="R$ %d"),
                            "Phí ship":     st.column_config.NumberColumn(format="R$ %d"),
                            "Tổng chi phí": st.column_config.NumberColumn(format="R$ %d"),
                            "Rating": st.column_config.ProgressColumn(
                                min_value=1, max_value=5, format="%.2f ★"),
                            "Lượt bán": st.column_config.NumberColumn(format="%d"),
                        },
                    )

                    # ── Cải thiện 2: Upsell + Seasonal trong 2 cột ──────────
                    st.divider()
                    col_up, col_seas = st.columns(2)

                    with col_up:
                        st.markdown("**⬆️ Gợi ý nâng cấp** *(cùng danh mục, phân khúc giá cao hơn)*")
                        upsells = upsell_recommend(user_id, top_n=4)
                        if upsells:
                            for up_pid in upsells:
                                up_info   = prod_lookup.get(up_pid, {})
                                up_price  = up_info.get("avg_price",  0)
                                up_ship   = up_info.get("avg_freight", 0)
                                up_total  = up_price + up_ship
                                up_rat    = up_info.get("avg_rating", 0)
                                up_cat    = up_info.get("category",  "?")
                                up_badges = get_product_badges(up_pid, prod_lookup)
                                badge_str = " ".join(up_badges) if up_badges else ""
                                st.markdown(
                                    f"{get_icon(up_cat)} **{up_cat}** {badge_str}  \n"
                                    f"R$ {up_price:.0f} + R$ {up_ship:.0f} ship = "
                                    f"**R$ {up_total:.0f} tổng** · {chr(9733)}{up_rat:.2f}"
                                )
                                st.divider()
                        else:
                            st.caption("Không có gợi ý nâng cấp — user đang ở mức giá cao nhất.")

                    with col_seas:
                        from datetime import datetime
                        month_now = datetime.now().month
                        boost     = get_seasonal_boost(month_now)
                        if boost:
                            st.markdown(f"**📅 Hot tháng {month_now}** *(seasonal boost)*")
                            for s_cat, w in list(boost.items())[:5]:
                                bar_w = int((w - 1.0) / 1.5 * 100)
                                st.markdown(
                                    f"{get_icon(s_cat)} {s_cat}  \n"
                                    f"Boost ×{w:.1f} — ưu tiên xuất hiện trong gợi ý"
                                )
                            # Nếu segment là one_time → dùng seasonal rec
                            if segment == "one_time":
                                seas_recs = seasonal_recommend(user_id, top_n=3)
                                if seas_recs:
                                    st.caption("**🎯 Top 3 SP theo mùa vụ tháng này:**")
                                    for sp in seas_recs:
                                        sp_info = prod_lookup.get(sp, {})
                                        sp_total = sp_info.get("avg_price",0) + sp_info.get("avg_freight",0)
                                        sp_badges = get_product_badges(sp, prod_lookup)
                                        st.caption(
                                            f"  {get_icon(sp_info.get('category',''))} "
                                            f"{sp_info.get('category','?')} · "
                                            f"R${sp_total:.0f} tổng "
                                            f"{'· ' + ' '.join(sp_badges) if sp_badges else ''}"
                                        )
                        else:
                            st.caption(f"**📅 Tháng {month_now}** — không có seasonal boost đặc biệt.")
                            st.caption("Seasonal boost kích hoạt: T5, T11, T12, T1")

                    # ── Biểu đồ phân phối danh mục gợi ý ──────────────────
                    st.divider()
                    rec_cats = [prod_lookup.get(p,{}).get("category","unknown") for p in recs]
                    cat_cnt = pd.Series(rec_cats).value_counts().reset_index()
                    cat_cnt.columns = ["category","count"]
                    cat_cnt["label"] = cat_cnt["category"].apply(get_icon) + " " + cat_cnt["category"]
                    fig_rcat = px.bar(
                        cat_cnt, x="label", y="count",
                        color="category",
                        color_discrete_sequence=PLOTLY_COLORS,
                        text="count",
                        labels={"label":"","count":"Số SP gợi ý"},
                    )
                    fig_rcat.update_traces(textposition="outside")
                    fig_rcat.update_layout(
                        **PLOTLY_LAYOUT, height=240,
                        showlegend=False,
                        xaxis=dict(tickangle=-20),
                    )
                    st.plotly_chart(fig_rcat, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # TAB 2 — GỢI Ý THEO DANH MỤC (browse by category)
    # Nhà quản trị chọn danh mục → xem top SP + danh mục liên quan
    # ══════════════════════════════════════════════════════
    with tab2:
        st.subheader("Khám phá sản phẩm theo Danh mục")
        st.caption("Chọn danh mục để xem top sản phẩm và danh mục thường mua kèm")

        all_cats = sorted(
            cat_stats[cat_stats["orders"] >= 10]["category"].tolist()
        )
        col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
        with col_c1:
            chosen_cat = st.selectbox(
                "Chọn danh mục",
                options=all_cats,
                format_func=lambda x: get_icon(x) + " " + x,
                key="tab2_cat",
            )
        with col_c2:
            sort_prod = st.selectbox(
                "Sắp xếp",
                ["Doanh thu","Lượt bán","Rating","Giá"],
                key="tab2_sort",
            )
        with col_c3:
            min_rating = st.slider(
                "Rating tối thiểu", 1.0, 5.0, 4.0, 0.1, key="tab2_rating"
            )

        sort_col_map = {
            "Doanh thu": "revenue",
            "Lượt bán":  "n_sold",
            "Rating":    "avg_rating",
            "Giá":       "avg_price",
        }

        # Thống kê danh mục được chọn
        cat_row = cat_stats[cat_stats["category"] == chosen_cat]
        if not cat_row.empty:
            cr = cat_row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng doanh thu", f"R$ {cr['revenue']:,.0f}")
            m2.metric("Số đơn hàng",    f"{cr['orders']:,}")
            m3.metric("Avg Rating",     f"{cr['avg_rating']:.2f} ★")
            m4.metric("Giá trung bình", f"R$ {cr['avg_price']:.0f}")

        st.divider()
        col_prod, col_related = st.columns([3, 1])

        with col_prod:
            st.subheader(f"Top sản phẩm · {get_icon(chosen_cat)} {chosen_cat}")
            cat_prods = prod_stats[
                (prod_stats["category"] == chosen_cat) &
                (prod_stats["avg_rating"] >= min_rating)
            ].sort_values(sort_col_map[sort_prod], ascending=False).head(15)

            if cat_prods.empty:
                st.info(f"Không có sản phẩm nào có rating ≥ {min_rating}★ trong danh mục này.")
            else:
                show = cat_prods[[
                    "product_id","avg_price","avg_freight",
                    "avg_rating","n_sold","revenue"
                ]].copy()
                show["icon"] = chosen_cat
                show["icon"] = show["icon"].apply(get_icon)
                show["product_id"] = show["product_id"].str[:20] + "..."
                show.insert(0, "", show.pop("icon"))
                show.columns = [
                    "","Product ID","Giá TB (BRL)","Phí ship",
                    "Rating","Lượt bán","Doanh thu (BRL)"
                ]
                st.dataframe(
                    show, use_container_width=True, hide_index=True,
                    column_config={
                        "Giá TB (BRL)":    st.column_config.NumberColumn(format="R$ %.0f"),
                        "Phí ship":        st.column_config.NumberColumn(format="R$ %.0f"),
                        "Rating":          st.column_config.ProgressColumn(
                            min_value=1, max_value=5, format="%.2f ★"),
                        "Lượt bán":        st.column_config.NumberColumn(format="%d"),
                        "Doanh thu (BRL)": st.column_config.NumberColumn(format="R$ %d"),
                    },
                )

                # Biểu đồ giá vs rating
                fig_pv = px.scatter(
                    cat_prods,
                    x="avg_price", y="avg_rating",
                    size="n_sold", size_max=40,
                    color="revenue",
                    color_continuous_scale=["#1e3a8a","#3b82f6","#10b981"],
                    hover_data={"n_sold": ":,", "revenue": ":,.0f"},
                    labels={
                        "avg_price":  "Giá trung bình (BRL)",
                        "avg_rating": "Rating trung bình",
                        "n_sold":     "Lượt bán",
                        "revenue":    "Doanh thu",
                    },
                    title="Giá vs Rating (kích thước = lượt bán)",
                )
                fig_pv.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
                st.plotly_chart(fig_pv, use_container_width=True)

        with col_related:
            st.subheader("Danh mục liên quan")
            st.caption("Thường được mua cùng nhau")
            related_cats = get_related_cats(chosen_cat, top_k=8)
            if related_cats:
                for rc, cnt in related_cats:
                    rc_row = cat_stats[cat_stats["category"] == rc]
                    rc_rev = rc_row["revenue"].values[0] if not rc_row.empty else 0
                    st.markdown(f"""
**{get_icon(rc)} {rc}**
- Mua kèm: **{cnt}** lần
- DT: R$ {rc_rev:,.0f}
""")
                    st.divider()
            else:
                st.info("Không có dữ liệu co-purchase.")

    # ══════════════════════════════════════════════════════
    # TAB 3 — GỢI Ý THEO NGÂN SÁCH
    # Nhà quản trị nhập khoảng giá → hệ thống trả về
    # sản phẩm phù hợp nhất trong ngân sách đó
    # ══════════════════════════════════════════════════════
    with tab3:
        st.subheader("Tìm sản phẩm phù hợp theo Ngân sách")
        st.caption("Lọc sản phẩm theo khoảng giá và tiêu chí ưu tiên của khách hàng")

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            price_min = st.number_input(
                "Giá tối thiểu (BRL)", 0, 5000, 0, step=10, key="b_min"
            )
        with col_b2:
            price_max = st.number_input(
                "Giá tối đa (BRL)", 10, 6000, 200, step=10, key="b_max"
            )
        with col_b3:
            budget_cat = st.selectbox(
                "Danh mục (tuỳ chọn)",
                ["Tất cả"] + sorted(cat_stats["category"].tolist()),
                key="b_cat",
            )
        with col_b4:
            budget_sort = st.selectbox(
                "Ưu tiên",
                ["Rating cao nhất", "Bán chạy nhất",
                 "Giá thấp nhất", "Doanh thu cao nhất"],
                key="b_sort",
            )

        budget_sort_map = {
            "Rating cao nhất":    ("avg_rating", False),
            "Bán chạy nhất":      ("n_sold",     False),
            "Giá thấp nhất":      ("avg_price",  True),
            "Doanh thu cao nhất": ("revenue",    False),
        }
        b_col, b_asc = budget_sort_map[budget_sort]

        # Lọc sản phẩm
        b_filtered = prod_stats[
            (prod_stats["avg_price"] >= price_min) &
            (prod_stats["avg_price"] <= price_max) &
            (prod_stats["avg_rating"] >= 4.0)
        ]
        if budget_cat != "Tất cả":
            b_filtered = b_filtered[b_filtered["category"] == budget_cat]

        b_result = b_filtered.sort_values(b_col, ascending=b_asc).head(20)

        # Summary KPI
        st.divider()
        bk1, bk2, bk3, bk4 = st.columns(4)
        bk1.metric("Sản phẩm tìm thấy",  f"{len(b_filtered):,}")
        bk2.metric("Giá trung bình",      f"R$ {b_filtered['avg_price'].mean():.0f}" if len(b_filtered) else "—")
        bk3.metric("Rating TB",           f"{b_filtered['avg_rating'].mean():.2f} ★" if len(b_filtered) else "—")
        bk4.metric("Danh mục có sản phẩm", f"{b_filtered['category'].nunique()}")

        if b_result.empty:
            st.warning(f"Không có sản phẩm nào trong khoảng R${price_min}–R${price_max} với rating ≥ 4.0★")
        else:
            col_tbl, col_chart = st.columns([3, 2])

            with col_tbl:
                st.subheader(f"Top 20 sản phẩm · R${price_min}–R${price_max}")
                b_show = b_result[[
                    "product_id","category","avg_price",
                    "avg_freight","avg_rating","n_sold"
                ]].copy()
                b_show["total_cost"] = b_show["avg_price"] + b_show["avg_freight"]
                b_show["badge"]      = b_show["product_id"].apply(
                    lambda p: " ".join(get_product_badges(p, prod_lookup))
                )
                b_show["icon"] = b_show["category"].apply(get_icon)
                b_show["product_id"] = b_show["product_id"].str[:18] + "..."
                b_show = b_show[["icon","category","product_id",
                                  "avg_price","avg_freight","total_cost",
                                  "avg_rating","n_sold","badge"]]
                b_show.columns = ["","Danh mục","Product ID",
                                   "Giá (BRL)","Ship","Tổng chi phí",
                                   "Rating","Bán","Badges"]
                st.dataframe(
                    b_show, use_container_width=True, hide_index=True,
                    column_config={
                        "Giá (BRL)":     st.column_config.NumberColumn(format="R$ %.0f"),
                        "Ship":          st.column_config.NumberColumn(format="R$ %.0f"),
                        "Tổng chi phí":  st.column_config.NumberColumn(format="R$ %.0f"),
                        "Rating":        st.column_config.ProgressColumn(
                            min_value=1, max_value=5, format="%.2f ★"),
                    },
                )

            with col_chart:
                st.subheader("Phân bổ theo Danh mục")
                b_cat_dist = (
                    b_filtered.groupby("category")
                    .agg(count=("product_id","count"),
                         avg_rating=("avg_rating","mean"))
                    .reset_index()
                    .sort_values("count", ascending=False)
                    .head(10)
                )
                b_cat_dist["label"] = b_cat_dist["category"].apply(get_icon) + " " + b_cat_dist["category"]
                fig_bdist = px.bar(
                    b_cat_dist.sort_values("count"),
                    x="count", y="label",
                    orientation="h",
                    color="avg_rating",
                    color_continuous_scale=["#f59e0b","#10b981"],
                    range_color=[3.5, 5.0],
                    labels={"count":"Số SP","label":"","avg_rating":"Rating TB"},
                    text="count",
                )
                fig_bdist.update_traces(textposition="outside")
                fig_bdist.update_layout(
                    **PLOTLY_LAYOUT, height=320,
                    showlegend=False,
                    coloraxis_colorbar=dict(title="Rating", thickness=10, len=0.5),
                )
                st.plotly_chart(fig_bdist, use_container_width=True)

                # Phân phối giá
                fig_bprice = px.histogram(
                    b_filtered, x="avg_price",
                    nbins=20,
                    labels={"avg_price": "Giá (BRL)", "count": "Số SP"},
                    color_discrete_sequence=["#3b82f6"],
                    title="Phân phối giá trong kết quả",
                )
                fig_bprice.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False)
                st.plotly_chart(fig_bprice, use_container_width=True)

    # ══════════════════════════════════════════════════════
    # TAB 4 — TOP BÁN CHẠY (trending / bestseller)
    # Không cần user — hiển thị xu hướng toàn sàn
    # ══════════════════════════════════════════════════════
    with tab4:
        st.subheader("Sản phẩm & Danh mục bán chạy nhất")
        st.caption("Gợi ý dựa trên xu hướng toàn sàn — phù hợp cho khách hàng mới")

        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            trend_by = st.radio(
                "Xem theo",
                ["Lượt bán", "Doanh thu", "Rating cao nhất"],
                horizontal=True,
                key="trend_by",
            )
        with col_t2:
            trend_cat_filter = st.selectbox(
                "Lọc danh mục",
                ["Tất cả"] + sorted(cat_stats["category"].tolist()),
                key="trend_cat",
            )

        trend_col_map = {
            "Lượt bán":        "n_sold",
            "Doanh thu":       "revenue",
            "Rating cao nhất": "avg_rating",
        }
        t_col = trend_col_map[trend_by]

        t_data = prod_stats[prod_stats["avg_rating"] >= 4.0].copy()
        if trend_cat_filter != "Tất cả":
            t_data = t_data[t_data["category"] == trend_cat_filter]
        t_top = t_data.sort_values(t_col, ascending=False).head(20)

        st.divider()

        # KPI top3
        col_p1, col_p2, col_p3 = st.columns(3)
        for col, (_, row) in zip([col_p1, col_p2, col_p3],
                                  t_top.head(3).iterrows()):
            with col:
                st.metric(
                    label=f"{get_icon(row['category'])} #{list(t_top.index).index(_)+1} · {row['category']}",
                    value=f"R$ {row['avg_price']:.0f}",
                    delta=f"★ {row['avg_rating']:.2f} · {row['n_sold']:,} bán",
                )

        st.divider()
        col_list, col_vis = st.columns([3, 2])

        with col_list:
            st.subheader(f"Top 20 · {trend_by}")
            t_show = t_top[[
                "product_id","category","avg_price","avg_freight",
                "avg_rating","n_sold","revenue"
            ]].copy()
            t_show["total_cost"] = t_show["avg_price"] + t_show["avg_freight"]
            t_show["badge"]      = t_show["product_id"].apply(
                lambda p: " ".join(get_product_badges(p, prod_lookup))
            )
            t_show.insert(0, "rank", range(1, len(t_show)+1))
            t_show["icon"] = t_show["category"].apply(get_icon)
            t_show["product_id"] = t_show["product_id"].str[:18] + "..."
            t_show = t_show[[
                "rank","icon","category","product_id",
                "avg_price","total_cost","avg_rating","n_sold","badge"
            ]]
            t_show.columns = [
                "#","","Danh mục","Product ID",
                "Giá (BRL)","Tổng chi phí","Rating","Lượt bán","Badges"
            ]
            st.dataframe(
                t_show, use_container_width=True, hide_index=True,
                column_config={
                    "Giá (BRL)":     st.column_config.NumberColumn(format="R$ %.0f"),
                    "Tổng chi phí":  st.column_config.NumberColumn(format="R$ %.0f"),
                    "Rating":        st.column_config.ProgressColumn(
                        min_value=1, max_value=5, format="%.2f ★"),
                    "Lượt bán":      st.column_config.NumberColumn(format="%d"),
                },
            )

        with col_vis:
            # Top categories doanh thu
            st.subheader("Top danh mục")
            # Map t_col (prod_stats columns) -> aggregated column names
            _tcat_sort_map = {
                "n_sold":     "total_sold",
                "revenue":    "total_rev",
                "avg_rating": "avg_rat",
            }
            _tcat_sort_col = _tcat_sort_map.get(t_col, "total_rev")

            t_cat = (
                t_data.groupby("category")
                .agg(total_rev=("revenue","sum"),
                     total_sold=("n_sold","sum"),
                     avg_rat=("avg_rating","mean"))
                .reset_index()
                .sort_values(_tcat_sort_col, ascending=False)
                .head(10)
            )
            t_cat["label"] = t_cat["category"].apply(get_icon) + " " + t_cat["category"]
            fig_tcat = px.bar(
                t_cat.sort_values("total_rev"),
                x="total_rev", y="label",
                orientation="h",
                color="avg_rat",
                color_continuous_scale=["#f59e0b","#10b981"],
                range_color=[3.8, 5.0],
                text=t_cat.sort_values("total_rev")["total_rev"].apply(
                    lambda v: f"R${v/1000:.0f}K"
                ),
                labels={"total_rev":"Doanh thu","label":"","avg_rat":"Rating TB"},
            )
            fig_tcat.update_traces(textposition="outside")
            fig_tcat.update_layout(
                **PLOTLY_LAYOUT, height=320, showlegend=False,
                coloraxis_colorbar=dict(title="Rating", thickness=10, len=0.5),
            )
            st.plotly_chart(fig_tcat, use_container_width=True)

            # Rating distribution của top products
            fig_trat = px.histogram(
                t_top, x="avg_rating", nbins=10,
                color_discrete_sequence=["#10b981"],
                labels={"avg_rating":"Rating","count":"Số SP"},
                title="Phân phối rating top SP",
            )
            fig_trat.update_layout(**PLOTLY_LAYOUT, height=180, showlegend=False)
            st.plotly_chart(fig_trat, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TRANG 5 — CẢNH BÁO
# ══════════════════════════════════════════════════════════
elif "Cảnh báo" in page:
    st.header("⚠️ Cảnh báo & Rủi ro Kinh doanh")
    st.caption("Các vấn đề cần xử lý để tối ưu hiệu quả kinh doanh")

    # ── KPI cảnh báo ─────────────────────────────────────
    train_df["freight_ratio"] = train_df["freight_value"] / train_df["price"].replace(0, np.nan)
    high_freight_pct = (train_df["freight_ratio"] > 0.5).mean() * 100
    one_time_pct = (seg_df["segment"] == "one_time").mean() * 100
    nov_rev = monthly[monthly["month_str"] == "2017-11"]["revenue"].values[0]
    avg_rev_no_nov = monthly[monthly["month_str"] != "2017-11"]["revenue"].mean()
    spike = nov_rev / avg_rev_no_nov
    low_cat_count = (cat_stats["avg_rating"] < 4.0).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚚 Đơn phí ship > 50% giá SP", f"{high_freight_pct:.1f}%",
              delta="Cần tối ưu logistics", delta_color="inverse")
    k2.metric("👋 Tỷ lệ khách mua 1 lần", f"{one_time_pct:.1f}%",
              delta="Retention thấp", delta_color="inverse")
    k3.metric("📈 Spike Black Friday (T11)", f"{spike:.1f}x",
              delta="Hạ tầng cần chuẩn bị", delta_color="off")
    k4.metric("⭐ Danh mục Rating < 4.0", f"{low_cat_count} danh mục",
              delta="Đã loại khỏi gợi ý", delta_color="off")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["🚚 Phí vận chuyển", "⭐ Chất lượng sản phẩm", "📋 Hành động ưu tiên"])

    with tab1:
        st.subheader("Danh mục có Phí Ship cao (freight > 50% giá sản phẩm)")
        freight_risk = (
            train_df[train_df["freight_ratio"] > 0.5]
            .groupby("product_category_name_english")
            .agg(
                high_freight_orders=("order_id", "count"),
                avg_freight=("freight_value", "mean"),
                avg_price=("price", "mean"),
            )
            .reset_index()
            .sort_values("high_freight_orders", ascending=False)
            .head(12)
        )
        freight_risk["avg_ratio"] = freight_risk["avg_freight"] / freight_risk["avg_price"]
        freight_risk["icon"] = freight_risk["product_category_name_english"].apply(get_icon)
        freight_risk["label"] = freight_risk["icon"] + " " + freight_risk["product_category_name_english"]

        fig_freight = px.bar(
            freight_risk.sort_values("high_freight_orders"),
            x="high_freight_orders",
            y="label",
            orientation="h",
            color="avg_ratio",
            color_continuous_scale=["#fbbf24", "#f97316", "#ef4444"],
            labels={
                "high_freight_orders": "Số đơn phí ship cao",
                "label": "",
                "avg_ratio": "Tỷ lệ TB",
            },
            text="high_freight_orders",
        )
        fig_freight.update_traces(textposition="outside")
        fig_freight.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig_freight, use_container_width=True)
        st.warning("**Khuyến nghị:** Đàm phán lại hợp đồng logistics cho garden_tools, housewares. Cân nhắc freeship đơn ≥ R$200 để tăng tỷ lệ chuyển đổi.")

    with tab2:
        st.subheader("Danh mục Rating thấp — Cần cải thiện")
        col_low, col_high = st.columns(2)

        with col_low:
            st.caption("⚠️ Dưới ngưỡng 4.0 — Đã loại khỏi hệ thống gợi ý")
            low_cats = cat_stats[cat_stats["avg_rating"] < 4.0].sort_values("avg_rating").head(10)
            low_show = low_cats[["category", "avg_rating", "revenue", "orders"]].copy()
            low_show["icon"] = low_show["category"].apply(get_icon)
            low_show = low_show[["icon", "category", "avg_rating", "revenue", "orders"]]
            low_show.columns = ["", "Danh mục", "Rating", "Doanh thu", "Đơn"]
            st.dataframe(
                low_show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rating": st.column_config.ProgressColumn(
                        min_value=1, max_value=5, format="%.2f ★"
                    ),
                    "Doanh thu": st.column_config.NumberColumn(format="R$ %d"),
                },
            )

        with col_high:
            st.caption("✅ Trên ngưỡng 4.0 — Đang được gợi ý")
            high_cats = cat_stats[cat_stats["avg_rating"] >= 4.0].sort_values("avg_rating", ascending=False).head(10)
            high_show = high_cats[["category", "avg_rating", "revenue", "orders"]].copy()
            high_show["icon"] = high_show["category"].apply(get_icon)
            high_show = high_show[["icon", "category", "avg_rating", "revenue", "orders"]]
            high_show.columns = ["", "Danh mục", "Rating", "Doanh thu", "Đơn"]
            st.dataframe(
                high_show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rating": st.column_config.ProgressColumn(
                        min_value=1, max_value=5, format="%.2f ★"
                    ),
                    "Doanh thu": st.column_config.NumberColumn(format="R$ %d"),
                },
            )

    with tab3:
        st.subheader("Danh sách Hành động Ưu tiên")
        actions = [
            ("🔴", "Khẩn cấp", "Tối ưu logistics garden_tools & housewares",
             f"Hiện có hơn 1,000 đơn phí ship > 50% giá sản phẩm. "
             f"Cần đàm phán lại hợp đồng vận chuyển hoặc điều chỉnh giá bán."),
            ("🔴", "Khẩn cấp", f"Chương trình giữ chân {one_time_pct:.0f}% khách vãng lai",
             "Thiết lập email/SMS retargeting sau 30 ngày không mua. "
             "Mục tiêu: chuyển đổi 5% khách vãng lai thành occasional — tương đương +R$250K doanh thu."),
            ("🟡", "Quan trọng", f"Chuẩn bị hạ tầng Black Friday (spike {spike:.1f}x)",
             "Tháng 11 cần tăng cường kho vận và server. "
             "Cập nhật model gợi ý theo seasonal trends trước tháng 10."),
            ("🟡", "Quan trọng", "Nâng chất lượng sản phẩm rating < 4.0",
             f"{low_cat_count} danh mục đã bị loại khỏi hệ thống gợi ý. "
             "Làm việc với người bán để cải thiện — khi đạt 4.0★ sẽ được đưa trở lại pool."),
            ("🟢", "Dài hạn", "Mở rộng thị trường RS, PR, SC",
             "3 bang miền Nam tăng trưởng ổn định, dư địa lớn. "
             "Chạy chiến dịch marketing địa phương kết hợp gợi ý theo bang."),
            ("🟢", "Dài hạn", "Cập nhật model Recommendation định kỳ",
             "Model hiện tại train trên data 2017. Cần retrain theo quý "
             "để bắt kịp xu hướng và đưa sản phẩm mới vào pool gợi ý."),
        ]

        for ico, level, title, desc in actions:
            if level == "Khẩn cấp":
                st.error(f"**{ico} [{level}] {title}**\n\n{desc}")
            elif level == "Quan trọng":
                st.warning(f"**{ico} [{level}] {title}**\n\n{desc}")
            else:
                st.success(f"**{ico} [{level}] {title}**\n\n{desc}")


# ══════════════════════════════════════════════════════════
# TRANG 6 — DANH SÁCH KHÁCH HÀNG
# ══════════════════════════════════════════════════════════
elif "Danh sách" in page:
    st.header("🗂️ Danh sách Khách hàng trong Hệ thống Đề xuất")
    st.caption("Toàn bộ customer_unique_id đã được phân loại và sẵn sàng nhận gợi ý")

    # ── Tính sẵn bảng user_info đầy đủ ──────────────────
    @st.cache_data(show_spinner=False)
    def build_user_table(_train, _seg_df):
        info = _train.groupby("customer_unique_id").agg(
            so_don     = ("order_id",                      "nunique"),
            tong_chi   = ("payment_value",                 "sum"),
            avg_rating = ("review_score",                  "mean"),
            state      = ("customer_state",                "first"),
            city       = ("customer_city",                 "first"),
            categories = ("product_category_name_english",
                          lambda x: ", ".join(x.dropna().unique()[:3])),
            last_order = ("order_purchase_timestamp",      "max"),
        ).reset_index()
        df = _seg_df.merge(info, on="customer_unique_id", how="left")
        df["avg_rating"] = df["avg_rating"].round(2)
        df["tong_chi"]   = df["tong_chi"].round(0)
        seg_order_map = {"regular": 0, "occasional": 1, "one_time": 2}
        df["_so"] = df["segment"].map(seg_order_map)
        df = df.sort_values(["_so", "tong_chi"], ascending=[True, False]).drop(columns="_so")
        return df

    user_table = build_user_table(train_df, seg_df)

    # ── KPI tổng quan ────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tổng khách hàng",    f"{len(user_table):,}")
    k2.metric("Regular (≥4 lần)",   f"{(user_table['segment']=='regular').sum():,}")
    k3.metric("Occasional (2–3 lần)",f"{(user_table['segment']=='occasional').sum():,}")
    k4.metric("One-time (1 lần)",   f"{(user_table['segment']=='one_time').sum():,}")

    st.divider()

    # ── Bộ lọc ───────────────────────────────────────────
    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 1])
    with col_f1:
        seg_filter = st.multiselect(
            "Phân khúc",
            options=["regular", "occasional", "one_time"],
            default=["regular", "occasional", "one_time"],
            format_func=lambda x: SEG_LABELS.get(x, x),
            key="ul_seg",
        )
    with col_f2:
        state_filter = st.multiselect(
            "Bang",
            options=sorted(user_table["state"].dropna().unique()),
            default=[],
            key="ul_state",
            placeholder="Tất cả bang",
        )
    with col_f3:
        search_id = st.text_input(
            "Tìm theo Customer ID",
            placeholder="Nhập một phần mã...",
            key="ul_search",
        )
    with col_f4:
        sort_by = st.selectbox(
            "Sắp xếp",
            ["Tổng chi tiêu", "Số đơn", "Rating"],
            key="ul_sort",
        )

    sort_col_map = {
        "Tổng chi tiêu": "tong_chi",
        "Số đơn":        "so_don",
        "Rating":        "avg_rating",
    }

    # ── Áp bộ lọc ────────────────────────────────────────
    df_show = user_table.copy()
    if seg_filter:
        df_show = df_show[df_show["segment"].isin(seg_filter)]
    if state_filter:
        df_show = df_show[df_show["state"].isin(state_filter)]
    if search_id.strip():
        df_show = df_show[
            df_show["customer_unique_id"].str.contains(
                search_id.strip(), case=False, na=False
            )
        ]
    df_show = df_show.sort_values(sort_col_map[sort_by], ascending=False)

    st.caption(f"Hiển thị **{len(df_show):,}** / {len(user_table):,} khách hàng")

    # ── Bảng chính ───────────────────────────────────────
    display_df = df_show[[
        "customer_unique_id", "segment",
        "so_don", "tong_chi", "avg_rating",
        "state", "city", "categories",
    ]].copy()
    display_df["segment"] = display_df["segment"].map(SEG_LABELS)
    display_df.columns = [
        "Customer ID", "Phân khúc",
        "Số đơn", "Tổng chi (BRL)", "Rating TB",
        "Bang", "Thành phố", "Danh mục đã mua",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "Customer ID":      st.column_config.TextColumn(width="medium"),
            "Tổng chi (BRL)":   st.column_config.NumberColumn(format="R$ %d"),
            "Rating TB":        st.column_config.ProgressColumn(
                                    min_value=1, max_value=5, format="%.2f ★"),
            "Số đơn":           st.column_config.NumberColumn(format="%d"),
            "Danh mục đã mua":  st.column_config.TextColumn(width="large"),
        },
    )

    st.divider()

    # ── Copy nhanh + Xuất CSV ─────────────────────────────
    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex1:
        st.subheader("Danh sách Customer ID thuần")
        id_list = df_show["customer_unique_id"].tolist()
        st.text_area(
            label="Sao chép để dùng trong hệ thống khác",
            value=chr(10).join(id_list),
            height=200,
            key="ul_ids",
        )
        st.caption(f"{len(id_list):,} mã · Có thể copy paste trực tiếp vào ô nhập trên trang Đề xuất")

    with col_ex2:
        st.subheader("Xuất dữ liệu")

        # Xuất theo segment
        for seg_name in ["regular", "occasional", "one_time"]:
            seg_data = df_show[df_show["segment"].map(
                lambda x: x == SEG_LABELS.get(seg_name, seg_name)
                if isinstance(x, str) and x in SEG_LABELS.values()
                else x == seg_name
            ) if "segment" in df_show.columns else df_show.index]

            # Lấy từ df_show gốc (trước khi đổi tên cột)
            seg_raw = user_table[user_table["segment"] == seg_name]
            if state_filter:
                seg_raw = seg_raw[seg_raw["state"].isin(state_filter)]

            csv_bytes = seg_raw.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            lbl = SEG_LABELS.get(seg_name, seg_name)
            st.download_button(
                label=f"⬇️ {lbl} ({len(seg_raw):,})",
                data=csv_bytes,
                file_name=f"customers_{seg_name}.csv",
                mime="text/csv",
                use_container_width=True,
                key=f"dl_{seg_name}",
            )

        st.divider()

        # Xuất toàn bộ kết quả lọc hiện tại
        full_csv = df_show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label=f"⬇️ Tất cả ({len(df_show):,} KH)",
            data=full_csv,
            file_name="customers_filtered.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_all",
        )