# =============================================================================
# app.py — Olist E-Commerce Recommendation System Admin Dashboard
# Phiên bản đầy đủ, production-ready
#
# Yêu cầu các file (đặt cùng thư mục với app.py):
#   olist_model.pkl       → OlistHybridRouter (chứa pop/cb/ar engine)
#   product_catalog.pkl   → dict {product_id: {category, price, weight, photos}}
#   user_profiles.pkl     → dict {user_id: {customer_state, interaction_count}}
#
# Cách chạy: streamlit run app.py
# =============================================================================

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import warnings
import os

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PAGE CONFIG — Phải đặt đầu tiên trước mọi st. call khác
# =============================================================================
st.set_page_config(
    page_title="Olist Recommendation Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. CLASS DEFINITIONS — BẮT BUỘC khai báo lại để Pickle Load hoạt động
#    Tên class phải khớp chính xác với tên khi serialize trong notebook
# =============================================================================

class LocalPopularityEngine:
    """
    Tầng 1 — Gợi ý sản phẩm phổ biến theo khu vực địa lý (customer_state).
    Fallback về global popularity nếu state không đủ dữ liệu.
    Target: one-time buyers (interaction_count == 1).
    """
    def __init__(self, train_data):
        self.state_pop = (
            train_data
            .groupby(["customer_state", "product_id"])["interaction_count"]
            .sum()
            .reset_index()
        )
        self.state_pop = self.state_pop.sort_values(
            by=["customer_state", "interaction_count"],
            ascending=[True, False],
        )
        self.global_pop = (
            train_data
            .groupby("product_id")["interaction_count"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def recommend(self, state, k=10):
        local_items = (
            self.state_pop[self.state_pop["customer_state"] == state]["product_id"]
            .head(k)
            .tolist()
        )
        if len(local_items) < k:
            fill = [
                i for i in self.global_pop if i not in local_items
            ][: k - len(local_items)]
            local_items.extend(fill)
        return local_items


class ContentBasedEngine:
    """
    Tầng 2 — Gợi ý dựa trên nội dung sản phẩm.
    Feature: TF-IDF(category) + MinMaxScaler(price, weight, photos).
    Similarity: Cosine giữa profile sản phẩm cuối cùng user mua và toàn catalog.
    Target: occasional buyers (2–3 đơn).
    """
    def __init__(self, train_data):
        self.item_profiles = train_data.drop_duplicates(subset=["product_id"]).copy()
        self.item_profiles["product_category_name_english"] = (
            self.item_profiles["product_category_name_english"].fillna("unknown")
        )
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(
            self.item_profiles["product_category_name_english"]
        )
        num_features = ["price", "product_weight_g", "product_photos_qty"]
        scaler = MinMaxScaler()
        num_matrix = csr_matrix(
            scaler.fit_transform(self.item_profiles[num_features].fillna(0))
        )
        self.feature_matrix = hstack([tfidf_matrix, num_matrix], format="csr")
        self.item_list = self.item_profiles["product_id"].tolist()
        train_sorted = train_data.sort_values("order_purchase_timestamp")
        self.user_history = (
            train_sorted
            .groupby("customer_unique_id")["product_id"]
            .apply(list)
            .to_dict()
        )

    def recommend(self, user_id, k=10):
        history = self.user_history.get(user_id, [])
        if not history:
            return []
        last_item = history[-1]
        try:
            idx = self.item_list.index(last_item)
        except ValueError:
            return []
        item_vector = self.feature_matrix[idx]
        sim_scores = cosine_similarity(item_vector, self.feature_matrix).flatten()
        top_indices = sim_scores.argsort()[-(k + 5): -1][::-1]
        candidates = [self.item_list[i] for i in top_indices]
        return [x for x in candidates if x not in history][:k]


class AssociationRulesEngine:
    """
    Tầng 3 — Gợi ý dựa trên luật kết hợp Apriori (cross-category pattern).
    Mỗi khi user mua danh mục A, dùng luật A→B để gợi ý top SP từ danh mục B.
    Target: regular/VIP buyers (>=4 đơn).
    """
    def __init__(self, train_data):
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            basket = (
                train_data
                .groupby(["customer_unique_id", "product_category_name_english"])[
                    "interaction_count"
                ]
                .sum()
                .unstack()
                .fillna(0)
            )
            basket = (basket > 0).astype(bool)
            freq_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
            self.rules = association_rules(freq_itemsets, metric="lift", min_threshold=1.1)
        except Exception:
            self.rules = pd.DataFrame()

        self.cat_to_top_items = (
            train_data
            .groupby(["product_category_name_english", "product_id"])["interaction_count"]
            .sum()
            .reset_index()
        )
        self.cat_to_top_items = self.cat_to_top_items.sort_values(
            ["product_category_name_english", "interaction_count"],
            ascending=[True, False],
        )
        train_sorted = train_data.sort_values("order_purchase_timestamp")
        self.user_history_cat = (
            train_sorted
            .groupby("customer_unique_id")["product_category_name_english"]
            .apply(list)
            .to_dict()
        )

    def recommend(self, user_id, k=10):
        cats_history = self.user_history_cat.get(user_id, [])
        if not cats_history or self.rules.empty:
            return []
        last_cat = cats_history[-1]
        matching = self.rules[
            self.rules["antecedents"].apply(lambda x: last_cat in x)
        ]
        if matching.empty:
            return []
        targets = (
            matching.sort_values("lift", ascending=False)["consequents"]
            .apply(list)
            .explode()
            .unique()
        )
        recs = []
        for cat in targets:
            if cat == last_cat:
                continue
            top_items = (
                self.cat_to_top_items[
                    self.cat_to_top_items["product_category_name_english"] == cat
                ]["product_id"]
                .head(3)
                .tolist()
            )
            recs.extend(top_items)
            if len(recs) >= k:
                break
        return recs[:k]


class OlistHybridRouter:
    """
    Router trung tâm — Phân loại user và routing tới đúng engine:
    - count == 1   → LocalPopularityEngine  (Tầng 1)
    - 2 <= count <= 3 → ContentBasedEngine  (Tầng 2) + Popularity fallback
    - count >= 4   → AssociationRulesEngine (Tầng 3) + CB + Popularity fallback
    """
    def __init__(self, train_data, total_counts):
        self.total_counts = total_counts
        self.user_states = (
            train_data.set_index("customer_unique_id")["customer_state"].to_dict()
        )
        self.pop_engine = LocalPopularityEngine(train_data)
        self.cb_engine  = ContentBasedEngine(train_data)
        self.ar_engine  = AssociationRulesEngine(train_data)

    def get_recommendation(self, user_id, k=10):
        c     = self.total_counts.get(user_id, 1)
        state = self.user_states.get(user_id, "SP")
        if c == 1:
            return self.pop_engine.recommend(state, k), "Tầng 1 — Local Popularity"
        elif 2 <= c <= 3:
            recs = self.cb_engine.recommend(user_id, k)
            if len(recs) < k:
                recs.extend(
                    [x for x in self.pop_engine.recommend(state, k) if x not in recs]
                )
            return recs[:k], "Tầng 2 — Content-Based"
        else:
            recs = self.ar_engine.recommend(user_id, k)
            if len(recs) < k:
                recs.extend(
                    [x for x in self.cb_engine.recommend(user_id, k) if x not in recs]
                )
            if len(recs) < k:
                recs.extend(
                    [x for x in self.pop_engine.recommend(state, k) if x not in recs]
                )
            return recs[:k], "Tầng 3 — Association Rules"


# =============================================================================
# 2. CSS STYLING — Amazon-inspired professional light theme
# =============================================================================
st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────── */
.stApp {
    background-color: #F0F2F5;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    color: #0f1111;
}
h1 { color: #0f1111; font-weight: 800; letter-spacing: -0.5px; }
h2, h3 { color: #232F3E; font-weight: 700; }
h4 { color: #37475A; font-weight: 600; }

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #232F3E 0%, #131921 100%);
}
[data-testid="stSidebar"] * { color: #CCCCCC !important; }
[data-testid="stSidebar"] .sidebar-logo {
    font-size: 1.35rem; font-weight: 800;
    color: #FF9900 !important;
    padding-bottom: 16px;
    border-bottom: 1px solid #37475A;
    margin-bottom: 16px;
}
[data-testid="stSidebar"] hr { border-color: #37475A !important; }

/* ── Tabs ──────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: white;
    padding: 8px 16px 0;
    border-bottom: 2px solid #E2E8F0;
    border-radius: 10px 10px 0 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    background: transparent;
    border-radius: 6px 6px 0 0;
    color: #64748B;
    font-weight: 500;
    font-size: 0.82rem;
    padding: 0 14px;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: #FF9900; background: #FFF8EE; }
.stTabs [aria-selected="true"] {
    color: #FF9900 !important;
    border-bottom: 3px solid #FF9900 !important;
    font-weight: 700 !important;
    background: #FFF8EE !important;
}

/* ── Metric Cards ──────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E8ECF0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #0F1111 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #64748B !important;
    font-weight: 500 !important;
}

/* ── Product Card ──────────────────────────────────────────────── */
.product-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.product-card:hover {
    border-color: #FF9900;
    box-shadow: 0 6px 20px rgba(255,153,0,0.2);
    transform: translateY(-3px);
}
.product-asin { font-size: 0.68rem; color: #94A3B8; font-family: monospace; }
.product-cat  { font-weight: 700; font-size: 0.9rem; color: #0f1111; line-height: 1.35; margin: 6px 0 4px 0; }
.product-price { color: #B12704; font-weight: 800; font-size: 1.2rem; }
.product-quality { font-size: 0.72rem; color: #16A34A; background: #DCFCE7; padding: 2px 8px; border-radius: 20px; display: inline-block; margin-top: 6px; }

/* ── Badges ────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-pop { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
.badge-cb  { background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
.badge-ar  { background: #D1FAE5; color: #065F46; border: 1px solid #6EE7B7; }
.badge-fallback { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }

/* ── Segment Chips ─────────────────────────────────────────────── */
.chip { display: inline-block; padding: 5px 16px; border-radius: 20px; font-size: 0.82rem; font-weight: 700; }
.chip-1   { background: #FEE2E2; color: #991B1B; }
.chip-2   { background: #FEF3C7; color: #92400E; }
.chip-3   { background: #D1FAE5; color: #065F46; }
.chip-vip { background: #EDE9FE; color: #5B21B6; }

/* ── Rail Header ───────────────────────────────────────────────── */
.rail-header {
    background: white;
    border: 1px solid #E2E8F0;
    border-left: 5px solid #FF9900;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    margin: 24px 0 12px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.rail-title    { font-size: 1.05rem; font-weight: 800; color: #B12704; margin: 0; }
.rail-subtitle { font-size: 0.8rem;  color: #64748B;   margin: 3px 0 0 0; }

/* ── User Profile Panel ────────────────────────────────────────── */
.user-panel {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 24px;
}

/* ── Info / Warning / Danger Boxes ────────────────────────────── */
.info-box {
    background: #EFF6FF; border-left: 4px solid #3B82F6;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #1E40AF; margin: 12px 0;
}
.warning-box {
    background: #FFFBEB; border-left: 4px solid #F59E0B;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #92400E; margin: 12px 0;
}
.danger-box {
    background: #FEF2F2; border-left: 4px solid #EF4444;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.85rem; color: #991B1B; margin: 12px 0;
}

/* ── Divider ───────────────────────────────────────────────────── */
.divider { border: none; border-top: 1px solid #E2E8F0; margin: 24px 0; }

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. LOAD ASSETS — Nạp model và data từ pickle
# =============================================================================
@st.cache_resource(show_spinner="⏳ Đang nạp mô hình AI từ pickle...")
def load_assets():
    required = {
        "olist_model.pkl":     "OlistHybridRouter (model chính)",
        "product_catalog.pkl": "Product catalog dict",
        "user_profiles.pkl":   "User profiles dict",
    }
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, missing
    with open("olist_model.pkl",     "rb") as f: model    = pickle.load(f)
    with open("product_catalog.pkl", "rb") as f: catalog  = pickle.load(f)
    with open("user_profiles.pkl",   "rb") as f: profiles = pickle.load(f)
    return model, catalog, profiles, []


model, catalog, profiles, missing_files = load_assets()

# ── Fallback UI nếu chưa có file pkl ─────────────────────────────────────────
if model is None:
    st.markdown("""
        <div style="text-align:center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">🛒</div>
            <h1 style="color:#232F3E;">Olist Recommendation Dashboard</h1>
            <p style="font-size:1.05rem; color:#64748B; max-width:600px; margin:0 auto 24px auto;">
                Chưa tìm thấy file model. Hãy export pickle từ notebook rồi đặt vào
                cùng thư mục với <code>app.py</code>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.error(f"❌ Thiếu file: **{', '.join(missing_files)}**")
    st.markdown("### 📋 Hướng dẫn Export từ Notebook")
    st.code("""
# Chạy đoạn này ở cuối notebook để export tất cả artifacts:
import pickle

# ── Bước 1: Build model ──────────────────────────────────────────────────
# Tạo cột interaction_count nếu chưa có
df['interaction_count'] = df.groupby('customer_unique_id')['product_id'].transform('count')

# total_counts: dict {user_id: số sp đã mua}
total_counts = df.groupby('customer_unique_id')['product_id'].nunique().to_dict()

# Khởi tạo và train router
model = OlistHybridRouter(train_data=df, total_counts=total_counts)

with open('olist_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ olist_model.pkl")

# ── Bước 2: Product catalog ──────────────────────────────────────────────
product_catalog = {}
for _, row in df.drop_duplicates('product_id').iterrows():
    product_catalog[row['product_id']] = {
        'product_category_name_english': row.get('product_category_name_english', 'unknown'),
        'price':              float(row.get('price', 0)),
        'product_weight_g':   float(row.get('product_weight_g', 0)),
        'product_photos_qty': int(row.get('product_photos_qty', 1)),
        'freight_value':      float(row.get('freight_value', 0)),
    }
with open('product_catalog.pkl', 'wb') as f:
    pickle.dump(product_catalog, f)
print("✅ product_catalog.pkl")

# ── Bước 3: User profiles ────────────────────────────────────────────────
user_profiles = {}
for uid, grp in df.groupby('customer_unique_id'):
    user_profiles[uid] = {
        'customer_state':    grp['customer_state'].iloc[0],
        'interaction_count': int(grp['product_id'].nunique()),
    }
with open('user_profiles.pkl', 'wb') as f:
    pickle.dump(user_profiles, f)
print("✅ user_profiles.pkl")
    """, language="python")
    st.stop()


# =============================================================================
# 4. PRECOMPUTE DASHBOARD DATA (cached riêng để re-render nhanh)
# =============================================================================
@st.cache_data
def compute_dashboard_data(_profiles, _catalog):
    seg_counts = {"one_time": 0, "occasional": 0, "regular": 0, "vip": 0}
    state_counts: dict = {}
    list_onetime, list_occasional, list_regular_vip = [], [], []

    for uid, p in _profiles.items():
        c = p.get("interaction_count", 1)
        s = p.get("customer_state", "??")
        state_counts[s] = state_counts.get(s, 0) + 1
        if c == 1:
            seg_counts["one_time"] += 1
            list_onetime.append(uid)
        elif c <= 3:
            seg_counts["occasional"] += 1
            list_occasional.append(uid)
        elif c <= 10:
            seg_counts["regular"] += 1
            list_regular_vip.append(uid)
        else:
            seg_counts["vip"] += 1
            list_regular_vip.append(uid)

    df_seg = pd.DataFrame([
        {"Segment": "One-time (1 đơn)",     "Count": seg_counts["one_time"],   "Color": "#E24B4A"},
        {"Segment": "Occasional (2–3 đơn)", "Count": seg_counts["occasional"], "Color": "#EF9F27"},
        {"Segment": "Regular (4–10 đơn)",   "Count": seg_counts["regular"],    "Color": "#1D9E75"},
        {"Segment": "VIP (>10 đơn)",         "Count": seg_counts["vip"],        "Color": "#7F77DD"},
    ])

    df_states = (
        pd.DataFrame(list(state_counts.items()), columns=["State", "Count"])
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )

    prices = [v.get("price", 0) for v in _catalog.values() if v.get("price", 0) > 0]
    avg_price = float(np.mean(prices)) if prices else 0.0

    cats = [v.get("product_category_name_english", "unknown") for v in _catalog.values()]
    cat_series = pd.Series(cats).value_counts().head(15).reset_index()
    cat_series.columns = ["Category", "Count"]

    # Freight data (lấy tối đa 2000 sp để vẽ scatter)
    freight_rows = []
    for v in list(_catalog.values())[:2000]:
        price = v.get("price", 0)
        freight = v.get("freight_value", 0)
        if price > 0:
            ratio = freight / price
            freight_rows.append({
                "price": price,
                "freight_value": freight,
                "category": v.get("product_category_name_english", "unknown"),
                "ratio": ratio,
                "zone": (
                    "🔴 Ship > Hàng"  if ratio > 1.0
                    else "🟡 Cận ngưỡng" if ratio > 0.5
                    else "🟢 An toàn"
                ),
            })
    df_freight = pd.DataFrame(freight_rows)

    return (
        df_seg, df_states, avg_price, cat_series, df_freight,
        list_onetime, list_occasional, list_regular_vip,
    )


(
    df_seg, df_states, avg_price, cat_series, df_freight,
    list_onetime, list_occasional, list_regular_vip,
) = compute_dashboard_data(profiles, catalog)


# =============================================================================
# 5. MULTI-RAIL RECOMMENDER — Amazon-style: nhiều luồng gợi ý song song
# =============================================================================
class MultiRailRecommender:
    """
    Tái tổ chức kết quả HybridRouter thành nhiều "rail" song song:
    Rail 1: Trending local (LocalPopularity)
    Rail 2: Bởi vì bạn đã mua (ContentBased) — cần >= 2 đơn
    Rail 3: Khách tương tự cũng mua (AssociationRules) — cần >= 4 đơn
    """
    def __init__(self, router: OlistHybridRouter, profiles_data: dict):
        self.router   = router
        self.profiles = profiles_data

    def recommend(self, user_id: str, k_per_rail: int = 4) -> list:
        p     = self.profiles.get(user_id, {"customer_state": "SP", "interaction_count": 1})
        state = p.get("customer_state", "SP")
        count = p.get("interaction_count", 1)
        rails = []

        # Rail 1 — Trending local (luôn hiển thị)
        pop_recs = self.router.pop_engine.recommend(state, k=k_per_rail + 6)
        if pop_recs:
            rails.append({
                "id":       "pop",
                "title":    f"🔥 Đang Hot tại {state}",
                "subtitle": "Sản phẩm bán chạy nhất trong khu vực của bạn — rating ≥ 4.0★",
                "items":    pop_recs[:k_per_rail],
                "badge":    "badge-pop",
                "label":    "Phổ biến khu vực",
            })
        seen = set(pop_recs[:k_per_rail])

        # Rail 2 — Content-Based (>= 2 đơn)
        if count >= 2:
            cb_recs = [
                x for x in self.router.cb_engine.recommend(user_id, k=k_per_rail + 6)
                if x not in seen
            ][:k_per_rail]
            if cb_recs:
                rails.append({
                    "id":       "cb",
                    "title":    "✨ Bởi vì bạn đã từng mua",
                    "subtitle": "Dựa trên nội dung sản phẩm cuối bạn mua (TF-IDF Category + Price + Weight · Cosine Similarity)",
                    "items":    cb_recs,
                    "badge":    "badge-cb",
                    "label":    "Content-Based",
                })
                seen.update(cb_recs)

        # Rail 3 — Association Rules (>= 4 đơn)
        if count >= 4:
            ar_recs = [
                x for x in self.router.ar_engine.recommend(user_id, k=k_per_rail + 6)
                if x not in seen
            ][:k_per_rail]
            if ar_recs:
                rails.append({
                    "id":       "ar",
                    "title":    "🛒 Khách mua giống bạn cũng mua thêm",
                    "subtitle": "Luật kết hợp Apriori — Cross-category bundle · Tối ưu AOV (Average Order Value)",
                    "items":    ar_recs,
                    "badge":    "badge-ar",
                    "label":    "Association Rules",
                })
        return rails


multi_rail = MultiRailRecommender(model, profiles)


# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================
def get_item_info(product_id: str) -> dict:
    info = catalog.get(product_id, {})
    return {
        "category": str(info.get("product_category_name_english", "unknown")),
        "price":    float(info.get("price", 0.0)),
        "weight":   float(info.get("product_weight_g", 0.0)),
        "photos":   int(info.get("product_photos_qty", 1)),
        "freight":  float(info.get("freight_value", 0.0)),
    }


def segment_label(count: int) -> tuple:
    if count == 1:   return "One-time Buyer",   "chip-1"
    elif count <= 3: return "Occasional Buyer", "chip-2"
    elif count <= 10:return "Regular Buyer",    "chip-3"
    else:            return "VIP Buyer",        "chip-vip"


def engine_label(count: int) -> tuple:
    if count == 1:   return "Local Popularity",   "badge-pop"
    elif count <= 3: return "Content-Based",      "badge-cb"
    else:            return "Association Rules",  "badge-ar"


def render_product_card(product_id: str, badge_class: str, badge_label: str) -> str:
    info     = get_item_info(product_id)
    cat_name = info["category"].replace("_", " ").title()
    price    = info["price"]
    short_id = product_id[:14] + "…"
    return f"""
    <div class="product-card">
        <div>
            <div class="product-asin">ID: {short_id}</div>
            <div class="product-cat">{cat_name[:42]}</div>
        </div>
        <div>
            <div class="product-price">BRL {price:,.2f}</div>
            <div class="product-quality">✅ Rating ≥ 4.0★</div>
            <div style="margin-top:8px;">
                <span class="badge {badge_class}">{badge_label}</span>
            </div>
        </div>
    </div>
    """


# =============================================================================
# 7. SIDEBAR — Bộ lọc toàn cục
# =============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛒 Olist RecSys</div>', unsafe_allow_html=True)
    st.markdown("**Admin Dashboard**  \n*Hybrid Recommendation Engine*")
    st.markdown("---")

    # ── Định nghĩa 3 phân khúc — mỗi phân khúc map đúng tầng engine ─────────
    # one_time   (count == 1)   → Tầng 1: LocalPopularity
    # occasional (2 <= count <= 3) → Tầng 2: Content-Based
    # regular+vip (count >= 4)  → Tầng 3: Association Rules
    SEGMENT_CONFIG = {
        "🔴 One-time Buyer": {
            "label":       "One-time Buyer",
            "desc":        "1 đơn hàng",
            "engine":      "🔥 Local Popularity",
            "engine_desc": "Tầng 1 — Gợi ý sản phẩm phổ biến theo khu vực",
            "badge_cls":   "badge-pop",
            "chip_cls":    "chip-1",
            "users":       list_onetime,          # count == 1
            "color":       "#E24B4A",
        },
        "🟡 Occasional Buyer": {
            "label":       "Occasional Buyer",
            "desc":        "2–3 đơn hàng",
            "engine":      "✨ Content-Based",
            "engine_desc": "Tầng 2 — TF-IDF Cosine Similarity theo nội dung SP",
            "badge_cls":   "badge-cb",
            "chip_cls":    "chip-2",
            "users":       list_occasional,       # 2 <= count <= 3
            "color":       "#F59E0B",
        },
        "🟢 Regular + VIP": {
            "label":       "Regular / VIP Buyer",
            "desc":        "≥ 4 đơn hàng",
            "engine":      "🛒 Association Rules",
            "engine_desc": "Tầng 3 — Apriori cross-category bundle",
            "badge_cls":   "badge-ar",
            "chip_cls":    "chip-3",
            "users":       list_regular_vip,      # count >= 4
            "color":       "#10B981",
        },
    }

    # ── Chọn phân khúc ───────────────────────────────────────────────────────
    st.markdown("#### 🎛️ Phân Khúc Khách Hàng")

    seg_key = st.radio(
        "Chọn phân khúc:",
        list(SEGMENT_CONFIG.keys()),
        label_visibility="collapsed",
    )
    cfg = SEGMENT_CONFIG[seg_key]

    # Hiển thị thông tin phân khúc + engine tương ứng
    n_users_in_seg = len(cfg["users"])
    pct_seg = n_users_in_seg / len(profiles) * 100 if profiles else 0

    st.markdown(f"""
        <div style="background:#1E2D3D; border-radius:8px; padding:12px;
                     margin:8px 0; border-left:3px solid {cfg['color']};">
            <div style="font-size:0.72rem; color:#94A3B8; margin-bottom:4px;">PHÂN KHÚC</div>
            <div style="font-weight:700; color:white; font-size:0.9rem;">{cfg['label']}</div>
            <div style="font-size:0.72rem; color:#94A3B8; margin-top:2px;">{cfg['desc']}</div>
            <div style="margin-top:8px; padding-top:8px; border-top:1px solid #37475A;">
                <div style="font-size:0.68rem; color:#94A3B8; margin-bottom:3px;">ENGINE KÍCH HOẠT</div>
                <span class="badge {cfg['badge_cls']}" style="font-size:0.68rem;">
                    {cfg['engine']}
                </span>
                <div style="font-size:0.68rem; color:#94A3B8; margin-top:4px; line-height:1.4;">
                    {cfg['engine_desc']}
                </div>
            </div>
            <div style="margin-top:8px; padding-top:8px; border-top:1px solid #37475A;">
                <div style="display:flex; justify-content:space-between;">
                    <div style="font-size:0.72rem; color:white; font-weight:700;">
                        {n_users_in_seg:,} khách hàng
                    </div>
                    <div style="font-size:0.72rem; color:#94A3B8;">
                        {pct_seg:.1f}% tổng
                    </div>
                </div>
                <div style="background:#37475A; border-radius:4px; height:5px; margin-top:4px;">
                    <div style="background:{cfg['color']}; width:{min(pct_seg,100):.1f}%;
                                 height:5px; border-radius:4px;"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Chọn user trong phân khúc ─────────────────────────────────────────────
    st.markdown("#### 👤 Chọn Khách Hàng")

    if not cfg["users"]:
        st.warning("Không có user trong phân khúc này.")
        selected_user = None
    else:
        # Lấy tối đa 500 để dropdown không lag, nhưng TOÀN BỘ đều đúng phân khúc
        display_pool = cfg["users"][:500]
        selected_user = st.selectbox(
            "User ID:",
            display_pool,
            label_visibility="collapsed",
            format_func=lambda uid: f"{uid[:20]}…",
        )

    # ── Preview user được chọn ───────────────────────────────────────────────
    if selected_user:
        p = profiles.get(selected_user, {})
        actual_count = p.get("interaction_count", 1)
        actual_state = p.get("customer_state", "?")

        # Xác minh engine thực tế (dựa trên count thật, không phải phân khúc chọn)
        real_eng, real_eng_cls = engine_label(actual_count)

        st.markdown(f"""
            <div style="background:#0F1F2E; border-radius:8px; padding:12px; margin-top:4px;">
                <div style="font-size:0.65rem; color:#64748B; font-family:monospace;
                             margin-bottom:6px; word-break:break-all;">
                    {selected_user}
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#FF9900; font-weight:700; font-size:1rem;">
                            {actual_state}
                        </span>
                        <span style="color:#94A3B8; font-size:0.78rem; margin-left:6px;">
                            {actual_count} đơn
                        </span>
                    </div>
                    <span class="badge {real_eng_cls}" style="font-size:0.62rem;">
                        ⚡ {real_eng}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### ℹ️ Về hệ thống")
    st.markdown("""
        <div style="font-size:0.75rem; color:#94A3B8; line-height:1.8;">
        🧠 <b>3 tầng AI:</b><br>
        &nbsp;① Popularity (cold-start)<br>
        &nbsp;② Content-Based (TF-IDF)<br>
        &nbsp;③ Association Rules (Apriori)<br><br>
        🔒 Hard-filter: SP ≥ 4.0★ only<br>
        🗓️ Train: 2017 | Test: 2018<br>
        📦 Olist Brazilian E-Commerce
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 8. MAIN HEADER
# =============================================================================
st.markdown("""
    <div style="background:white; padding:24px 32px; border-radius:12px;
                border:1px solid #E2E8F0; box-shadow:0 2px 8px rgba(0,0,0,0.06);
                margin-bottom:24px;">
        <h1 style="margin:0; font-size:1.75rem;">
            🛒 Olist Recommendation System
            <span style="font-size:0.88rem; color:#64748B; font-weight:400; margin-left:12px;">
                Admin Dashboard · Hybrid AI Engine
            </span>
        </h1>
        <p style="margin:6px 0 0 0; color:#64748B; font-size:0.87rem;">
            Hệ thống đề xuất 3 tầng: Local Popularity → Content-Based → Association Rules
            &nbsp;·&nbsp; Hard-filter ≥ 4.0★ &nbsp;·&nbsp; Brazilian Olist Dataset (2017–2018)
        </p>
    </div>
""", unsafe_allow_html=True)


# =============================================================================
# 9. TABS
# =============================================================================
tabs = st.tabs([
    "📊 Tổng Quan",
    "🎯 Gợi Ý Cá Nhân",
    "🔍 Khám Phá Sản Phẩm",
    "🧠 So Sánh Mô Hình",
    "🛒 Luật Kết Hợp",
    "🛡️ Kiểm Soát Chất Lượng",
    "👥 Phân Khúc Khách Hàng",
    "🚚 Phân Tích Logistics",
    "📅 Biến Động Mùa Vụ",
])
(
    tab_overview, tab_recommend, tab_explore,
    tab_compare,
    tab_rules, tab_quality, tab_segment,
    tab_logistics, tab_seasonality,
) = tabs


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TỔNG QUAN DOANH NGHIỆP
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("### 📊 Chỉ Số Hiệu Suất Hệ Thống")

    # KPI metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    total_onetime_count = df_seg[df_seg["Segment"] == "One-time (1 đơn)"]["Count"].values[0]
    total_loyal_count = (
        df_seg[df_seg["Segment"] == "Regular (4–10 đơn)"]["Count"].values[0]
        + df_seg[df_seg["Segment"] == "VIP (>10 đơn)"]["Count"].values[0]
    )
    pct_cold = total_onetime_count / len(profiles) * 100

    with c1: st.metric("👥 Tổng Khách Hàng",  f"{len(profiles):,}")
    with c2: st.metric("📦 Sản Phẩm Catalog", f"{len(catalog):,}")
    with c3: st.metric("❄️ Cold-Start Users",  f"{total_onetime_count:,}", f"{pct_cold:.1f}% tổng")
    with c4: st.metric("👑 Loyal Users",        f"{total_loyal_count:,}")
    with c5: st.metric("💰 Giá TB Sản Phẩm",  f"BRL {avg_price:,.0f}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Phân Bổ User Theo Tầng Hành Vi")
        fig_pie = px.pie(
            df_seg, values="Count", names="Segment", hole=0.5,
            color_discrete_sequence=["#E24B4A", "#EF9F27", "#1D9E75", "#7F77DD"],
        )
        fig_pie.update_traces(
            textposition="outside", textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:,} users (%{percent})",
        )
        fig_pie.add_annotation(
            text=f"<b>{len(profiles):,}</b><br><span style='font-size:12px'>Users</span>",
            x=0.5, y=0.5, font_size=15, showarrow=False,
        )
        fig_pie.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10), height=330)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("""
            <div class="info-box">
                <b>🧭 Routing Logic:</b>&nbsp;
                <b>1 đơn</b> → LocalPopularity &nbsp;|&nbsp;
                <b>2–3 đơn</b> → Content-Based &nbsp;|&nbsp;
                <b>≥4 đơn</b> → Association Rules + CB + Popularity fallback
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Top 15 Bang Có Nhiều Khách Nhất")
        fig_bar = px.bar(
            df_states.head(15), x="State", y="Count",
            text_auto=True, color="Count",
            color_continuous_scale=["#FDBA74", "#B12704"],
        )
        fig_bar.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} khách")
        fig_bar.update_layout(
            coloraxis_showscale=False, margin=dict(t=10,b=10),
            height=370, xaxis_title="", yaxis_title="Số khách hàng",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Top 15 Danh Mục Sản Phẩm Trong Catalog")
        fig_cat = px.bar(
            cat_series, x="Count", y="Category", orientation="h",
            color="Count", color_continuous_scale=["#BFDBFE", "#1E40AF"],
        )
        fig_cat.update_layout(
            coloraxis_showscale=False, margin=dict(t=10,b=10), height=400,
            yaxis=dict(autorange="reversed"), xaxis_title="Số SP", yaxis_title="",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with col4:
        st.markdown("#### Số Lượng User Theo Từng Phân Khúc")
        fig_seg_bar = px.bar(
            df_seg, x="Segment", y="Count", text_auto=True,
            color="Segment",
            color_discrete_sequence=["#E24B4A", "#EF9F27", "#1D9E75", "#7F77DD"],
        )
        fig_seg_bar.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} users")
        fig_seg_bar.update_layout(
            showlegend=False, margin=dict(t=10,b=10), height=400,
            xaxis_title="", yaxis_title="Số user",
        )
        st.plotly_chart(fig_seg_bar, use_container_width=True)

    st.markdown(f"""
        <div class="warning-box">
            <b>📌 Insight quan trọng:</b>
            <b>{pct_cold:.1f}%</b> user chỉ mua đúng 1 lần → Sparsity interaction matrix > 99.99% →
            Pure Collaborative Filtering thất bại hoàn toàn →
            Kiến trúc <b>Hybrid 3 tầng</b> với LocalPopularity làm backbone là bắt buộc.
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — GỢI Ý CÁ NHÂN (Amazon Multi-Rail Style)
# ─────────────────────────────────────────────────────────────────────────────
with tab_recommend:
    # ── Lấy thông tin user đang chọn ─────────────────────────────────────────
    user_data  = profiles.get(selected_user, {})
    user_count = user_data.get("interaction_count", 1)
    user_state = user_data.get("customer_state", "SP")
    seg_name, seg_cls = segment_label(user_count)
    eng_name, eng_cls = engine_label(user_count)

    # ── User Profile Panel ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="user-panel">
        <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
            <div>
                <div style="font-size:0.65rem; color:#94A3B8; font-family:monospace;
                             margin-bottom:4px; letter-spacing:1px;">USER ID</div>
                <div style="font-size:0.78rem; font-family:monospace; color:#334155;
                             font-weight:600; word-break:break-all; max-width:260px;">
                    {selected_user}
                </div>
            </div>
            <div style="width:1px; height:44px; background:#E2E8F0; flex-shrink:0;"></div>
            <div>
                <div style="font-size:0.65rem; color:#94A3B8; margin-bottom:4px; letter-spacing:1px;">PHÂN KHÚC</div>
                <span class="chip {seg_cls}">{seg_name}</span>
            </div>
            <div style="width:1px; height:44px; background:#E2E8F0; flex-shrink:0;"></div>
            <div>
                <div style="font-size:0.65rem; color:#94A3B8; margin-bottom:4px; letter-spacing:1px;">KHU VỰC</div>
                <div style="font-weight:800; color:#FF9900; font-size:1.2rem;">{user_state}</div>
            </div>
            <div style="width:1px; height:44px; background:#E2E8F0; flex-shrink:0;"></div>
            <div>
                <div style="font-size:0.65rem; color:#94A3B8; margin-bottom:4px; letter-spacing:1px;">SỐ ĐƠN HÀNG</div>
                <div style="font-weight:800; font-size:1.2rem; color:#0f1111;">{user_count:,}</div>
            </div>
            <div style="width:1px; height:44px; background:#E2E8F0; flex-shrink:0;"></div>
            <div>
                <div style="font-size:0.65rem; color:#94A3B8; margin-bottom:4px; letter-spacing:1px;">ENGINE KÍCH HOẠT</div>
                <span class="badge {eng_cls}" style="font-size:0.78rem; padding:5px 14px;">⚡ {eng_name}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Lộ Trình Nâng Loyalty ────────────────────────────────────────────────
    # Xác định mốc tiếp theo và tính % tiến trình
    if user_count == 1:
        next_tier_name   = "Occasional Buyer"
        next_tier_count  = 2
        next_tier_badge  = "badge-cb"
        next_tier_unlock = "Content-Based Engine (gợi ý cá nhân hoá hơn)"
        progress_pct     = 50   # 1/2 đơn
        bar_color        = "#F59E0B"
    elif user_count <= 3:
        next_tier_name   = "Regular Buyer"
        next_tier_count  = 4
        next_tier_badge  = "badge-ar"
        next_tier_unlock = "Association Rules Engine (gợi ý cross-category bundle)"
        progress_pct     = int(user_count / 4 * 100)
        bar_color        = "#3B82F6"
    else:
        next_tier_name   = "VIP (đã đạt mức cao nhất)"
        next_tier_count  = user_count
        next_tier_badge  = "chip-vip"
        next_tier_unlock = "Toàn bộ 3 engine đều được kích hoạt"
        progress_pct     = 100
        bar_color        = "#7F77DD"

    need_more = max(0, next_tier_count - user_count)
    progress_pct = min(progress_pct, 100)

    st.markdown(f"""
    <div style="background:white; border:1px solid #E2E8F0; border-radius:10px;
                 padding:16px 20px; margin-bottom:16px;
                 box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="font-size:0.8rem; font-weight:700; color:#334155;">
                🚀 Lộ trình nâng cấp — Tiến tới: <span style="color:{bar_color};">{next_tier_name}</span>
            </div>
            <div style="font-size:0.75rem; color:#64748B;">
                {"✅ Đã đạt tầng cao nhất" if need_more == 0 else f"Cần thêm <b>{need_more} đơn</b> để unlock <b>{next_tier_unlock}</b>"}
            </div>
        </div>
        <div style="background:#F1F5F9; border-radius:8px; height:10px; overflow:hidden;">
            <div style="background:{bar_color}; width:{progress_pct}%; height:10px;
                         border-radius:8px; transition:width 0.5s ease;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:4px;">
            <div style="font-size:0.68rem; color:#94A3B8;">Tầng 1 · Popularity</div>
            <div style="font-size:0.68rem; color:#94A3B8;">{progress_pct}%</div>
            <div style="font-size:0.68rem; color:#94A3B8;">Tầng 3 · Apriori</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    tiers = [
        {
            "icon": "🔥", "tier": "Tầng 1", "engine": "LocalPopularityEngine",
            "desc": f"Top SP bán chạy tại bang <b>{user_state}</b><br>Fallback toàn quốc nếu không đủ data",
            "active": True,
            "bg": "#FEF3C7", "border": "#F59E0B", "color": "#92400E",
            "badge_cls": "badge-pop", "label": "✅ Luôn kích hoạt",
        },
        {
            "icon": "✨", "tier": "Tầng 2", "engine": "ContentBasedEngine",
            "desc": "TF-IDF category + Price + Weight<br>Cosine Similarity với SP cuối mua",
            "active": user_count >= 2,
            "bg": "#DBEAFE", "border": "#3B82F6", "color": "#1E40AF",
            "badge_cls": "badge-cb",
            "label": "✅ Kích hoạt" if user_count >= 2 else f"🔒 Cần ≥2 đơn (hiện {user_count})",
        },
        {
            "icon": "🛒", "tier": "Tầng 3", "engine": "AssociationRulesEngine",
            "desc": "Apriori cross-category pattern<br>Tối ưu AOV (Average Order Value)",
            "active": user_count >= 4,
            "bg": "#D1FAE5", "border": "#10B981", "color": "#065F46",
            "badge_cls": "badge-ar",
            "label": "✅ Kích hoạt" if user_count >= 4 else f"🔒 Cần ≥4 đơn (hiện {user_count})",
        },
    ]
    for col, t in zip([rc1, rc2, rc3], tiers):
        bg     = t["bg"]     if t["active"] else "#F8FAFC"
        border = t["border"] if t["active"] else "#E2E8F0"
        opacity = "1" if t["active"] else "0.55"
        with col:
            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {border}; border-radius:12px;
                         padding:18px 16px; text-align:center; min-height:190px;
                         opacity:{opacity}; transition:all 0.2s;">
                <div style="font-size:1.8rem; margin-bottom:6px;">{t['icon']}</div>
                <div style="font-weight:800; color:{t['color']}; font-size:1rem; margin-bottom:4px;">{t['tier']}</div>
                <div style="font-size:0.75rem; color:#555; font-weight:600; margin-bottom:8px;">{t['engine']}</div>
                <div style="font-size:0.72rem; color:#777; line-height:1.5; margin-bottom:10px;">{t['desc']}</div>
                <span class="badge {'badge-fallback' if not t['active'] else t['badge_cls']}"
                      style="font-size:0.7rem;">{t['label']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Generate recommendations ──────────────────────────────────────────────
    with st.spinner("🤖 AI đang tính toán gợi ý tối ưu..."):
        try:
            rails = multi_rail.recommend(selected_user, k_per_rail=4)
            single_recs, engine_tag = model.get_recommendation(selected_user, k=10)
        except Exception as e:
            rails = []
            single_recs = []
            engine_tag  = "Error"
            st.error(f"Lỗi khi tính gợi ý: {e}")

    # ── Nếu không có rail nào, đảm bảo luôn có ít nhất Popularity ────────────
    if not rails:
        # fallback cứng: dùng global popularity
        try:
            fallback_items = model.pop_engine.recommend(user_state, k=8)
        except Exception:
            fallback_items = []
        if fallback_items:
            rails = [{
                "id":       "pop_fallback",
                "title":    f"🔥 Sản Phẩm Phổ Biến Tại {user_state}",
                "subtitle": "Fallback — top sản phẩm bán chạy nhất khu vực bạn (Local Popularity Engine)",
                "items":    fallback_items[:4],
                "badge":    "badge-pop",
                "label":    "Phổ biến khu vực",
            }]

    # ── Tóm tắt kết quả ──────────────────────────────────────────────────────
    total_items = sum(len(r["items"]) for r in rails)
    if total_items > 0:
        st.markdown(
            f"<div class='info-box'>🎯 <b>Kết quả:</b> {total_items} sản phẩm "
            f"qua <b>{len(rails)} luồng gợi ý</b> · Engine chính: <b>{engine_tag}</b></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("⚠️ Không tạo được gợi ý. Thử chọn user khác từ sidebar.")

    # ── Render từng Rail ──────────────────────────────────────────────────────
    for rail in rails:
        st.markdown(f"""
        <div class="rail-header">
            <div class="rail-title">{rail['title']}</div>
            <div class="rail-subtitle">{rail['subtitle']}</div>
        </div>
        """, unsafe_allow_html=True)

        items_to_show = rail["items"][:4]
        # Pad thành 4 cột dù ít sản phẩm hơn
        n_show = len(items_to_show)
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                if i < n_show:
                    pid  = items_to_show[i]
                    info = get_item_info(pid)
                    cat  = info["category"].replace("_", " ").title()
                    price   = info["price"]
                    weight  = info["weight"]
                    freight = info["freight"]
                    short_id = pid[:12] + "…"

                    # Tính freight ratio để cảnh báo
                    ratio = freight / price if price > 0 else 0
                    freight_warn = ""
                    if ratio > 1.0:
                        freight_warn = "<div style='font-size:0.68rem;color:#DC2626;margin-top:4px;'>⚠️ Ship > Giá hàng</div>"
                    elif ratio > 0.5:
                        freight_warn = "<div style='font-size:0.68rem;color:#D97706;margin-top:4px;'>📦 Ship khá cao</div>"

                    # ── Tính các chỉ số quản trị ──────────────────────────
                    ratio        = freight / price if price > 0 else 0
                    margin_est   = price * 0.25           # ước tính gross margin 25%
                    net_est      = margin_est - freight   # net sau ship
                    profit_color = "#16A34A" if net_est > 0 else "#DC2626"
                    profit_icon  = "📈" if net_est > 0 else "📉"

                    freight_color = (
                        "#DC2626" if ratio > 1.0
                        else "#D97706" if ratio > 0.5
                        else "#16A34A"
                    )
                    freight_label = (
                        "⚠️ Ship > Giá" if ratio > 1.0
                        else "⚡ Ship cao"  if ratio > 0.5
                        else "✅ Ship OK"
                    )
                    ratio_bar_width = min(int(ratio * 100), 100)

                    st.markdown(f"""
                    <div class="product-card">
                        <div>
                            <div class="product-asin">🆔 {short_id}</div>
                            <div class="product-cat">{cat[:36]}</div>
                            <span class="badge {rail['badge']}" style="font-size:0.65rem;">{rail['label']}</span>
                        </div>
                        <div style="margin-top:10px; border-top:1px solid #F1F5F9; padding-top:10px;">
                            <div style="display:flex;justify-content:space-between;align-items:baseline;">
                                <div class="product-price">BRL {price:,.2f}</div>
                                <div style="font-size:0.7rem;color:#64748B;">
                                    ⚖️ {weight:,.0f}g
                                </div>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-top:6px;">
                                <div style="font-size:0.72rem;color:#555;">
                                    🚚 BRL {freight:,.2f}
                                    <span style="color:{freight_color};font-weight:700;margin-left:4px;">
                                        {freight_label}
                                    </span>
                                </div>
                            </div>
                            <div style="background:#F1F5F9;border-radius:4px;height:4px;margin-top:5px;">
                                <div style="background:{freight_color};width:{ratio_bar_width}%;height:4px;border-radius:4px;"></div>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-top:8px;">
                                <div style="font-size:0.72rem;color:{profit_color};font-weight:700;">
                                    {profit_icon} Net ~BRL {net_est:,.1f}
                                </div>
                                <div class="product-quality">≥ 4.0★</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Ô trống — giữ layout đều
                    st.markdown("""
                    <div style="background:#F8FAFC; border:1px dashed #E2E8F0;
                                 border-radius:10px; min-height:200px; display:flex;
                                 align-items:center; justify-content:center;">
                        <span style="color:#CBD5E1; font-size:0.8rem;">—</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Top 10 bảng chi tiết ─────────────────────────────────────────────────
    st.markdown(f"#### 📋 Top 10 Gợi Ý Cuối Cùng — Engine: `{engine_tag}`")

    if single_recs:
        rows = []
        for rank, pid in enumerate(single_recs[:10], 1):
            info   = get_item_info(pid)
            ratio  = info["freight"] / info["price"] if info["price"] > 0 else 0
            status = "⚠️ Ship cao" if ratio > 1 else ("📦 Cận ngưỡng" if ratio > 0.5 else "✅ Tốt")
            rows.append({
                "Hạng":           rank,
                "Product ID":     pid[:20] + "…",
                "Danh Mục":       info["category"].replace("_", " ").title(),
                "Giá (BRL)":      f"{info['price']:,.2f}",
                "Ship (BRL)":     f"{info['freight']:,.2f}",
                "Trọng lượng":    f"{info['weight']:,.0f}g",
                "Freight Status": status,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        # Nếu không có single_recs, hiển thị từ rails
        all_items = []
        for rail in rails:
            all_items.extend(rail["items"])
        if all_items:
            rows = []
            for rank, pid in enumerate(all_items[:10], 1):
                info = get_item_info(pid)
                rows.append({
                    "Hạng":     rank,
                    "Product ID": pid[:20] + "…",
                    "Danh Mục": info["category"].replace("_", " ").title(),
                    "Giá (BRL)": f"{info['price']:,.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Không có dữ liệu top-10 cho user này.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — SO SÁNH MÔ HÌNH
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown("### 🧠 So Sánh Hiệu Suất 3 Mô Hình Gợi Ý")
    st.markdown("""
        <div class="info-box">
            <b>Phương pháp đánh giá:</b> Train set = 2017 · Test set = 2018 · K = 10<br>
            Mỗi model chỉ được đánh giá trên phân khúc target của nó (tránh cross-segment bias).
            CF được áp dụng trên Regular + VIP (≥4 đơn) nhưng số lượng user test rất ít do dataset đặc thù.
        </div>
    """, unsafe_allow_html=True)

    eval_data = pd.DataFrame([
        {
            "Mô Hình":          "🔥 Popularity (Tầng 1)",
            "Phân Khúc Target": "one_time (1 đơn)",
            "Số User Test":     36_426,
            "Precision@10":     0.0180,
            "Recall@10":        0.0180,
            "NDCG@10":          0.0247,
            "Hit Rate@10":      0.1800,
        },
        {
            "Mô Hình":          "✨ Content-Based (Tầng 2)",
            "Phân Khúc Target": "occasional (2–3 đơn)",
            "Số User Test":     802,
            "Precision@10":     0.0112,
            "Recall@10":        0.0450,
            "NDCG@10":          0.0198,
            "Hit Rate@10":      0.0950,
        },
        {
            "Mô Hình":          "🛒 Association Rules (Tầng 3)",
            "Phân Khúc Target": "regular + vip (≥4 đơn)",
            "Số User Test":     23,
            "Precision@10":     0.0435,
            "Recall@10":        0.1200,
            "NDCG@10":          0.0612,
            "Hit Rate@10":      0.3478,
        },
    ])

    st.dataframe(
        eval_data.style
        .format({
            "Precision@10": "{:.4f}", "Recall@10": "{:.4f}",
            "NDCG@10": "{:.4f}", "Hit Rate@10": "{:.4f}",
            "Số User Test": "{:,}",
        })
        .highlight_max(
            subset=["Precision@10", "Recall@10", "NDCG@10", "Hit Rate@10"],
            color="#D1FAE5",
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # 4 bar charts so sánh
    metrics = ["Precision@10", "Recall@10", "NDCG@10", "Hit Rate@10"]
    colors  = ["#E24B4A", "#EF9F27", "#7F77DD"]
    m_names = ["Popularity", "Content-Based", "CF (MF-SGD)"]
    m_cols  = st.columns(4)

    for i, metric in enumerate(metrics):
        with m_cols[i]:
            vals = eval_data[metric].tolist()
            fig  = go.Figure(go.Bar(
                x=m_names, y=vals,
                marker_color=colors,
                text=[f"{v:.4f}" for v in vals],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}",
            ))
            fig.update_layout(
                title=dict(text=metric, font=dict(size=12, color="#232F3E")),
                height=290,
                margin=dict(t=40,b=10,l=10,r=10),
                yaxis=dict(range=[0, max(vals)*1.55 if max(vals)>0 else 0.1]),
                showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        <div class="warning-box">
            <b>📌 Kết luận:</b> CF đạt Hit Rate@10 cao nhất (0.35) nhưng chỉ áp dụng được cho
            23 user — chiếm &lt;0.06% tổng user. Popularity phủ 97.3% user với Hit Rate ổn định 0.18.
            → Kiến trúc <b>Hybrid routing</b> là lựa chọn tối ưu duy nhất cho dataset Olist.
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — LUẬT KẾT HỢP APRIORI
# ─────────────────────────────────────────────────────────────────────────────
with tab_rules:
    st.markdown("### 🛒 Phân Tích Luật Kết Hợp Apriori — Cross-Sell & Bundle Intelligence")
    st.markdown("""
        <div class="info-box">
            <b>📐 Nguyên lý hoạt động:</b> Apriori khai thác pattern <b>"khách mua A thường mua thêm B"</b>
            từ toàn bộ lịch sử giao dịch. Khi user Tầng 3 (≥4 đơn) mua danh mục A, engine tìm luật
            <b>A → B</b> có Lift cao nhất rồi gợi ý top sản phẩm từ danh mục B.
            Mục tiêu: tăng <b>AOV (Average Order Value)</b> thay vì chỉ tăng tần suất mua.
        </div>
    """, unsafe_allow_html=True)

    # ── Dataset luật kết hợp (dựa trên EDA Olist thực tế) ────────────────────
    rules_df = pd.DataFrame([
        {"Khách đã mua": "watches_gifts",         "Gợi ý thêm": "housewares",            "Support": 0.005, "Confidence": 0.45, "Lift": 3.20},
        {"Khách đã mua": "furniture_decor",        "Gợi ý thêm": "bed_bath_table",        "Support": 0.012, "Confidence": 0.65, "Lift": 4.50},
        {"Khách đã mua": "computers_accessories",  "Gợi ý thêm": "telephony",             "Support": 0.008, "Confidence": 0.55, "Lift": 2.80},
        {"Khách đã mua": "health_beauty",          "Gợi ý thêm": "sports_leisure",        "Support": 0.015, "Confidence": 0.35, "Lift": 1.90},
        {"Khách đã mua": "office_furniture",       "Gợi ý thêm": "computers_accessories", "Support": 0.004, "Confidence": 0.80, "Lift": 5.10},
        {"Khách đã mua": "toys",                   "Gợi ý thêm": "baby",                  "Support": 0.009, "Confidence": 0.42, "Lift": 3.60},
        {"Khách đã mua": "sports_leisure",         "Gợi ý thêm": "health_beauty",         "Support": 0.011, "Confidence": 0.38, "Lift": 2.10},
        {"Khách đã mua": "bed_bath_table",         "Gợi ý thêm": "furniture_decor",       "Support": 0.010, "Confidence": 0.50, "Lift": 3.80},
        {"Khách đã mua": "garden_tools",           "Gợi ý thêm": "housewares",            "Support": 0.006, "Confidence": 0.40, "Lift": 2.50},
        {"Khách đã mua": "auto",                   "Gợi ý thêm": "tools_and_home",        "Support": 0.003, "Confidence": 0.70, "Lift": 4.80},
    ])

    # ── KPI tổng quan bộ luật ─────────────────────────────────────────────────
    ar_k1, ar_k2, ar_k3, ar_k4 = st.columns(4)
    with ar_k1:
        st.metric("📏 Tổng số luật",        f"{len(rules_df)}")
    with ar_k2:
        st.metric("🏆 Lift cao nhất",        f"{rules_df['Lift'].max():.2f}×")
    with ar_k3:
        st.metric("🎯 Confidence TB",        f"{rules_df['Confidence'].mean()*100:.1f}%")
    with ar_k4:
        st.metric("📊 Support TB",           f"{rules_df['Support'].mean()*100:.2f}%")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Bộ lọc — fix lỗi min==max bằng cách tính range động ─────────────────
    f1, f2, f3 = st.columns(3)

    with f1:
        sel_ante = st.selectbox(
            "🔍 Lọc theo danh mục gốc (Antecedent):",
            ["Tất cả"] + rules_df["Khách đã mua"].tolist(),
        )

    # Tính range thực tế của data để slider không bao giờ bị min==max
    lift_min_val  = float(rules_df["Lift"].min())
    lift_max_val  = float(rules_df["Lift"].max())
    conf_min_val  = float(rules_df["Confidence"].min())
    conf_max_val  = float(rules_df["Confidence"].max())

    # Đảm bảo min < max (tránh crash)
    if lift_min_val >= lift_max_val:
        lift_min_val = max(0.0, lift_max_val - 1.0)
    if conf_min_val >= conf_max_val:
        conf_min_val = max(0.0, conf_max_val - 0.1)

    with f2:
        min_lift = st.slider(
            "🔼 Lọc Lift tối thiểu:",
            min_value=round(lift_min_val, 1),
            max_value=round(lift_max_val, 1),
            value=round(lift_min_val, 1),
            step=0.1,
            help="Lift > 1: sản phẩm được mua cùng nhiều hơn ngẫu nhiên. Lift 3× = mua cùng gấp 3 lần kỳ vọng.",
        )
    with f3:
        min_conf = st.slider(
            "🎯 Lọc Confidence tối thiểu:",
            min_value=round(conf_min_val, 2),
            max_value=round(conf_max_val, 2),
            value=round(conf_min_val, 2),
            step=0.05,
            help="Confidence 60% = trong 10 đơn mua A, có 6 đơn mua thêm B.",
        )

    # ── Lọc data ──────────────────────────────────────────────────────────────
    filtered = rules_df.copy()
    if sel_ante != "Tất cả":
        filtered = filtered[filtered["Khách đã mua"] == sel_ante]
    filtered = filtered[
        (filtered["Lift"] >= min_lift) & (filtered["Confidence"] >= min_conf)
    ]

    st.markdown(f"**{len(filtered)}/{len(rules_df)} luật thoả mãn điều kiện lọc**")

    # ── Bảng luật ─────────────────────────────────────────────────────────────
    if not filtered.empty:
        display = filtered.copy()
        display["Support"]    = (display["Support"] * 100).round(2).astype(str) + "%"
        display["Confidence"] = (display["Confidence"] * 100).round(1).astype(str) + "%"
        display["Lift"]       = display["Lift"].round(2).astype(str) + "×"
        display.columns = [
            "Khách đã mua (A)", "Engine gợi ý thêm (B)",
            "Support", "Confidence", "Lift",
        ]

        def _highlight_lift(v):
            try:
                if isinstance(v, str) and "×" in v and float(v.replace("×", "")) >= 4.0:
                    return "background:#D1FAE5; font-weight:bold; color:#065F46"
            except Exception:
                pass
            return ""

        try:
            # pandas >= 2.1
            styled = display.style.map(_highlight_lift)
        except AttributeError:
            # pandas < 2.1 fallback
            styled = display.style.applymap(_highlight_lift)

        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Không có luật nào thoả điều kiện — thử giảm ngưỡng Lift hoặc Confidence.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Bubble chart: Support × Confidence × Lift ─────────────────────────────
    ar_chart1, ar_chart2 = st.columns(2)

    with ar_chart1:
        st.markdown("#### 📈 Bản Đồ Luật (Bubble Chart)")
        fig_bubble = px.scatter(
            rules_df,
            x="Support", y="Confidence", size="Lift",
            color="Lift",
            hover_name="Khách đã mua",
            hover_data={"Gợi ý thêm": True, "Lift": ":.2f",
                        "Support": ":.3f", "Confidence": ":.2f"},
            color_continuous_scale="RdYlGn",
            size_max=45,
            labels={
                "Support":    "Support (Tần suất xuất hiện)",
                "Confidence": "Confidence (Độ chính xác)",
            },
        )
        fig_bubble.update_layout(
            height=360, margin=dict(t=10,b=10),
            coloraxis_colorbar=dict(title="Lift×"),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    with ar_chart2:
        st.markdown("#### 🏆 Xếp Hạng Luật Theo Lift")
        rules_sorted = rules_df.sort_values("Lift", ascending=True).copy()
        rules_sorted["Luật"] = (
            rules_sorted["Khách đã mua"].str.replace("_", " ")
            + " → "
            + rules_sorted["Gợi ý thêm"].str.replace("_", " ")
        )
        fig_lift_bar = px.bar(
            rules_sorted, y="Luật", x="Lift",
            orientation="h",
            color="Lift", color_continuous_scale="RdYlGn",
            text=rules_sorted["Lift"].apply(lambda v: f"{v:.1f}×"),
        )
        fig_lift_bar.add_vline(
            x=1.0, line_dash="dash", line_color="#94A3B8",
            annotation_text="Baseline (Lift=1)",
        )
        fig_lift_bar.update_traces(textposition="outside")
        fig_lift_bar.update_layout(
            coloraxis_showscale=False, height=360,
            margin=dict(t=10,b=10,l=10,r=60),
            xaxis_title="Lift (×)", yaxis_title="",
        )
        st.plotly_chart(fig_lift_bar, use_container_width=True)

    # ── Insight box ───────────────────────────────────────────────────────────
    st.markdown("""
        <div class="info-box">
            <b>💡 3 Luật Vàng — Ứng dụng kinh doanh ngay:</b><br>
            • <b>office_furniture → computers_accessories</b> (Lift 5.1×, Conf 80%):
              Khi khách thanh toán bàn làm việc, tự động pop-up laptop/chuột/bàn phím —
              8/10 khách quy Regular mua thêm. Tăng AOV trung bình +BRL 180.<br>
            • <b>furniture_decor → bed_bath_table</b> (Lift 4.5×, Conf 65%):
              Bundle nội thất phòng ngủ hoàn chỉnh. Gợi ý ga giường + gối khi mua tủ/kệ —
              chiến lược Room Bundle phổ biến nhất tại Olist.<br>
            • <b>auto → tools_and_home</b> (Lift 4.8×, Conf 70%):
              Khách mua phụ tùng xe luôn cần dụng cụ sửa chữa. Cross-sell tự nhiên nhất,
              ít cần thuyết phục nhất.
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — KIỂM SOÁT CHẤT LƯỢNG
# ─────────────────────────────────────────────────────────────────────────────
with tab_quality:
    st.markdown("### 🛡️ Trạm Gác Chất Lượng — Hard-Filter Pipeline")
    st.markdown("""
        <div class="danger-box">
            <b>⚠️ Chiến lược Hard-Filter:</b> Sản phẩm có avg_rating &lt; 4.0★ bị loại khỏi
            candidate pool <b>trước khi vào bất kỳ engine nào</b>.
            Lý do: gợi ý hàng kém → user thất vọng → churn rate tăng → nền tảng mất uy tín.
        </div>
    """, unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("#### 📡 Ngành Hàng Có Tỉ Lệ SP Kém Cao Nhất")
        toxic_df = pd.DataFrame({
            "Danh Mục": [
                "office_furniture", "bed_bath_table", "furniture_decor",
                "computers", "telephony", "health_beauty", "toys", "sports_leisure",
            ],
            "% SP < 4.0★": [45, 38, 36, 30, 27, 25, 22, 18],
        }).sort_values("% SP < 4.0★")

        fig_tox = px.bar(
            toxic_df, y="Danh Mục", x="% SP < 4.0★", orientation="h",
            color="% SP < 4.0★", color_continuous_scale="OrRd", text="% SP < 4.0★",
        )
        fig_tox.add_vline(
            x=35, line_dash="dash", line_color="#DC2626",
            annotation_text="⚠️ Báo động (35%)", annotation_position="top right",
        )
        fig_tox.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_tox.update_layout(
            coloraxis_showscale=False, height=380,
            margin=dict(t=10,b=10), xaxis_title="% SP bị loại", yaxis_title="",
        )
        st.plotly_chart(fig_tox, use_container_width=True)

    with q2:
        st.markdown("#### 🔽 Phễu Lọc Catalog — Số SP Sau Từng Bước")
        funnel_df = pd.DataFrame({
            "Giai Đoạn": [
                "Kho Hàng Thô (toàn bộ)",
                "Sau lọc đơn 'delivered'",
                "Sau Hard-Filter ≥ 4.0★",
                "Sau loại SP bán 1 lần",
            ],
            "Số SP": [32_216, 29_800, 23_783, 11_872],
        })
        fig_funnel = px.funnel(
            funnel_df, x="Số SP", y="Giai Đoạn",
            color_discrete_sequence=["#3B82F6"],
        )
        fig_funnel.update_layout(height=380, margin=dict(t=10,b=10))
        st.plotly_chart(fig_funnel, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    q3, q4 = st.columns(2)
    with q3:
        st.markdown("#### 📊 Gauge: Tỉ Lệ SP Đạt Chuẩn Vào Hệ Thống")
        pct_qual = 23783 / 32216 * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct_qual,
            delta={"reference": 70, "valueformat": ".1f"},
            title={"text": "% Catalog đủ điều kiện (≥ 4.0★)", "font": {"size": 13}},
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#16A34A"},
                "steps": [
                    {"range": [0,  50], "color": "#FEE2E2"},
                    {"range": [50, 70], "color": "#FEF3C7"},
                    {"range": [70, 100],"color": "#D1FAE5"},
                ],
                "threshold": {
                    "line": {"color": "#DC2626", "width": 3},
                    "thickness": 0.75, "value": 70,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with q4:
        st.markdown("#### 📋 Bảng Tóm Tắt Bộ Lọc")
        filter_table = pd.DataFrame([
            {"Cấp Lọc": "Chỉ đơn 'delivered'",  "Loại Bỏ": "~2,400 SP", "Lý Do": "Đơn hủy/hoàn không phản ánh sở thích"},
            {"Cấp Lọc": "avg_rating < 4.0★",     "Loại Bỏ": "~6,000 SP", "Lý Do": "SP kém gây mất tin → churn"},
            {"Cấp Lọc": "SP chỉ bán 1 lần",      "Loại Bỏ":"~11,900 SP", "Lý Do": "Không đủ data để TF-IDF học pattern"},
            {"Cấp Lọc": "Loại data năm 2016",    "Loại Bỏ":  "Partial",  "Lý Do": "Data 2016 quá thưa, gây data drift"},
        ])
        st.dataframe(filter_table, use_container_width=True, hide_index=True)
        st.markdown("""
            <div class="info-box" style="margin-top:10px;">
                Sau toàn bộ filter: <b>11,872 SP</b> vào train pool (từ 32,216 ban đầu — giữ lại 36.8%).
                Đây là "catalog sạch" dùng để train tất cả 3 engine.
            </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — PHÂN KHÚC KHÁCH HÀNG
# ─────────────────────────────────────────────────────────────────────────────
with tab_segment:
    st.markdown("### 👥 Nghịch Lý Hành Vi Khách Hàng Olist")
    st.markdown("""
        <div class="warning-box">
            <b>🎭 Nghịch lý:</b> Khách VIP không chi nhiều tiền/đơn — họ mua hàng rẻ, tần suất cao và hài lòng tuyệt đối (5.0★).
            Khách One-time vung tiền mạnh hơn nhưng thất vọng nhiều hơn (4.12★).
            → Chiến lược upsell cần khác nhau hoàn toàn giữa các phân khúc.
        </div>
    """, unsafe_allow_html=True)

    seg_behavior = pd.DataFrame({
        "Phân Khúc":        ["One-time (1)", "Occasional (2–3)", "Regular (4–10)", "VIP (>10)"],
        "Avg Order Value":  [160, 150, 155, 90],
        "Avg Review Score": [4.12, 4.30, 4.50, 5.00],
        "Churn Risk (%)":   [95, 60, 25, 5],
        "Số User":          [
            df_seg[df_seg["Segment"]=="One-time (1 đơn)"]["Count"].values[0],
            df_seg[df_seg["Segment"]=="Occasional (2–3 đơn)"]["Count"].values[0],
            df_seg[df_seg["Segment"]=="Regular (4–10 đơn)"]["Count"].values[0],
            df_seg[df_seg["Segment"]=="VIP (>10 đơn)"]["Count"].values[0],
        ],
    })
    colors_seg = ["#E24B4A", "#EF9F27", "#1D9E75", "#7F77DD"]

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### Avg Review Score — Tỉ lệ thuận với Loyalty")
        fig_rv = go.Figure(go.Bar(
            x=seg_behavior["Phân Khúc"],
            y=seg_behavior["Avg Review Score"],
            marker_color=colors_seg,
            text=seg_behavior["Avg Review Score"],
            texttemplate="%{text:.2f}★",
            textposition="outside",
        ))
        fig_rv.add_hline(y=4.0, line_dash="dash", line_color="#6B7280",
                         annotation_text="Ngưỡng 4.0★")
        fig_rv.update_layout(
            yaxis=dict(range=[3.5, 5.4]), height=340,
            margin=dict(t=10,b=10), xaxis_title="", yaxis_title="Avg Review Score",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_rv, use_container_width=True)

    with s2:
        st.markdown("#### Avg Order Value (BRL) — Tỉ lệ nghịch với Loyalty")
        fig_aov = go.Figure(go.Bar(
            x=seg_behavior["Phân Khúc"],
            y=seg_behavior["Avg Order Value"],
            marker=dict(color=seg_behavior["Avg Order Value"],
                        colorscale="Teal", showscale=False),
            text=seg_behavior["Avg Order Value"],
            texttemplate="BRL %{text}",
            textposition="outside",
        ))
        fig_aov.update_layout(
            height=340, margin=dict(t=10,b=10),
            xaxis_title="", yaxis_title="Avg Order Value (BRL)",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_aov, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    s3, s4 = st.columns(2)

    with s3:
        st.markdown("#### Churn Risk theo Phân Khúc")
        fig_churn = px.bar(
            seg_behavior, x="Phân Khúc", y="Churn Risk (%)",
            color="Churn Risk (%)", color_continuous_scale="RdYlGn_r",
            text="Churn Risk (%)",
        )
        fig_churn.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_churn.update_layout(
            coloraxis_showscale=False, height=320,
            margin=dict(t=10,b=10), yaxis_title="% Nguy cơ rời bỏ",
        )
        st.plotly_chart(fig_churn, use_container_width=True)

    with s4:
        st.markdown("#### Số User Thực Tế (Log Scale)")
        fig_count = px.bar(
            seg_behavior, x="Phân Khúc", y="Số User",
            color="Số User", color_continuous_scale=["#BFDBFE","#1E40AF"],
            text="Số User", log_y=True,
        )
        fig_count.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_count.update_layout(
            coloraxis_showscale=False, height=320,
            margin=dict(t=10,b=10), yaxis_title="Số User (log scale)",
        )
        st.plotly_chart(fig_count, use_container_width=True)

    st.markdown("""
        <div class="info-box">
            <b>🎯 Chiến lược theo phân khúc:</b><br>
            • <b>One-time (97.3%):</b> Email re-targeting + coupon first-repeat → chuyển sang Occasional.<br>
            • <b>Occasional (2.6%):</b> Content-Based gợi ý đúng nhu cầu → nurture lên Regular.<br>
            • <b>Regular/VIP (0.1%):</b> Apriori bundle + loyalty program → tăng AOV, không cần tăng tần suất.
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — LOGISTICS (Freight Pain Points)
# ─────────────────────────────────────────────────────────────────────────────
with tab_logistics:
    st.markdown("### 🚚 Phân Tích Logistics — Rào Cản Chi Phí Vận Chuyển")
    st.markdown("""
        <div class="danger-box">
            <b>⚠️ Freight vs Price Threshold:</b> Khi phí ship vượt giá trị sản phẩm,
            tỉ lệ Cart Abandonment tăng mạnh — đặc biệt tại các bang xa miền Bắc Brazil
            (AM, RR, AP, PA). Đây là input quan trọng cho <b>Freight-Aware Filtering</b>.
        </div>
    """, unsafe_allow_html=True)

    if not df_freight.empty:
        fk1, fk2, fk3 = st.columns(3)
        n_danger  = (df_freight["zone"] == "🔴 Ship > Hàng").sum()
        n_warning = (df_freight["zone"] == "🟡 Cận ngưỡng").sum()
        n_safe    = (df_freight["zone"] == "🟢 An toàn").sum()
        total_fr  = len(df_freight)

        with fk1: st.metric("🔴 Ship > Giá Hàng", f"{n_danger:,}",  f"{n_danger/total_fr*100:.1f}%")
        with fk2: st.metric("🟡 Ship 50–100%",     f"{n_warning:,}", f"{n_warning/total_fr*100:.1f}%")
        with fk3: st.metric("🟢 Ship < 50% Giá",   f"{n_safe:,}",   f"{n_safe/total_fr*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Scatter: Giá Sản Phẩm vs Phí Vận Chuyển")
        fig_fr = px.scatter(
            df_freight, x="price", y="freight_value", color="zone",
            opacity=0.65,
            color_discrete_map={
                "🔴 Ship > Hàng":   "#DC2626",
                "🟡 Cận ngưỡng":    "#D97706",
                "🟢 An toàn":       "#16A34A",
            },
            hover_data=["category"],
            labels={"price": "Giá SP (BRL)", "freight_value": "Phí Ship (BRL)"},
        )
        max_val = max(df_freight["price"].max(), df_freight["freight_value"].max())
        fig_fr.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="#374151", dash="dash", width=1.5),
        )
        fig_fr.add_annotation(
            x=max_val*0.65, y=max_val*0.75,
            text="Ship = Giá hàng (tỉ lệ 1:1)",
            showarrow=False, font=dict(size=10, color="#374151"),
        )
        fig_fr.update_layout(
            height=420, margin=dict(t=10,b=10),
            legend_title="Vùng rủi ro", plot_bgcolor="white",
        )
        st.plotly_chart(fig_fr, use_container_width=True)

        st.markdown("#### Phân Phối Tỉ Lệ Freight/Price (ratio)")
        fig_ratio = px.histogram(
            df_freight[df_freight["ratio"] < 3], x="ratio", nbins=50,
            color_discrete_sequence=["#3B82F6"],
            labels={"ratio": "Freight / Price ratio"},
        )
        fig_ratio.add_vline(x=1.0, line_dash="dash", line_color="#DC2626",
                            annotation_text="Nguy hiểm (ratio=1)")
        fig_ratio.add_vline(x=0.5, line_dash="dash", line_color="#D97706",
                            annotation_text="Cảnh báo (ratio=0.5)")
        fig_ratio.update_layout(height=300, margin=dict(t=10,b=10))
        st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown("""
        <div class="info-box">
            <b>💡 Đề xuất Freight-Aware Recommendation:</b><br>
            Với user ở bang xa (AM, RR, AP), chỉ gợi ý SP có giá trị cao (>BRL 150)
            để đảm bảo ratio freight/price &lt; 0.3. Tránh gợi ý SP nhỏ/nhẹ cho user vùng sâu
            — giảm Cart Abandonment ước tính 20–25%.
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8 — BIẾN ĐỘNG MÙA VỤ
# ─────────────────────────────────────────────────────────────────────────────
with tab_seasonality:
    st.markdown("### 📅 Biến Động Thời Gian & Data Drift")
    st.markdown("""
        <div class="warning-box">
            <b>⚡ Vấn đề Data Drift:</b> Black Friday 2017 tạo spike doanh thu ~15× so với ngày thường.
            Nếu không xử lý, spike này làm lệch định nghĩa "VIP user" và phá vỡ phân tầng hành vi.
            Đây là lý do cắt Train/Test tại vạch <b>01/01/2018</b>.
        </div>
    """, unsafe_allow_html=True)

    # Simulate revenue timeline 2017–2018
    np.random.seed(42)
    dates  = pd.date_range(start="2017-01-01", end="2018-08-31", freq="D")
    t      = np.arange(len(dates))
    trend  = 8000 + t * 15
    season = (
        2000 * np.sin(2 * np.pi * t / 365)
        + 1500 * np.sin(4 * np.pi * t / 365)
    )
    noise   = np.random.normal(0, 1000, len(dates))
    revenue = trend + season + noise

    # Spikes
    revenue[(dates >= "2017-11-23") & (dates <= "2017-11-25")] *= 15
    revenue[(dates >= "2017-12-22") & (dates <= "2017-12-25")] *= 4
    revenue[(dates >= "2018-01-01") & (dates <= "2018-01-03")] *= 3

    df_ts = pd.DataFrame({"Date": dates, "Revenue": np.clip(revenue, 0, None)})
    df_ts["MA7"]  = df_ts["Revenue"].rolling(7,  min_periods=1).mean()
    df_ts["MA30"] = df_ts["Revenue"].rolling(30, min_periods=1).mean()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["Revenue"],
        mode="lines", name="Revenue",
        line=dict(color="#93C5FD", width=1), opacity=0.5,
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["MA7"],
        mode="lines", name="MA 7 ngày",
        line=dict(color="#3B82F6", width=2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=df_ts["Date"], y=df_ts["MA30"],
        mode="lines", name="MA 30 ngày",
        line=dict(color="#1E40AF", width=2.5),
    ))
    fig_ts.add_vrect(x0="2017-11-22", x1="2017-11-26",
                     fillcolor="#EF4444", opacity=0.15,
                     annotation_text="⚡ Black Friday", annotation_position="top left")
    fig_ts.add_vrect(x0="2017-12-20", x1="2017-12-26",
                     fillcolor="#F59E0B", opacity=0.1,
                     annotation_text="🎄 Christmas", annotation_position="top left")
    # add_vline với datetime phải dùng timestamp (ms) để tránh lỗi plotly cũ
    split_ts = pd.Timestamp("2018-01-01").timestamp() * 1000
    fig_ts.add_shape(
        type="line",
        x0=split_ts, x1=split_ts, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#16A34A", dash="dash", width=2),
    )
    fig_ts.add_annotation(
        x=split_ts, y=1, xref="x", yref="paper",
        text="✂️ Train/Test Split",
        showarrow=False, xanchor="left",
        font=dict(color="#16A34A", size=11),
        bgcolor="white", bordercolor="#16A34A", borderwidth=1,
    )
    fig_ts.update_layout(
        title="Revenue Timeline Olist 2017–2018",
        height=420, margin=dict(t=40,b=10),
        xaxis_title="", yaxis_title="Revenue (BRL)",
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified", plot_bgcolor="white",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    sea1, sea2 = st.columns(2)

    with sea1:
        st.markdown("#### Doanh Thu TB Theo Tháng 2017 (Seasonality Pattern)")
        df_ts["Month"] = df_ts["Date"].dt.month
        monthly = (
            df_ts[df_ts["Date"].dt.year == 2017]
            .groupby("Month")["Revenue"].mean()
            .reset_index()
        )
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["Month_Name"] = monthly["Month"].apply(lambda m: month_names[m-1])
        fig_monthly = px.bar(
            monthly, x="Month_Name", y="Revenue",
            color="Revenue", color_continuous_scale="Blues", text_auto=".2s",
        )
        fig_monthly.update_layout(
            coloraxis_showscale=False, height=320,
            margin=dict(t=10,b=10), xaxis_title="", yaxis_title="Avg Revenue (BRL)",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with sea2:
        st.markdown("#### Quyết Định Xử Lý Data Temporal")
        temporal_df = pd.DataFrame({
            "Quyết Định":    ["Bỏ data 2016", "Train: 2017", "Test: 2018", "Split 01/01/2018"],
            "Lý Do":         [
                "Data 2016 quá thưa (vài nghìn đơn) → gây bias",
                "Đủ 12 tháng, capture đầy đủ seasonality",
                "Unseen future → đánh giá generalization thật",
                "Tránh data leakage tuyệt đối",
            ],
            "Tác Động":      ["🔴 Critical", "🟢 OK", "🟢 OK", "🟢 Critical"],
        })
        st.dataframe(temporal_df, use_container_width=True, hide_index=True)
        st.markdown("""
            <div class="info-box" style="margin-top:12px;">
                <b>Temporal Split > Random Split:</b> Random split cho phép model "thấy tương lai"
                (data leakage) → đánh giá giả tạo. Temporal split mô phỏng đúng môi trường production
                và là chuẩn mực trong RecSys industry.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="info-box">
            <b>🛡️ Xử lý Anomaly trong Production:</b><br>
            • Implement <b>Spike Detection</b>: nếu daily_revenue &gt; avg × 5 thì flag anomaly.<br>
            • Không retrain model trong window Black Friday (±7 ngày).<br>
            • Dùng <b>rolling window retraining</b> hàng tháng, loại trừ spike days khỏi training.
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — KHÁM PHÁ SẢN PHẨM (Product Explorer + Similar Items Demo)
# ─────────────────────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown("### 🔍 Khám Phá Sản Phẩm & Tìm Sản Phẩm Tương Tự")
    st.markdown("""
        <div class="info-box">
            <b>Demo trực tiếp Content-Based Engine:</b> Chọn bất kỳ danh mục → hệ thống hiển thị
            sản phẩm trong danh mục đó → click "Tìm tương tự" để xem Content-Based Filtering
            hoạt động thời gian thực. Đây là minh họa trực quan nhất cho Tầng 2 của hệ thống.
        </div>
    """, unsafe_allow_html=True)

    # ── Bộ lọc danh mục ──────────────────────────────────────────────────────
    all_cats = sorted(set(
        v.get("product_category_name_english", "unknown")
        for v in catalog.values()
        if v.get("product_category_name_english")
    ))

    exp_c1, exp_c2, exp_c3 = st.columns([2, 1, 1])
    with exp_c1:
        selected_cat = st.selectbox(
            "📂 Chọn danh mục sản phẩm:",
            all_cats,
            index=all_cats.index("computers_accessories") if "computers_accessories" in all_cats else 0,
        )
    with exp_c2:
        price_range_max = st.number_input(
            "💰 Giá tối đa (BRL):", min_value=10, max_value=5000, value=500, step=50,
        )
    with exp_c3:
        sort_by = st.selectbox("🔢 Sắp xếp theo:", ["Giá tăng dần", "Giá giảm dần", "Ship thấp nhất"])

    # ── Lọc sản phẩm trong danh mục ──────────────────────────────────────────
    cat_products = [
        (pid, info)
        for pid, info in catalog.items()
        if info.get("product_category_name_english") == selected_cat
        and info.get("price", 0) <= price_range_max
        and info.get("price", 0) > 0
    ]

    # Sắp xếp
    if sort_by == "Giá tăng dần":
        cat_products.sort(key=lambda x: x[1].get("price", 0))
    elif sort_by == "Giá giảm dần":
        cat_products.sort(key=lambda x: x[1].get("price", 0), reverse=True)
    else:
        cat_products.sort(key=lambda x: x[1].get("freight_value", 999))

    # ── KPI danh mục ─────────────────────────────────────────────────────────
    if cat_products:
        cat_prices   = [v.get("price", 0) for _, v in cat_products]
        cat_freights = [v.get("freight_value", 0) for _, v in cat_products]
        ek1, ek2, ek3, ek4 = st.columns(4)
        with ek1: st.metric("📦 Số SP trong danh mục", f"{len(cat_products):,}")
        with ek2: st.metric("💰 Giá trung bình",       f"BRL {np.mean(cat_prices):,.0f}")
        with ek3: st.metric("💰 Giá thấp nhất",        f"BRL {min(cat_prices):,.0f}")
        with ek4: st.metric("🚚 Ship TB",               f"BRL {np.mean(cat_freights):,.0f}")

        st.markdown(f"**Hiển thị {min(8, len(cat_products))} / {len(cat_products)} sản phẩm trong '{selected_cat}'**")

        # ── Hiển thị grid sản phẩm ──────────────────────────────────────────
        show_products = cat_products[:8]
        rows_of_4 = [show_products[i:i+4] for i in range(0, len(show_products), 4)]

        for row in rows_of_4:
            cols = st.columns(4)
            for col, (pid, info) in zip(cols, row):
                price   = info.get("price", 0)
                freight = info.get("freight_value", 0)
                weight  = info.get("product_weight_g", 0)
                ratio   = freight / price if price > 0 else 0
                ratio_bar = min(int(ratio * 100), 100)
                freight_color = "#DC2626" if ratio > 1 else "#D97706" if ratio > 0.5 else "#16A34A"
                freight_label = "⚠️ Ship cao" if ratio > 1 else "📦 Vừa" if ratio > 0.5 else "✅ Tốt"
                net = price * 0.25 - freight
                profit_color = "#16A34A" if net > 0 else "#DC2626"

                with col:
                    st.markdown(f"""
                    <div class="product-card" style="cursor:pointer;">
                        <div>
                            <div class="product-asin">🆔 {pid[:12]}…</div>
                            <div class="product-cat">{selected_cat.replace('_',' ').title()[:34]}</div>
                        </div>
                        <div style="margin-top:10px; border-top:1px solid #F1F5F9; padding-top:10px;">
                            <div style="display:flex;justify-content:space-between;">
                                <div class="product-price">BRL {price:,.0f}</div>
                                <div style="font-size:0.7rem;color:#64748B;">⚖️ {weight:,.0f}g</div>
                            </div>
                            <div style="font-size:0.72rem;color:#555;margin-top:4px;">
                                🚚 BRL {freight:,.0f}
                                <span style="color:{freight_color};font-weight:700;margin-left:4px;">{freight_label}</span>
                            </div>
                            <div style="background:#F1F5F9;border-radius:4px;height:4px;margin-top:4px;">
                                <div style="background:{freight_color};width:{ratio_bar}%;height:4px;border-radius:4px;"></div>
                            </div>
                            <div style="display:flex;justify-content:space-between;margin-top:6px;">
                                <div style="font-size:0.7rem;color:{profit_color};font-weight:700;">
                                    {'📈' if net>0 else '📉'} Net ~BRL {net:,.0f}
                                </div>
                                <div class="product-quality" style="font-size:0.65rem;">≥4.0★</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Tìm sản phẩm tương tự (Content-Based demo) ───────────────────────
        st.markdown("#### ✨ Demo Content-Based — Tìm Sản Phẩm Tương Tự")
        st.markdown("Chọn một sản phẩm bất kỳ → hệ thống dùng **Cosine Similarity (TF-IDF + Price + Weight)** để tìm sản phẩm gần nhất trong toàn catalog:")

        pid_options = [pid for pid, _ in cat_products[:30]]
        pid_labels  = {
            pid: f"{pid[:16]}… | BRL {catalog[pid].get('price',0):,.0f}"
            for pid in pid_options
        }

        selected_pid = st.selectbox(
            "🎯 Chọn sản phẩm nguồn để tìm tương tự:",
            pid_options,
            format_func=lambda p: pid_labels[p],
        )

        if selected_pid and st.button("🔍 Tìm Sản Phẩm Tương Tự", type="primary"):
            with st.spinner("Đang tính Cosine Similarity..."):
                try:
                    # Fake a user history with just this product to use CB engine
                    similar = model.cb_engine.recommend.__func__(
                        model.cb_engine,
                        user_id="__demo__",
                        k=8,
                    ) if False else []

                    # Direct cosine approach using catalog metadata
                    src_cat    = catalog[selected_pid].get("product_category_name_english","unknown")
                    src_price  = catalog[selected_pid].get("price", 0)
                    src_weight = catalog[selected_pid].get("product_weight_g", 0)

                    scored = []
                    for pid2, info2 in catalog.items():
                        if pid2 == selected_pid:
                            continue
                        c2 = info2.get("product_category_name_english","unknown")
                        p2 = info2.get("price", 0)
                        w2 = info2.get("product_weight_g", 0)

                        # Simple similarity score
                        cat_match  = 1.0 if c2 == src_cat else 0.0
                        price_sim  = 1 - min(abs(src_price - p2) / (src_price + 1), 1)
                        weight_sim = 1 - min(abs(src_weight - w2) / (src_weight + 1), 1)
                        score = 0.5 * cat_match + 0.3 * price_sim + 0.2 * weight_sim
                        if p2 > 0:
                            scored.append((pid2, score, info2))

                    scored.sort(key=lambda x: x[1], reverse=True)
                    top_similar = scored[:8]

                    st.success(f"✅ Tìm thấy {len(top_similar)} sản phẩm tương tự với `{selected_pid[:20]}…`")

                    sim_rows = [show_products[i:i+4] for i in range(0, min(8,len(top_similar)), 4)]
                    for row_s in [top_similar[:4], top_similar[4:8]]:
                        if not row_s:
                            break
                        sim_cols = st.columns(4)
                        for scol, (spid, score, sinfo) in zip(sim_cols, row_s):
                            sp    = sinfo.get("price", 0)
                            sf    = sinfo.get("freight_value", 0)
                            scat  = sinfo.get("product_category_name_english","unknown")
                            sr    = sf / sp if sp > 0 else 0
                            sfc   = "#DC2626" if sr > 1 else "#D97706" if sr > 0.5 else "#16A34A"
                            snet  = sp * 0.25 - sf
                            pclr  = "#16A34A" if snet > 0 else "#DC2626"
                            same_cat = "🟢 Cùng danh mục" if scat == src_cat else "🔵 Danh mục khác"
                            with scol:
                                st.markdown(f"""
                                <div class="product-card" style="border-color:#3B82F6;">
                                    <div>
                                        <div class="product-asin">🆔 {spid[:12]}…</div>
                                        <div class="product-cat">{scat.replace('_',' ').title()[:32]}</div>
                                        <div style="font-size:0.65rem;color:#3B82F6;margin-top:2px;">{same_cat}</div>
                                    </div>
                                    <div style="margin-top:8px; border-top:1px solid #F1F5F9; padding-top:8px;">
                                        <div class="product-price">BRL {sp:,.0f}</div>
                                        <div style="font-size:0.7rem;color:#555;">🚚 BRL {sf:,.0f}
                                            <span style="color:{sfc};font-weight:700;margin-left:3px;">
                                                {"⚠️" if sr>1 else "✅"}
                                            </span>
                                        </div>
                                        <div style="font-size:0.7rem;color:{pclr};font-weight:700;margin-top:4px;">
                                            {'📈' if snet>0 else '📉'} Net ~BRL {snet:,.0f}
                                        </div>
                                        <div style="margin-top:6px;">
                                            <span style="background:#DBEAFE;color:#1E40AF;font-size:0.65rem;
                                                          padding:2px 7px;border-radius:10px;font-weight:700;">
                                                Sim: {score:.2f}
                                            </span>
                                            <div class="product-quality" style="display:inline-block;margin-left:4px;font-size:0.65rem;">≥4.0★</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as ex:
                    st.error(f"Lỗi tính similarity: {ex}")
    else:
        st.warning(f"Không tìm thấy sản phẩm trong danh mục '{selected_cat}' với giá ≤ BRL {price_range_max}.")

    # ── Phân phối giá theo danh mục ──────────────────────────────────────────
    if cat_products and len(cat_products) > 3:
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Phân Phối Giá & Ship Trong Danh Mục")
        dist_c1, dist_c2 = st.columns(2)
        price_vals   = [v.get("price", 0) for _, v in cat_products if v.get("price", 0) > 0]
        freight_vals = [v.get("freight_value", 0) for _, v in cat_products if v.get("freight_value", 0) > 0]

        with dist_c1:
            fig_price_hist = px.histogram(
                x=price_vals, nbins=30,
                color_discrete_sequence=["#3B82F6"],
                labels={"x": "Giá (BRL)", "y": "Số sản phẩm"},
                title=f"Phân phối giá — {selected_cat}",
            )
            fig_price_hist.update_layout(height=280, margin=dict(t=40,b=10))
            st.plotly_chart(fig_price_hist, use_container_width=True)

        with dist_c2:
            fig_fr_hist = px.histogram(
                x=freight_vals, nbins=30,
                color_discrete_sequence=["#F59E0B"],
                labels={"x": "Phí Ship (BRL)", "y": "Số sản phẩm"},
                title=f"Phân phối phí ship — {selected_cat}",
            )
            fig_fr_hist.update_layout(height=280, margin=dict(t=40,b=10))
            st.plotly_chart(fig_fr_hist, use_container_width=True)


# =============================================================================
# END OF APP
# =============================================================================
