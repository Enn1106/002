import streamlit as st
import pandas as pd

from recommender import load_and_clean_data, CarRecommender, DEFAULT_WEIGHTS, normalize_weights
from history import load_history, save_search, clear_history

st.set_page_config(page_title="Gợi ý xe ô tô theo nhu cầu (KNN)", page_icon="🚗", layout="wide")


@st.cache_data
def get_data():
    return load_and_clean_data("car0.csv")


@st.cache_resource
def get_recommender(_df: pd.DataFrame):
    return CarRecommender(_df)

def format_vnd(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " VNĐ"


def show_score_bar(label: str, score: float):
    """Hiển thị thanh tiến trình đáp ứng tiêu chí"""
    if score >= 80:
        color = "#4CAF50"  # xanh lá
        text_style = f"color: {color}; font-weight: bold;"
    elif score >= 50:
        color = "#FFC400"  # vàng
        text_style = f"color: {color}; font-weight: normal;"
    else:
        color = "#F44336"  # đỏ
        text_style = f"color: {color}; font-weight: normal;"

    html_str = f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="{text_style}">{label}</span>
            <span style="{text_style}">{score:.1f}%</span>
        </div>
        <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 8px;">
            <div style="width: {score}%; background-color: {color}; height: 100%; border-radius: 4px;"></div>
        </div>
    </div>
    """
    st.markdown(html_str, unsafe_allow_html=True)


WEIGHT_LABELS = {
    "Giá( VNĐ)": "Giá",
    "Nhiên liệu": "Nhiên liệu",
    "Công suất(HP)": "Công suất",
    "Chỗ ngồi": "Chỗ ngồi",
    "Mục đích": "Mục đích",
    "Tình trạng": "Tình trạng",
    "Kiểu xe": "Kiểu xe",
    "Hộp số": "Hộp số",
}

df = get_data()
recommender = get_recommender(df)

st.title(" Hệ thống hỗ trợ lựa chọn xe ô tô theo nhu cầu sử dụng")
st.caption("Nhóm 8 - Đồ án môn Hệ trợ giúp quyết định — Gợi ý xe bằng thuật toán K-Nearest Neighbors (KNN)")

# --- Sidebar: Trọng số ---
with st.sidebar:
    st.header(" Trọng số thuộc tính")
    st.caption("Tổng trọng số sẽ tự động chuẩn hóa bằng 1.00.")

    raw_weights = {}
    for group in DEFAULT_WEIGHTS:
        raw_weights[group] = st.slider(
            WEIGHT_LABELS[group], 0.0, 1.0, DEFAULT_WEIGHTS[group], 0.05
        )

    active_weights = normalize_weights(raw_weights)

    st.divider()
    st.caption("Trọng số thực tế sau chuẩn hóa:")
    weight_df = pd.DataFrame(
        {
            "Thuộc tính": [WEIGHT_LABELS[g] for g in active_weights],
            "Trọng số": [f"{v:.2f}" for v in active_weights.values()],
        }
    )
    st.dataframe(weight_df, hide_index=True, use_container_width=True)

st.subheader("1. Nhập tiêu chí mong muốn")

col1, col2, col3, col4 = st.columns(4)

with col1:
    use_price = st.checkbox("Có ngân sách tối đa", value=True)
    price_trieu = st.slider(
        "Ngân sách tối đa (triệu VNĐ)",
        min_value=int(df["Giá( VNĐ)"].min() / 1_000_000),
        max_value=int(df["Giá( VNĐ)"].max() / 1_000_000),
        value=int(df["Giá( VNĐ)"].median() / 1_000_000),
        step=10,
        disabled=not use_price,
    )
    price = price_trieu * 1_000_000
    if use_price:
        st.caption(f"≈ {format_vnd(price)} (Lọc cứng: Các xe vượt ngân sách này sẽ bị loại bỏ hoàn toàn)")

with col2:
    use_power = st.checkbox("Có yêu cầu về công suất", value=False)
    power = st.slider(
        "Công suất mong muốn (HP)",
        min_value=int(df["Công suất(HP)"].min()),
        max_value=int(df["Công suất(HP)"].max()),
        value=int(df["Công suất(HP)"].median()),
        disabled=not use_power,
    )

with col3:
    seats_options = ["Không yêu cầu"] + [str(s) for s in recommender.unique_values("Chỗ ngồi")]
    seats = st.selectbox("Số chỗ ngồi", seats_options)

    fuel_options = ["Không yêu cầu"] + recommender.unique_values("Nhiên liệu")
    fuel = st.selectbox("Nhiên liệu", fuel_options)

with col4:
    purpose_options = ["Không yêu cầu"] + recommender.unique_values("Mục đích")
    purpose = st.selectbox("Mục đích sử dụng", purpose_options)

    body_options = ["Không yêu cầu"] + recommender.unique_values("Kiểu xe")
    body = st.selectbox("Kiểu xe", body_options)

col5, col6, col7 = st.columns(3)
with col5:
    condition_options = ["Không yêu cầu"] + recommender.unique_values("Tình trạng")
    condition = st.selectbox("Tình trạng", condition_options)
with col6:
    gearbox_options = ["Không yêu cầu"] + recommender.unique_values("Hộp số")
    gearbox = st.selectbox("Hộp số", gearbox_options)
with col7:
    k = st.number_input("Số lượng xe gợi ý (K)", min_value=1, max_value=20, value=5)

st.divider()

# Dùng session_state để lưu kết quả gợi ý giữa các lần rerun (bấm xem lịch sử không bị mất kết quả)
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if st.button("Tìm xe phù hợp", type="primary"):
    query = {}
    if use_price:
        query["Giá( VNĐ)"] = price
    if use_power:
        query["Công suất(HP)"] = power
    if seats != "Không yêu cầu":
        query["Chỗ ngồi"] = int(seats)
    if fuel != "Không yêu cầu":
        query["Nhiên liệu"] = fuel
    if purpose != "Không yêu cầu":
        query["Mục đích"] = purpose
    if body != "Không yêu cầu":
        query["Kiểu xe"] = body
    if condition != "Không yêu cầu":
        query["Tình trạng"] = condition
    if gearbox != "Không yêu cầu":
        query["Hộp số"] = gearbox

    if not query:
        st.warning("Vui lòng nhập ít nhất một tiêu chí để hệ thống gợi ý.")
    else:
        result = recommender.recommend(query, weights=active_weights, k=int(k))
        st.session_state["last_result"] = result
        
        if len(result) > 0:
            save_search(query, active_weights, result["Tên xe"].tolist())
            st.toast("Đã lưu lượt tìm kiếm vào lịch sử.", icon="💾")

# HẠN CHẾ VẤN ĐỀ TRÔI GIAO DIỆN KHI XEM LỊCH SỬ: Hiển thị kết quả tìm kiếm nếu có
result = st.session_state["last_result"]

if result is not None:
    st.subheader("2. Kết quả gợi ý")

    if len(result) == 0:
        st.error("Không có xe nào nằm trong ngân sách tối đa bạn chọn. Vui lòng tăng ngân sách!")
    else:
        st.caption(f"Top {len(result)} xe gần nhất dựa trên thuật toán KNN:")

        display_df = result[
            [
                "Tên xe",
                "Giá( VNĐ)",
                "Nhiên liệu",
                "Công suất(HP)",
                "Chỗ ngồi",
                "Mục đích",
                "Tình trạng",
                "Kiểu xe",
                "Hộp số",
                "Độ phù hợp (%)",
            ]
        ].copy()
        display_df["Giá( VNĐ)"] = display_df["Giá( VNĐ)"].apply(format_vnd)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

       # --- Vùng Giải thích kết quả chi tiết từng xe ---
        st.subheader("3. Giải thích lý do đề xuất")
        for index, row in result.iterrows():
            with st.expander(
                f"{index + 1}. {row['Tên xe']} — Độ phù hợp: {row['Độ phù hợp (%)']:.1f}%",
                expanded=(index == 0),
            ):
                col_info, col_score = st.columns([3, 4])

                with col_info:
                    st.write(f"**Giá:** {format_vnd(row['Giá( VNĐ)'])}")
                    st.write(f"**Nhiên liệu:** {row['Nhiên liệu']}")
                    st.write(f"**Công suất:** {row['Công suất(HP)']} HP")
                    st.write(f"**Chỗ ngồi:** {row['Chỗ ngồi']} chỗ")
                    st.write(f"**Mục đích:** {row['Mục đích']}")
                    st.write(f"**Kiểu xe:** {row['Kiểu xe']}")
                    st.write(f"**Tình trạng:** {row['Tình trạng']}")
                    st.write(f"**Hộp số:** {row['Hộp số']}")

                with col_score:
                    st.metric("Độ phù hợp", f"{row['Độ phù hợp (%)']:.1f}%")
                    st.caption(f"Khoảng cách KNN: {row['Khoảng cách']:.4f}")

                # Bổ sung kiểm tra an toàn bằng .get() để tránh văng KeyError
                details = row.get("Chi tiết tiêu chí", {})
                if details:
                    st.markdown("**Mức độ đáp ứng từng tiêu chí:**")
                    for col_name, score in details.items():
                        show_score_bar(WEIGHT_LABELS.get(col_name, col_name), score)

                explanations = row.get("Giải thích", [])
                if explanations:
                    st.markdown("**Lý do đề xuất:**")
                    for explanation in explanations:
                        if "✓" in explanation:
                            st.markdown(f"<span style='color: #2e7d32; font-weight: bold;'>{explanation}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: #d32f2f;'>{explanation}</span>", unsafe_allow_html=True)

# --- 4. Lịch sử tìm kiếm (Sửa hiển thị rõ ràng, ổn định) ---
st.divider()
st.subheader("4. Lịch sử tìm kiếm")

history = load_history()
if not history:
    st.caption("Chưa có lượt tìm kiếm nào được lưu.")
else:
    st.caption(f"Đã lưu {len(history)} lượt tìm kiếm trong file `history.json`.")
    
    # Hiển thị lịch sử trình bày đẹp mắt thay vì JSON thô
    for idx, entry in enumerate(reversed(history[-10:])):
        with st.expander(f" Lượt tìm kiếm lúc: {entry.get('thoi_gian', 'N/A')}"):
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown("**Tiêu chí đã chọn:**")
                tieu_chi_dict = entry.get("tieu_chi", {})
                if not tieu_chi_dict:
                    st.write("- Không nhập tiêu chí cụ thể")
                else:
                    for k, v in tieu_chi_dict.items():
                        label_name = WEIGHT_LABELS.get(k, k)
                        val_str = format_vnd(v) if k == "Giá( VNĐ)" else str(v)
                        st.write(f"- **{label_name}:** {val_str}")
            
            with col_h2:
                st.markdown("**Top xe đề xuất:**")
                top_cars = entry.get("ket_qua_hang_dau", [])
                for car_num, car_name in enumerate(top_cars, 1):
                    st.write(f"{car_num}. {car_name}")

    if st.button("🗑️ Xóa toàn bộ lịch sử"):
        clear_history()
        st.session_state["last_result"] = None
        st.rerun()