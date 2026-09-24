from pathlib import Path

import streamlit as st

from history import clear_history, load_history, save_search
from recommender import CarRecommender, DEFAULT_WEIGHTS, load_and_clean_data, normalize_weights


CSV_FILE = Path(__file__).with_name("cars.csv")
MOC_GIA = [
    200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000,
    1200, 1500, 1800, 2000, 2500, 3000, 4000, 5000, 7000, 10000,
    13000, 16000,
]
NHAN_TS = {
    "Giá( VNĐ)": "Giá",
    "Nhiên liệu": "Nhiên liệu",
    "Công suất(HP)": "Công suất",
    "Chỗ ngồi": "Số chỗ",
    "Mục đích": "Nhu cầu sử dụng",
    "Tình trạng": "Tình trạng",
    "Kiểu xe": "Kiểu xe",
    "Hộp số": "Hộp số",
}


st.set_page_config(page_title="Gợi ý xe phù hợp", page_icon="🚗", layout="wide")
st.html(Path(__file__).with_name("styles.css"))


@st.cache_data
def doc_data():
    return load_and_clean_data(CSV_FILE)


def fmt_vnd(tien):
    return f"{tien:,.0f}".replace(",", ".") + " VNĐ"


def rut_gia(tr):
    if tr < 1000:
        return f"{tr:,.0f}".replace(",", ".") + " triệu VNĐ"
    ty = f"{tr / 1000:,.2f}".rstrip("0").rstrip(".")
    return ty.replace(",", "X").replace(".", ",").replace("X", ".") + " tỷ VNĐ"


def dong_bo(src, dst):
    st.session_state[dst] = st.session_state[src]


def ve_gia(df, ds_gia, max_gia):
    gia_tr = df["Giá( VNĐ)"] / 1_000_000
    khoang = []
    for i, duoi in enumerate(ds_gia[:-1]):
        tren = ds_gia[i + 1]
        trong = (gia_tr >= duoi) & (gia_tr < tren)
        if i == len(ds_gia) - 2:
            trong = (gia_tr >= duoi) & (gia_tr <= tren)
        dem = int(trong.sum())
        chon = dem if max_gia is None else int((trong & (gia_tr <= max_gia / 1_000_000)).sum())
        khoang.append((duoi, tren, dem, chon))

    max_dem = max((dem for _, _, dem, _ in khoang), default=1) or 1
    bars = []
    for duoi, tren, dem, chon in khoang:
        h = max(3, dem / max_dem * 100)
        tile = chon / dem * 100 if dem else 0
        nhan = f"{rut_gia(duoi)} đến {rut_gia(tren)}: {dem} xe"
        bars.append(
            f'<div class="price-bar" role="listitem" aria-label="{nhan}" '
            f'title="{nhan}" style="height:{h:.1f}%">'
            f'<span style="height:{tile:.1f}%"></span></div>'
        )
    st.html('<div class="price-histogram" role="list" aria-label="Phân bố xe theo khoảng giá">' + "".join(bars) + '</div>')


df = doc_data()
knn = CarRecommender(df)

st.title("Chọn xe theo cách của bạn")
st.caption("Đặt ngân sách, chọn nhu cầu và khám phá những mẫu xe phù hợp.")

st.subheader("1. Ngân sách")
g_min = int(df["Giá( VNĐ)"].min() / 1_000_000)
g_max = int(df["Giá( VNĐ)"].max() / 1_000_000)
if "budget_input" not in st.session_state:
    st.session_state["budget_input"] = int(df["Giá( VNĐ)"].median() / 1_000_000)
if "budget_slider" not in st.session_state:
    st.session_state["budget_slider"] = st.session_state["budget_input"]
if "no_budget" not in st.session_state:
    st.session_state["no_budget"] = False

with st.container(border=True, key="budget"):
    c_tien, c_che_do = st.columns(2, gap="large")
    with c_tien:
        ns_tr = st.number_input(
            "Ngân sách dự kiến (triệu VNĐ)", min_value=g_min, max_value=g_max,
            step=10, key="budget_input", disabled=st.session_state["no_budget"],
            on_change=dong_bo, args=("budget_input", "budget_slider"),
        )
    with c_che_do:
        cd_ns = st.segmented_control(
            "Ngân sách đã cố định chưa?", ["Cố định", "Linh hoạt 10%"],
            default="Cố định", required=True, width="stretch", key="budget_mode",
            disabled=st.session_state["no_budget"],
        )

    ns = None if st.session_state["no_budget"] else ns_tr * 1_000_000
    max_gia = ns
    if ns is not None and cd_ns == "Linh hoạt 10%":
        max_gia = round(ns * 1.10)
    ds_gia = sorted({g_min, g_max, ns_tr, *(p for p in MOC_GIA if g_min < p < g_max)})

    ve_gia(df, ds_gia, max_gia)
    st.select_slider(
        "Kéo để chọn ngân sách", options=ds_gia, format_func=rut_gia,
        key="budget_slider", disabled=st.session_state["no_budget"],
        label_visibility="collapsed", on_change=dong_bo,
        args=("budget_slider", "budget_input"),
    )
    c_trai, c_phai = st.columns([1, 2])
    with c_trai:
        st.checkbox("Chưa xác định ngân sách", key="no_budget")
    with c_phai:
        if max_gia is None:
            st.caption(f"Xem tất cả {len(df)} xe theo nhu cầu bên dưới.")
        else:
            n_hop_le = int((df["Giá( VNĐ)"] <= max_gia).sum())
            st.caption(f"{n_hop_le}/{len(df)} xe trong giới hạn · Tối đa {rut_gia(max_gia / 1_000_000)}")

st.subheader("2. Nhu cầu của bạn")
st.caption("Để “Không yêu cầu” với những tiêu chí bạn chưa quan tâm.")

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    md = st.selectbox("Nhu cầu sử dụng", ["Không yêu cầu"] + knn.unique_values("Mục đích"))
    cho = st.selectbox("Số chỗ ngồi", ["Không yêu cầu"] + [str(s) for s in knn.unique_values("Chỗ ngồi")])
with c2:
    nl = st.selectbox("Nhiên liệu", ["Không yêu cầu"] + knn.unique_values("Nhiên liệu"))
    kieu = st.selectbox("Kiểu xe", ["Không yêu cầu"] + knn.unique_values("Kiểu xe"))
with c3:
    tt = st.selectbox("Tình trạng", ["Không yêu cầu"] + knn.unique_values("Tình trạng"))
    hs = st.selectbox("Hộp số", ["Không yêu cầu"] + knn.unique_values("Hộp số"))

with st.expander("Tùy chỉnh thêm · Công suất, số kết quả và mức ưu tiên"):
    c_trai, c_phai = st.columns(2, gap="large")
    with c_trai:
        co_cs = st.checkbox("Có yêu cầu về công suất")
        cs = st.slider(
            "Công suất mong muốn (HP)",
            min_value=int(df["Công suất(HP)"].min()),
            max_value=int(df["Công suất(HP)"].max()),
            value=int(df["Công suất(HP)"].median()),
            disabled=not co_cs,
        )
    with c_phai:
        k = st.number_input("Số xe muốn xem", min_value=1, max_value=20, value=5)

    st.markdown("**Mức ưu tiên từng tiêu chí**")
    st.caption("Giữ nguyên mức mặc định nếu bạn chưa muốn tùy chỉnh. Tổng trọng số được tự động chuẩn hóa.")
    cot_w = st.columns(2, gap="large")
    w_raw = {}
    for i, nhom in enumerate(DEFAULT_WEIGHTS):
        with cot_w[i % 2]:
            w_raw[nhom] = st.slider(
                NHAN_TS[nhom], 0.0, 1.0, DEFAULT_WEIGHTS[nhom], 0.05,
                key=f"weight_{i}",
            )

w = normalize_weights(w_raw)

q = {}
if ns is not None:
    q["Giá( VNĐ)"] = ns
if co_cs:
    q["Công suất(HP)"] = cs
if cho != "Không yêu cầu":
    q["Chỗ ngồi"] = int(cho)
if nl != "Không yêu cầu":
    q["Nhiên liệu"] = nl
if md != "Không yêu cầu":
    q["Mục đích"] = md
if kieu != "Không yêu cầu":
    q["Kiểu xe"] = kieu
if tt != "Không yêu cầu":
    q["Tình trạng"] = tt
if hs != "Không yêu cầu":
    q["Hộp số"] = hs

cur_q = (q, w, max_gia, int(k))
if "last_search" not in st.session_state:
    st.session_state["last_search"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if st.button("Xem xe phù hợp", type="primary", width="stretch"):
    if not q:
        st.warning("Hãy chọn ít nhất một tiêu chí hoặc nhập ngân sách để tìm xe.")
    else:
        kq = knn.recommend(q, weights=w, k=int(k), max_price=max_gia)
        st.session_state["last_result"] = kq
        st.session_state["last_search"] = cur_q
        if not kq.empty:
            save_search(q, w, kq["Tên xe"].tolist())

kq = st.session_state["last_result"]
if kq is not None and st.session_state["last_search"] != cur_q:
    st.info("Bạn đã thay đổi lựa chọn. Nhấn “Xem xe phù hợp” để cập nhật kết quả.")
elif kq is not None:
    st.subheader("3. Xe được gợi ý")
    if kq.empty:
        st.warning("Chưa có xe trong giới hạn ngân sách này. Hãy tăng ngân sách hoặc chọn mức linh hoạt.")
    else:
        st.caption(f"{len(kq)} xe được xếp theo mức độ gần với tiêu chí bạn đã chọn.")
        df_sub = kq[["Tên xe", "Giá( VNĐ)", "Nhiên liệu", "Chỗ ngồi", "Kiểu xe", "Độ phù hợp (%)"]].copy()
        df_sub["Giá( VNĐ)"] = df_sub["Giá( VNĐ)"].map(fmt_vnd)
        df_sub = df_sub.rename(columns={"Giá( VNĐ)": "Giá", "Độ phù hợp (%)": "Mức phù hợp"})
        df_sub["Mức phù hợp"] = df_sub["Mức phù hợp"].map(lambda score: f"{score:.1f}%")
        st.dataframe(df_sub, hide_index=True, width="stretch")

        for i, r in kq.iterrows():
            with st.expander(f"{i + 1}. {r['Tên xe']} · {r['Độ phù hợp (%)']:.1f}% phù hợp", expanded=i == 0):
                c_tt, c_diem = st.columns([3, 2], gap="large")
                with c_tt:
                    st.write(f"**Giá:** {fmt_vnd(r['Giá( VNĐ)'])}")
                    st.write(f"**Loại xe:** {r['Kiểu xe']} · {r['Chỗ ngồi']} chỗ · {r['Hộp số']}")
                    st.write(f"**Nhiên liệu:** {r['Nhiên liệu']} · **Công suất:** {r['Công suất(HP)']:g} HP")
                    st.write(f"**Nhu cầu sử dụng:** {r['Mục đích']} · **Tình trạng:** {r['Tình trạng']}")
                with c_diem:
                    st.metric("Mức phù hợp", f"{r['Độ phù hợp (%)']:.1f}%")
                    for tc, diem in r.get("Chi tiết tiêu chí", {}).items():
                        st.progress(
                            min(max(float(diem) / 100, 0.0), 1.0),
                            text=f"{NHAN_TS.get(tc, tc)}: {diem:.1f}%",
                        )

                for gt in r.get("Giải thích", []):
                    st.write(gt)

        with st.expander("Cách đọc mức phù hợp"):
            st.write("Hệ thống lọc xe theo giới hạn ngân sách, rồi xếp hạng bằng khoảng cách KNN có trọng số.")
            st.write("Mức phù hợp là điểm quy đổi từ khoảng cách để dễ so sánh; đây không phải xác suất mua xe phù hợp.")

st.divider()
with st.expander("Lịch sử tìm kiếm"):
    ls = load_history()
    if not ls:
        st.caption("Chưa có lượt tìm kiếm nào được lưu.")
    else:
        st.caption(f"Đã lưu {len(ls)} lượt tìm kiếm. Dưới đây là 10 lượt gần nhất.")
        for item in reversed(ls[-10:]):
            st.markdown(f"**{item.get('thoi_gian', 'Không rõ thời gian')}**")
            tc = item.get("tieu_chi", {})
            txt = ", ".join(
                f"{NHAN_TS.get(k, k)}: {fmt_vnd(v) if k == 'Giá( VNĐ)' else v}"
                for k, v in tc.items()
            )
            st.caption(txt or "Không có tiêu chí")
            st.write("Gợi ý:", ", ".join(item.get("ket_qua_hang_dau", [])))
        if st.button("Xóa lịch sử"):
            clear_history()
            st.session_state["last_result"] = None
            st.session_state["last_search"] = None
            st.rerun()
