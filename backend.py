"""
backend.py
Xử lý dữ liệu và xây dựng mô hình KNN (Nearest Neighbors) để gợi ý xe
dựa trên tiêu chí khách hàng nhập vào.

- Trọng số do người dùng tự điều chỉnh qua giao diện (không còn luật IF-THEN tự động).
  Giao diện chịu trách nhiệm chuẩn hóa để tổng trọng số luôn bằng 1 trước khi gọi recommend().
- Ngân sách được xử lý bằng bộ lọc cứng (loại xe vượt ngân sách) trước khi tính khoảng cách.
- Khoảng cách Euclid có trọng số: d_w(p,q) = sqrt(sum_i w_i * (p_i - q_i)^2).
- Độ phù hợp: Score(p,q) = 1 / (1 + d_w) * 100%.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

NUMERIC_COLS = ["Giá( VNĐ)", "Công suất(HP)", "Chỗ ngồi"]
CATEGORICAL_COLS = ["Nhiên liệu", "Mục đích", "Tình trạng", "Kiểu xe", "Hộp số"]
ALL_GROUPS = ["Giá( VNĐ)", "Nhiên liệu", "Công suất(HP)", "Chỗ ngồi",
              "Mục đích", "Tình trạng", "Kiểu xe", "Hộp số"]

# Trọng số mặc định khi mở ứng dụng lần đầu (người dùng có thể chỉnh lại bằng slider).
# Chia đều 1/8 cho mỗi nhóm để không thiên vị thuộc tính nào.
DEFAULT_WEIGHTS = {g: 1.0 / len(ALL_GROUPS) for g in ALL_GROUPS}


def normalize_weights(raw_weights: dict) -> dict:
    """
    Chuẩn hóa một bộ trọng số bất kỳ (giá trị dương tùy ý từ slider) về đúng
    ràng buộc công thức (2.3): không âm và tổng bằng 1.
    Nếu tổng bằng 0 (người dùng kéo hết về 0), trả về bộ trọng số chia đều.
    """
    total = sum(max(v, 0.0) for v in raw_weights.values())
    if total <= 1e-9:
        return dict(DEFAULT_WEIGHTS)
    return {k: max(v, 0.0) / total for k, v in raw_weights.items()}


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Đọc CSV gốc và làm sạch dữ liệu."""
    df = pd.read_csv(path)

    # Giá: "458.000.000" -> 458000000
    df["Giá( VNĐ)"] = (
        df["Giá( VNĐ)"].astype(str).str.replace(".", "", regex=False).astype(float)
    )

    # Công suất: chuỗi số (có thể dùng dấu phẩy thập phân kiểu VN, vd "134,1") -> float
    df["Công suất(HP)"] = (
        df["Công suất(HP)"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    # Chuẩn hóa lỗi chính tả Nhiên liệu: "HYBRID" / "Hybrid" -> "Hybrid"
    df["Nhiên liệu"] = df["Nhiên liệu"].str.strip().str.title()

    # Chuẩn hóa khoảng trắng thừa ở các cột phân loại khác
    for col in CATEGORICAL_COLS:
        if col != "Nhiên liệu":
            df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    return df


class CarRecommender:
    """
    Bọc toàn bộ pipeline: chuẩn hóa Min-Max, One-Hot Encoding, phân bổ trọng số
    nhóm cho từng cột thành phần (chia đều: w_g,j = W_g / m_g), lọc cứng theo
    ngân sách, và tính khoảng cách Euclid có trọng số bằng công thức (2.2).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # --- Chuẩn hóa Min-Max các cột số về [0, 1] ---
        self.scaler = MinMaxScaler()
        self.numeric_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df[NUMERIC_COLS]),
            columns=NUMERIC_COLS,
        )

        # --- One-Hot Encoding cho các cột phân loại ---
        self.encoded_cat = pd.get_dummies(self.df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
        self.cat_col_map = {
            col: [c for c in self.encoded_cat.columns if c.startswith(f"{col}_")]
            for col in CATEGORICAL_COLS
        }

        # Ma trận đặc trưng đầy đủ (chưa nhân trọng số) — dùng lại cho mọi truy vấn
        self.feature_matrix = pd.concat(
            [self.numeric_scaled[NUMERIC_COLS], self.encoded_cat], axis=1
        ).astype(float)

    # ------------------------------------------------------------------
    def _scale_numeric(self, col: str, value: float) -> float:
        """Đưa 1 giá trị số về cùng thang [0, 1] mà scaler đã học từ dữ liệu gốc."""
        idx = NUMERIC_COLS.index(col)
        data_min = self.scaler.data_min_[idx]
        data_range = self.scaler.data_range_[idx]
        if data_range <= 1e-9:
            return 0.5
        return float(np.clip((value - data_min) / data_range, 0.0, 1.0))

    def _criteria_breakdown(self, query: dict, row: pd.Series):
        """
        Tính % đáp ứng cho TỪNG tiêu chí mà người dùng đã chọn (bỏ qua các
        tiêu chí để "Không yêu cầu") và sinh câu giải thích ngắn cho mỗi tiêu chí.

        - Cột số (Giá, Công suất, Chỗ ngồi):
          % = (1 - |giá trị xe đã chuẩn hóa - giá trị yêu cầu đã chuẩn hóa|) * 100
        - Cột phân loại (Nhiên liệu, Mục đích, Tình trạng, Kiểu xe, Hộp số):
          % = 100 nếu trùng khớp, 0 nếu khác.

        Trả về (chi_tiet: dict[str, float], giai_thich: list[str]).
        """
        chi_tiet = {}
        giai_thich = []

        for col in NUMERIC_COLS:
            target = query.get(col)
            if target is None:
                continue
            car_scaled = self._scale_numeric(col, row[col])
            target_scaled = self._scale_numeric(col, target)
            pct = round((1.0 - abs(car_scaled - target_scaled)) * 100, 1)
            chi_tiet[col] = pct

            if col == "Giá( VNĐ)":
                car_text = f"{row[col]:,.0f}".replace(",", ".") + " VNĐ"
                target_text = f"{target:,.0f}".replace(",", ".") + " VNĐ"
            elif col == "Công suất(HP)":
                car_text, target_text = f"{row[col]:g} HP", f"{target:g} HP"
            else:
                car_text, target_text = f"{int(row[col])} chỗ", f"{int(target)} chỗ"

            if pct >= 80:
                giai_thich.append(f"✓ {col}: xe có {car_text}, rất sát yêu cầu ({target_text}).")
            else:
                giai_thich.append(f"✗ {col}: xe có {car_text}, lệch khá nhiều so với yêu cầu ({target_text}).")

        for col in CATEGORICAL_COLS:
            target = query.get(col)
            if target is None:
                continue
            car_val = row[col]
            pct = 100.0 if str(car_val) == str(target) else 0.0
            chi_tiet[col] = pct
            if pct == 100.0:
                giai_thich.append(f"✓ {col}: đúng yêu cầu ({target}).")
            else:
                giai_thich.append(f"✗ {col}: xe là '{car_val}', khác với yêu cầu '{target}'.")

        return chi_tiet, giai_thich

    def _per_dimension_weights(self, group_weights: dict) -> np.ndarray:
        """
        Áp dụng công thức (2.4): w_g,j = W_g / m_g cho từng cột One-Hot;
        với cột số, w = W_g trực tiếp (m_g = 1).
        Trả về vector trọng số theo đúng thứ tự cột của feature_matrix.
        """
        weights = []
        for col in NUMERIC_COLS:
            weights.append(group_weights.get(col, 0.0))
        for col in CATEGORICAL_COLS:
            sub_cols = self.cat_col_map[col]
            m_g = max(len(sub_cols), 1)
            w_gj = group_weights.get(col, 0.0) / m_g
            weights.extend([w_gj] * len(sub_cols))
        return np.array(weights, dtype=float)

    def _encode_query(self, query: dict) -> np.ndarray:
        """Chuyển tiêu chí khách hàng thành vector cùng không gian đặc trưng."""
        row = {}
        numeric_input = {col: query.get(col) for col in NUMERIC_COLS}
        numeric_df = pd.DataFrame([numeric_input])
        for col in NUMERIC_COLS:
            if pd.isna(numeric_df[col].iloc[0]):
                # Không nhập -> đặt ở tâm thang đo để không lệch về hướng nào
                numeric_df[col] = 0.5
                row[col] = 0.5
        scaled_known = self.scaler.transform(
            numeric_df[NUMERIC_COLS].fillna(0)
        )[0]
        for i, col in enumerate(NUMERIC_COLS):
            if col not in row:
                row[col] = float(np.clip(scaled_known[i], 0.0, 1.0))

        for col in CATEGORICAL_COLS:
            chosen = query.get(col)
            for sc in self.cat_col_map[col]:
                row[sc] = 1.0 if (chosen is not None and sc == f"{col}_{chosen}") else 0.0

        return pd.DataFrame([row])[self.feature_matrix.columns].values[0].astype(float)

    def recommend(self, query: dict, weights: dict = None, k: int = 5) -> pd.DataFrame:
        """
        Quy trình suy luận: (1) nhận tiêu chí + lọc cứng theo ngân sách;
        (2) dùng bộ trọng số người dùng đã chỉnh (đã được chuẩn hóa tổng = 1
        ở tầng giao diện qua normalize_weights()); (3) chuẩn hóa vector truy vấn;
        (4) tính khoảng cách có trọng số trên tập xe hợp lệ; (5) sắp xếp lấy top-k.

        weights: dict 8 khóa (ALL_GROUPS) đã chuẩn hóa tổng = 1. Nếu để None,
        dùng DEFAULT_WEIGHTS (chia đều).
        """
        # (1) Lọc cứng theo ngân sách tối đa (Giai đoạn 1)
        budget = query.get("Giá( VNĐ)")
        if budget is not None:
            mask = self.df["Giá( VNĐ)"] <= budget
            valid_df = self.df[mask].reset_index(drop=True)
            valid_features = self.feature_matrix.loc[mask].reset_index(drop=True)
        else:
            valid_df = self.df
            valid_features = self.feature_matrix

        if len(valid_df) == 0:
            return valid_df.assign(**{"Khoảng cách": [], "Độ phù hợp (%)": []})

        # (2) Bộ trọng số do người dùng chỉnh tay (đã chuẩn hóa tổng = 1)
        group_weights = weights if weights is not None else DEFAULT_WEIGHTS
        w = self._per_dimension_weights(group_weights)

        # (3) Chuẩn hóa vector truy vấn
        q_vector = self._encode_query(query)

        # (4) Khoảng cách Euclid có trọng số: d_w = sqrt(sum(w_i * (p_i - q_i)^2))
        diff_sq = (valid_features.values - q_vector) ** 2
        d_w = np.sqrt((diff_sq * w).sum(axis=1))

        # (5) Sắp xếp và lấy top-k
        k = min(k, len(valid_df))
        order = np.argsort(d_w)[:k]

        result = valid_df.iloc[order].copy()
        result["Khoảng cách"] = d_w[order]
        # Công thức (2.7): Score = 1 / (1 + d_w) * 100%
        result["Độ phù hợp (%)"] = (1.0 / (1.0 + result["Khoảng cách"]) * 100).round(1)
        result = result.reset_index(drop=True)

        # (6) Chi tiết % đáp ứng từng tiêu chí + giải thích, phục vụ mục
        # "3. Giải thích lý do đề xuất" ở giao diện (thay cho biểu đồ cột tổng quát).
        chi_tiet_list, giai_thich_list = [], []
        for _, row in result.iterrows():
            chi_tiet, giai_thich = self._criteria_breakdown(query, row)
            chi_tiet_list.append(chi_tiet)
            giai_thich_list.append(giai_thich)
        result["Chi tiết tiêu chí"] = chi_tiet_list
        result["Giải thích"] = giai_thich_list

        return result

    def unique_values(self, col: str):
        return sorted(self.df[col].dropna().unique().tolist())