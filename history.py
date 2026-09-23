import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"


def load_history() -> list:
    """Đọc toàn bộ lịch sử từ file JSON. Trả về [] nếu file chưa tồn tại hoặc lỗi."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # File hỏng hoặc không đọc được -> coi như chưa có lịch sử, không làm sập app
        return []


def save_search(query: dict, weights: dict, top_car_names: list) -> None:
    """Thêm 1 lượt tìm kiếm mới vào cuối file JSON."""
    history = load_history()
    entry = {
        "thoi_gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tieu_chi": query,
        "trong_so": {k: round(v, 3) for k, v in weights.items()},
        "ket_qua_hang_dau": top_car_names,
    }
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def clear_history() -> None:
    """Xóa toàn bộ lịch sử (xóa hẳn file JSON)."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)