"""
gemini_helper.py
Tích hợp Gemini API vào hệ thống gợi ý xe KNN:
  1) parse_need(): hiểu nhu cầu viết bằng ngôn ngữ tự nhiên -> tiêu chí + gợi ý trọng số cho KNN
  2) explain_results(): Gemini tư vấn/diễn giải kết quả Top-K do KNN trả về

KNN vẫn là bộ máy xếp hạng chính; Gemini chỉ đóng vai "giao tiếp" ở đầu vào/đầu ra.
Nếu không có API key hoặc Gemini lỗi -> tự động dùng chế độ dự phòng (từ khoá / mẫu câu).
"""

import json
import os
import re

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Khoá tiêu chí dùng trong backend.py
K_PRICE, K_POWER, K_SEATS = "Giá( VNĐ)", "Công suất(HP)", "Chỗ ngồi"
K_FUEL, K_PURPOSE, K_COND, K_BODY, K_GEAR = "Nhiên liệu", "Mục đích", "Tình trạng", "Kiểu xe", "Hộp số"
GROUPS = [K_PRICE, K_FUEL, K_POWER, K_SEATS, K_PURPOSE, K_COND, K_BODY, K_GEAR]


def _get_client(api_key):
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)


def _match_option(value, options):
    """Khớp giá trị Gemini trả về với danh sách giá trị hợp lệ trong dữ liệu (không phân biệt hoa thường)."""
    if value is None:
        return None
    v = str(value).strip().lower()
    for o in options:
        if str(o).strip().lower() == v:
            return o
    return None


def _clean(raw, options):
    """Kiểm tra / ép kiểu kết quả Gemini về đúng định dạng backend cần."""
    criteria = {}
    try:
        if raw.get("budget_trieu") is not None:
            criteria[K_PRICE] = int(float(raw["budget_trieu"]))
        if raw.get("power_hp") is not None:
            criteria[K_POWER] = int(float(raw["power_hp"]))
        if raw.get("seats") is not None and int(raw["seats"]) in options[K_SEATS]:
            criteria[K_SEATS] = int(raw["seats"])
    except (TypeError, ValueError):
        pass
    for key, field in [(K_FUEL, "fuel"), (K_PURPOSE, "purpose"), (K_COND, "condition"),
                       (K_BODY, "body"), (K_GEAR, "gearbox")]:
        m = _match_option(raw.get(field), options[key])
        if m is not None:
            criteria[key] = m

    weights = {}
    for g in GROUPS:
        try:
            weights[g] = max(0.0, min(1.0, float(raw.get("weights", {}).get(g, 0.125))))
        except (TypeError, ValueError):
            weights[g] = 0.125
    return criteria, weights


# ----------------------------------------------------------------------------
# 1. Hiểu nhu cầu
# ----------------------------------------------------------------------------
def _parse_fallback(text, options):
    t = text.lower()
    raw = {"weights": {}}
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(tỷ|tỉ)", t)
    if m:
        raw["budget_trieu"] = float(m.group(1).replace(",", ".")) * 1000
    else:
        m = re.search(r"(\d+)\s*(triệu|tr)\b", t)
        if m:
            raw["budget_trieu"] = float(m.group(1))
    m = re.search(r"(\d+)\s*(người|chỗ|ghế)", t)
    if m:
        raw["seats"] = int(m.group(1))
    for f, pat in {"Xăng": r"(chạy|xe|máy) xăng", "Dầu": r"(chạy|xe|máy) dầu|diesel",
                   "Hybrid": r"hybrid", "Điện": r"(chạy|xe) điện|\bev\b"}.items():
        if re.search(pat, t):
            raw["fuel"] = f
    if re.search(r"dịch vụ|grab|taxi|kinh doanh", t):
        raw["purpose"] = "Dịch vụ"
    elif re.search(r"gia đình|cá nhân|đi làm", t):
        raw["purpose"] = "Cá nhân & Gia đình"
    if re.search(r"xe cũ|đã qua sử dụng", t):
        raw["condition"] = "Cũ"
    elif re.search(r"xe mới|mới 100", t):
        raw["condition"] = "Mới"
    for kw, b in {"suv": "SUV/CUV", "cuv": "SUV/CUV", "sedan": "Sedan", "mpv": "MPV",
                  "hatchback": "Hatchback", "bán tải": "Bán tải", "coupe": "Coupe", "van": "Thương mại/Van"}.items():
        if re.search(rf"\b{kw}\b", t):
            raw["body"] = b
    if re.search(r"số sàn|\bmt\b", t):
        raw["gearbox"] = "MT"
    elif re.search(r"số tự động|\bat\b|\bcvt\b", t):
        raw["gearbox"] = "AT"
    return raw


def parse_need(text, options, api_key=None):
    """
    options: dict {tên cột: danh sách giá trị hợp lệ} (lấy từ recommender.unique_values).
    Trả về (criteria, weights, nguồn). criteria dùng đúng khoá của backend
    (giá tính bằng TRIỆU VNĐ ở khoá K_PRICE, giao diện sẽ tự nhân 1_000_000).
    """
    client = _get_client(api_key)
    source = "từ khoá (dự phòng)"
    raw = None
    if client is not None:
        try:
            from google.genai import types
            prompt = f"""Bạn là chuyên viên tư vấn mua xe ô tô tại Việt Nam. Hãy chuyển mô tả nhu cầu của khách thành JSON.

Mô tả: \"\"\"{text}\"\"\"

Chỉ trả về JSON đúng schema sau (trường nào khách không nhắc tới thì để null):
{{
  "budget_trieu": ngân sách tối đa tính bằng TRIỆU VNĐ (1 tỷ = 1000),
  "power_hp": công suất mong muốn (HP) hoặc null,
  "seats": số chỗ ngồi hoặc null, một trong {options[K_SEATS]},
  "fuel": một trong {options[K_FUEL]} hoặc null,
  "purpose": một trong {options[K_PURPOSE]} hoặc null,
  "condition": một trong {options[K_COND]} hoặc null,
  "body": một trong {options[K_BODY]} hoặc null,
  "gearbox": một trong {options[K_GEAR]} hoặc null,
  "weights": {{ mỗi khoá trong {GROUPS} là số từ 0 đến 1 thể hiện mức quan trọng
              của tiêu chí đó với khách (khách nhấn mạnh -> cao, không nhắc tới -> 0.1) }}
}}"""
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
            )
            raw = json.loads(resp.text)
            source = "Gemini"
        except Exception as e:
            source = f"từ khoá (Gemini lỗi: {type(e).__name__})"
    if raw is None:
        raw = _parse_fallback(text, options)
    criteria, weights = _clean(raw, options)
    return criteria, weights, source


# ----------------------------------------------------------------------------
# 2. Giải thích kết quả
# ----------------------------------------------------------------------------
def _explain_fallback(result):
    if len(result) == 0:
        return "Không có xe phù hợp."
    top = result.iloc[0]
    lines = [f"**{top['Tên xe']}** là lựa chọn phù hợp nhất (độ phù hợp {top['Độ phù hợp (%)']:.1f}%)."]
    for _, r in result.iloc[1:].iterrows():
        lines.append(f"- {r['Tên xe']}: độ phù hợp {r['Độ phù hợp (%)']:.1f}%.")
    return "\n".join(lines)


def explain_results(query, result, api_key=None):
    """Gemini viết lời tư vấn dựa trên Top-K của KNN. Trả về (văn bản, nguồn)."""
    client = _get_client(api_key)
    if client is None or len(result) == 0:
        return _explain_fallback(result), "mẫu câu (dự phòng)"
    try:
        cols = ["Tên xe", "Giá( VNĐ)", "Nhiên liệu", "Công suất(HP)", "Chỗ ngồi", "Mục đích",
                "Tình trạng", "Kiểu xe", "Hộp số", "Độ phù hợp (%)"]
        cars = result[cols].to_dict("records")
        prompt = f"""Bạn là tư vấn viên bán xe thân thiện, trả lời bằng tiếng Việt, ngắn gọn (tối đa ~200 từ).
Tiêu chí khách đã chọn (giá tính bằng VNĐ): {json.dumps(query, ensure_ascii=False)}
Top xe do thuật toán KNN chọn (Độ phù hợp càng cao càng sát nhu cầu):
{json.dumps(cars, ensure_ascii=False)}

Hãy: (1) nêu xe nên ưu tiên và lý do bám theo tiêu chí; (2) nêu 1 điểm cần cân nhắc của từng xe còn lại;
(3) kết thúc bằng 1 lời khuyên ngắn (vd nên lái thử). Chỉ dùng dữ liệu đã cho, không bịa thông số."""
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text, "Gemini"
    except Exception as e:
        return _explain_fallback(result), f"mẫu câu (Gemini lỗi: {type(e).__name__})"