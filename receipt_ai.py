"""
receipt_ai.py
--------------
All "AI + parsing" logic:
- Convert receipt image to bytes/data URL
- Ask OpenAI to extract structured receipt JSON (items + totals + emissions)
- Ask OpenAI to generate a human-friendly interpretation + suggestions
- Cache the interpretation so Streamlit reruns don't re-bill 
"""

import json
import hashlib
import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple

import requests
import pandas as pd
import streamlit as st
from PIL import Image


# --- OpenAI endpoints ---
# 1) Chat Completions: used for "image -> structured JSON"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# 2) Responses API: used for "receipt JSON -> nice written interpretation"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


# Column Name Constants (helps readability + avoids repeated strings)
COL_ITEM = "Item"
COL_PRICE = "Price"
COL_CO2 = "Estimated g CO2"
COL_CONF = "Confidence"
COL_NOTES = "Notes"


# ---------------------------
# Image helpers
# ---------------------------

"""
Streamlit gives us an uploaded file / camera buffer.
We convert it to a consistent JPEG byte array so the API input is stable.
"""
def image_file_to_jpeg_bytes(uploaded_file) -> bytes:
    img = Image.open(uploaded_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


"""
    OpenAI image input expects a URL, so using a base64 "data:" URL
    so we don't need to host the image anywhere.
"""
def jpeg_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------
# Receipt extraction prompt + schema
# ---------------------------

"""
We give the model:
1) A plain-English instruction prompt
2) A JSON schema to force a predictable structured response
"""
def build_prompt_and_schema() -> Tuple[str, Dict[str, Any]]:
    prompt = (
        "You are an assistant that extracts structured receipt data from an image.\n"
        "Return ONLY valid JSON that matches the provided schema.\n\n"
        "Task:\n"
        "1) Identify store name if visible.\n"
        "2) Extract each purchased item line (name + price if visible).\n"
        "3) For each item, estimate its carbon footprint in grams of CO2 (estimated_g_co2).\n"
        "   Use common-sense lifecycle assumptions by category (e.g., red meat > poultry > dairy > grains/vegetables).\n"
        "4) Provide a confidence level (low/medium/high) and brief notes for each item.\n"
        "5) If subtotal/tax/total are visible, include them under totals.\n\n"
        "Important:\n"
        "- If you are unsure about an item name, approximate it.\n"
        "- If a price is not visible, set price=null.\n"
        "- If store/date/currency are not visible, set them to null.\n"
        "- Be conservative and consistent in emissions estimates.\n"
    )

    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "store": {"type": ["string", "null"]},
            "purchase_date": {"type": ["string", "null"]},
            "currency": {"type": ["string", "null"]},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": ["number", "null"]},
                        "estimated_g_co2": {"type": ["number", "null"]},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "notes": {"type": ["string", "null"]},
                    },
                    "required": ["name", "price", "estimated_g_co2", "confidence", "notes"],
                },
            },
            "totals": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subtotal": {"type": ["number", "null"]},
                    "tax": {"type": ["number", "null"]},
                    "total": {"type": ["number", "null"]},
                },
                "required": ["subtotal", "tax", "total"],
            },
        },
        "required": ["store", "purchase_date", "currency", "items", "totals"],
    }
    return prompt, schema


# ---------------------------
# OpenAI calls (receipt parsing)
# ---------------------------

"""
Sends the receipt image to OpenAI and requests strict JSON output that matches the schema.
Returns: a Python dictionary with store, items, totals, etc.
"""
def call_openai_receipt_parse(api_key: str, model: str, jpeg_bytes: bytes) -> Dict[str, Any]:
    prompt, schema = build_prompt_and_schema()
    image_url = jpeg_bytes_to_data_url(jpeg_bytes)

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Respond with JSON only."},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "receipt_carbon_schema", "schema": schema},
        },
        # For some newer models, the correct parameter is max_completion_tokens
        "max_completion_tokens": 900,
    }

    resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = str(data["choices"][0]["message"]["content"]).strip()

    # Sometimes models wrap JSON in ```...``` fences, so we strip that.
    if content.startswith("```"):
        content = content.strip("`").replace("json", "", 1).strip()

    return json.loads(content)


# ---------------------------
# Result formatting helpers
# ---------------------------

"""
Convert the extracted JSON into a table so it can be shown in Streamlit.
"""
def result_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    items = result.get("items", []) or []
    rows = []
    for it in items:
        rows.append(
            {
                COL_ITEM: it.get("name"),
                COL_PRICE: it.get("price"),
                COL_CO2: it.get("estimated_g_co2"),
                COL_CONF: it.get("confidence"),
                COL_NOTES: it.get("notes"),
            }
        )
    return pd.DataFrame(rows)


"""
Add up a list of numbers where some entries may be None / missing / not numeric.
"""
def safe_sum(values: List[Any]) -> float:
    total = 0.0
    for v in values:
        try:
            if v is None:
                continue
            total += float(v)
        except Exception:
            continue
    return total


# ---------------------------
# OpenAI calls (interpretation / suggestions)
# ---------------------------
"""
Use OpenAI to generate human-readable markdown:
- what the estimate means
- top drivers
- personalized suggestions based on this receipt
- uncertainty notes

Uses the Responses API because it tends to be reliable for "text output" style calls.
"""
def call_openai_environment_story(api_key: str, model: str, receipt: Dict[str, Any], total_g: float) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Only send what we detected (so we don't hallucinate extra items).
    safe_payload = {
        "store": receipt.get("store"),
        "purchase_date": receipt.get("purchase_date"),
        "currency": receipt.get("currency"),
        "totals": receipt.get("totals"),
        "items": receipt.get("items", []),
        "estimated_total_g_co2": total_g,
    }

    instructions = (
        "You are an assistant helping a user understand the environmental impact of a grocery receipt scan.\n"
        "Use ONLY the provided receipt data. Do NOT invent items, prices, or categories.\n"
        "If an item has null estimated_g_co2, treat it as unknown and mention limitations.\n\n"
        "Return Markdown (not JSON) with these sections:\n"
        "### Environmental interpretation\n"
        "2–3 sentences explaining what the estimate represents and that it is directional.\n\n"
        "### Top drivers\n"
        "List up to 3 items with the highest estimated_g_co2. For each: item name, estimated g CO₂, and % share.\n"
        "If fewer than 3 items have estimates, list what you can and say why.\n\n"
        "### Personalized suggestions\n"
        "Give 4–6 actionable suggestions tailored to THIS receipt.\n"
        "Make suggestions specific to the highest-impact items you see (e.g., swaps, frequency reduction, alternatives).\n"
        "Keep it practical and non-judgmental.\n\n"
        "### Notes on uncertainty\n"
        "1–2 bullets about confidence/limitations based on the 'confidence' fields.\n\n"
        "Keep the whole response under ~180 words."
    )

    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": "Receipt data:\n" + json.dumps(safe_payload, ensure_ascii=False),
        "reasoning": {"effort": "none"},
        "temperature": 0.35,
        "max_output_tokens": 350,
        "text": {"format": {"type": "text"}},
    }

    resp = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")

    data = resp.json()

    # 1) Best case: the API returns a direct output_text string
    out = (data.get("output_text") or "").strip()
    if out:
        return out

    # 2) Fallback: extract any text chunks from the output structure
    parts: List[str] = []
    for item in data.get("output", []) or []:
        content = item.get("content", [])
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue

                txt = c.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)

                ot = c.get("output_text")
                if isinstance(ot, dict):
                    ot_txt = ot.get("text")
                    if isinstance(ot_txt, str) and ot_txt.strip():
                        parts.append(ot_txt)

    out = "\n".join(parts).strip()
    if out:
        return out

    # 3) Last resort: return a non-empty, safe message so the UI never looks broken
    return (
        "### Environmental interpretation\n"
        "The receipt was successfully scanned, but a detailed narrative could not be generated.\n\n"
        "### Top drivers\n"
        "This estimate is driven by the highest-emission items detected on the receipt.\n\n"
        "### Personalized suggestions\n"
        "- Reduce frequency of the highest-impact items on the receipt.\n"
        "- Swap to lower-impact alternatives where possible.\n"
        "- Treat this estimate as directional, not exact.\n\n"
        "### Notes on uncertainty\n"
        "- Emissions values are category-based estimates and depend on item interpretation."
    )


"""
Streamlit reruns the script a lot:
This cache prevents from paying for the same interpretation repeatedly.
"""
def build_environment_story_cached(api_key: str, model: str, receipt: Dict[str, Any], df: pd.DataFrame, total_g: float) -> str:
    if df.empty or total_g <= 0:
        return (
            "### Environmental interpretation\n"
            "I couldn't confidently estimate the footprint from this receipt. "
            "Try a clearer photo (well-lit, flat, in focus) and re-scan."
        )

    # Cache key based on receipt items + total estimate
    cache_src = json.dumps(receipt.get("items", []), sort_keys=True, ensure_ascii=False) + "|" + str(total_g)
    cache_key = "env_story_" + hashlib.sha256(cache_src.encode("utf-8")).hexdigest()

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    story = call_openai_environment_story(api_key=api_key, model=model, receipt=receipt, total_g=total_g)
    st.session_state[cache_key] = story
    return story
