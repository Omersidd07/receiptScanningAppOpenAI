import os
import json
import base64
from io import BytesIO

import requests
import pandas as pd
import streamlit as st
from PIL import Image


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def image_file_to_jpeg_bytes(uploaded_file) -> bytes:
    """
    Converts an uploaded image (png/jpg/jpeg) into JPEG bytes.
    This makes downstream base64/image_url handling consistent.
    """
    img = Image.open(uploaded_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def to_base64_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def call_openai_vision(api_key: str, model: str, image_data_url: str) -> dict:
    """
    Calls the OpenAI Chat Completions API with a vision-capable model to:
      1) parse receipt line-items + prices
      2) estimate item-level carbon footprint (g CO2) with brief notes

    Returns parsed JSON (dict). Raises on hard failures.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # Keep the prompt "business friendly" and aligned with your resume bullet.
    # We request strict JSON so the app can reliably display results.
    schema = {
        "name": "receipt_carbon_footprint",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "store": {"type": ["string", "null"]},
                "purchase_date": {"type": ["string", "null"], "description": "If visible, ISO-like or raw receipt date string."},
                "currency": {"type": ["string", "null"], "description": "ISO currency code if obvious, else null."},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": ["number", "null"], "description": "Unit or line price as a number. Null if not visible."},
                            "estimated_g_co2": {"type": ["number", "null"], "description": "Estimated grams of CO2 for this item."},
                            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                            "notes": {"type": ["string", "null"], "description": "Short rationale/category assumption. Keep brief."},
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
        },
        "strict": True,
    }

    prompt = (
        "You are an AI assistant helping extract receipt line items and estimate the carbon footprint.\n"
        "Task:\n"
        "1) Read the receipt image and extract the purchased line items.\n"
        "2) For each item, estimate its carbon footprint in grams of CO2 (g CO2).\n\n"
        "Rules:\n"
        "- Only include actual purchased items (exclude totals/subtotal/tax/cash/change/loyalty lines).\n"
        "- If a price is not visible, set price = null.\n"
        "- Carbon footprint is an estimate. Use typical lifecycle assumptions for the item category.\n"
        "- Keep notes short (e.g., 'produce', 'meat', 'snack', 'household', etc.).\n"
        "- Return strictly valid JSON that matches the provided schema.\n"
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        # Ask for a lot of tokens so bigger receipts fit.
        "max_tokens": 1600,
        # Newer models support structured outputs. If the model doesn't, you'll still often get JSON.
        "response_format": {"type": "json_schema", "json_schema": schema},
        "temperature": 0.2,
    }

    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        # Include the server message to help debug.
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")

    content = resp.json()["choices"][0]["message"]["content"]

    # content should be JSON. If it's a string, parse it.
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        return json.loads(content)

    raise RuntimeError("Unexpected response format from OpenAI API.")


def normalize_items(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    # Make sure numeric columns are numeric when possible
    for col in ["price", "estimated_g_co2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    st.set_page_config(
        page_title="AI Receipt Scanner (Carbon Footprint)",
        page_icon="isee_logo.jpeg",
        layout="centered",
    )

    # Logo (your repo includes these filenames already)
    st.image("isee_logo-removebg-preview-2.png", width=60)
    st.title("AI Receipt Scanner")
    st.caption("Upload a receipt image → extract line items → estimate carbon footprint (g CO2).")

    with st.sidebar:
        st.header("Settings")

        # API key handling: env var preferred, but allow manual entry for demos
        env_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API key",
            value=env_key,
            type="password",
            help="Best practice: set OPENAI_API_KEY as an environment variable.",
        )

        model = st.text_input(
            "Model",
            value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            help="Use a vision-capable model available to your API key.",
        )

        st.markdown("---")
        st.write("Tip: For best results, use a well-lit, high-resolution photo.")

    st.subheader("Upload or take a photo")
    col1, col2 = st.columns(2)
    with col1:
        camera = st.camera_input("Take a picture")
    with col2:
        upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    img_file = camera or upload

    if not img_file:
        st.info("Upload or capture a receipt image to begin.")
        return

    if not api_key:
        st.error("Missing API key. Set OPENAI_API_KEY or paste it in the sidebar.")
        return

    # Show preview
    try:
        st.image(img_file, caption="Receipt preview", use_container_width=True)
    except Exception:
        pass

    # Convert to JPEG bytes + base64 data URL
    try:
        jpeg_bytes = image_file_to_jpeg_bytes(img_file)
        image_data_url = to_base64_data_url(jpeg_bytes)
    except Exception as e:
        st.error(f"Could not read/convert the image: {e}")
        return

    if st.button("Scan receipt", type="primary"):
        with st.spinner("Reading receipt and estimating carbon footprint..."):
            try:
                result = call_openai_vision(api_key=api_key, model=model, image_data_url=image_data_url)
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.success("Done!")

        # Display header info
        meta_cols = st.columns(3)
        meta_cols[0].metric("Store", result.get("store") or "—")
        meta_cols[1].metric("Date", result.get("purchase_date") or "—")
        meta_cols[2].metric("Currency", result.get("currency") or "—")

        items = result.get("items", [])
        if not items:
            st.warning("No items detected. Try a clearer photo or a different crop.")
            st.json(result)
            return

        df = normalize_items(items)

        # Compute totals
        total_g = float(df["estimated_g_co2"].dropna().sum()) if "estimated_g_co2" in df.columns else 0.0
        total_price = float(df["price"].dropna().sum()) if "price" in df.columns else 0.0

        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        sum_cols = st.columns(2)
        sum_cols[0].metric("Sum of item prices (detected)", f"{total_price:,.2f}")
        sum_cols[1].metric("Estimated footprint (sum)", f"{total_g:,.0f} g CO2")

        # Receipt totals if present
        totals = result.get("totals", {})
        if totals:
            st.subheader("Receipt totals (if detected)")
            st.write({k: totals.get(k) for k in ["subtotal", "tax", "total"]})

        with st.expander("Raw JSON (debug)"):
            st.json(result)


if __name__ == "__main__":
    main()
