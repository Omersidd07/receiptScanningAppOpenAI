"""
app.py
------
Streamlit UI only:
- Upload/camera input
- Button clicks + session state
- Display results (table + totals + interpretation)
"""

import os
from typing import Any

import streamlit as st
from PIL import Image

try:
    # Optional: load OPENAI_API_KEY from a .env file during local dev
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from receipt_ai import (
    image_file_to_jpeg_bytes,
    call_openai_receipt_parse,
    result_to_dataframe,
    safe_sum,
    build_environment_story_cached,
    COL_PRICE,
    COL_CO2,
)


def render_api_key_help() -> None:
    """
    If the user forgot to set OPENAI_API_KEY, we stop and show simple instructions.
    (Key is never shown in the UI.)
    """
    st.error("Missing OpenAI API key. Set the environment variable OPENAI_API_KEY (recommended).")

    with st.expander("How to set OPENAI_API_KEY on Windows (PowerShell)"):
        st.code(
            'setx OPENAI_API_KEY "PASTE_YOUR_KEY_HERE"\n'
            "# then close & reopen PowerShell\n"
            "# or set for this session only:\n"
            '$env:OPENAI_API_KEY="PASTE_YOUR_KEY_HERE"',
            language="powershell",
        )

    with st.expander("How to set OPENAI_API_KEY on macOS/Linux (bash/zsh)"):
        st.code('export OPENAI_API_KEY="PASTE_YOUR_KEY_HERE"', language="bash")

    st.stop()


"""Pretty-print money-like numbers."""
def fmt_money(v: Any) -> str:
    try:
        return f"{float(v):,.2f}"
    except Exception:
        return "—"


"""
Streamlit reruns the script top-to-bottom on every interaction.
We use session_state to remember:
- uploader_key: lets us reset file_uploader/camera widgets
- scan_result: stores the receipt JSON after scanning
"""
def init_session_state() -> None:
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = None


"""
Show camera + file upload widgets.
Return whichever one the user provided (camera wins if both exist).
"""
def render_uploader() -> Any:
    st.subheader("Upload or take a photo")

    col1, col2 = st.columns(2)
    with col1:
        camera = st.camera_input("Take a picture", key=f"camera_{st.session_state.uploader_key}")
    with col2:
        uploaded = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg"],
            key=f"upload_{st.session_state.uploader_key}",
        )

    return camera if camera is not None else uploaded


def maybe_preview_image(img_file: Any) -> None:
    """Try to show the receipt image. If preview fails, we still allow scanning."""
    if img_file is None:
        return

    st.write("Receipt preview")
    try:
        img = Image.open(img_file)
        st.image(img, use_container_width=True)
    except Exception:
        st.warning("Could not preview image, but we can still try scanning it.")


def scan_button_flow(api_key: str, model: str, img_file: Any) -> None:
    """
    If user clicks Scan, call OpenAI and store result in session_state.
    """
    if st.button("Scan receipt", type="primary", disabled=(img_file is None)):
        with st.spinner("Scanning receipt and estimating footprint..."):
            try:
                jpeg_bytes = image_file_to_jpeg_bytes(img_file)
                result = call_openai_receipt_parse(api_key=api_key, model=model, jpeg_bytes=jpeg_bytes)
            except Exception as e:
                st.error(f"Scan failed: {e}")
                st.stop()

        st.session_state.scan_result = result


"""
Once we have a scan_result:
- show store/date/currency
- show items table
- show totals + metrics
- show interpretation text (and retry once if empty)
"""
def render_results(api_key: str, model: str) -> None:
    result = st.session_state.scan_result
    if result is None:
        return

    st.success("Done!")

    # Basic receipt metadata
    st.subheader("Results")
    meta_cols = st.columns(3)
    meta_cols[0].write("**Store**")
    meta_cols[0].write(result.get("store") or "—")
    meta_cols[1].write("**Date**")
    meta_cols[1].write(result.get("purchase_date") or "—")
    meta_cols[2].write("**Currency**")
    meta_cols[2].write(result.get("currency") or "—")

    # Items table
    df = result_to_dataframe(result)
    if df.empty:
        st.warning("No line items detected. Try a clearer photo (flat, well-lit, in focus).")
        return

    st.dataframe(df, use_container_width=True)

    # Summary metrics
    price_sum = safe_sum(df[COL_PRICE].tolist() if COL_PRICE in df.columns else [])
    total_g = safe_sum(df[COL_CO2].tolist() if COL_CO2 in df.columns else [])

    sum_cols = st.columns(2)
    sum_cols[0].metric("Sum of item prices (detected)", f"{price_sum:,.2f}")
    sum_cols[1].metric("Estimated footprint (sum)", f"{total_g:,.0f} g CO2")

    # Receipt totals (if detected)
    totals = result.get("totals", {}) or {}
    if any(totals.get(k) is not None for k in ["subtotal", "tax", "total"]):
        st.subheader("Receipt totals (if detected)")
        tcols = st.columns(3)
        tcols[0].metric("Subtotal", fmt_money(totals.get("subtotal")))
        tcols[1].metric("Tax", fmt_money(totals.get("tax")))
        tcols[2].metric("Total", fmt_money(totals.get("total")))

    # Interpretation + suggestions (LLM-generated, cached)
    story = build_environment_story_cached(api_key, model, result, df, total_g)

    # If we somehow got empty output, retry once (helps avoid “blank” UI)
    if not story.strip():
        st.warning("No interpretation text was returned. Retrying once...")
        # Clear any cached env_story_ keys
        for k in list(st.session_state.keys()):
            if str(k).startswith("env_story_"):
                st.session_state.pop(k, None)
        story = build_environment_story_cached(api_key, model, result, df, total_g)

    st.markdown(story if story.strip() else "### Environmental interpretation\n(Unable to generate suggestions for this scan.)")

    # Reset button
    if st.button("Scan Another Receipt"):
        st.session_state.scan_result = None
        st.session_state.uploader_key += 1
        st.rerun()

    # Optional debug JSON (OFF by default)
    if os.getenv("SHOW_DEBUG_JSON", "").lower() in ("1", "true", "yes", "on"):
        with st.expander("Raw JSON (debug)"):
            st.json(result)


def main() -> None:
    st.set_page_config(page_title="AI Receipt Scanner", layout="centered")
    st.title("AI Receipt Scanner")
    st.caption("Upload a receipt image → extract line items → estimate carbon footprint (g CO2).")

    init_session_state()

    # API key stays on backend only (env var), never displayed.
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        render_api_key_help()

    # Default model can be overridden with OPENAI_MODEL
    model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    img_file = render_uploader()

    # If no image and no previous result, show help text and exit early
    if img_file is None and st.session_state.scan_result is None:
        st.info("Upload a receipt photo or take a picture to get started.")
        return

    maybe_preview_image(img_file)
    scan_button_flow(api_key, model, img_file)
    render_results(api_key, model)


if __name__ == "__main__":
    main()
