# AI Receipt Scanner (Carbon Footprint) – Streamlit

This project is a Streamlit web app that lets you upload (or photograph) a receipt, extracts the purchased line items, and estimates the carbon footprint for each item (in **g CO2**).

It uses:
- Streamlit (UI)
- Python (Pillow, Pandas)
- OpenAI API (vision-capable model for receipt understanding)

## What it does
1. You upload a receipt image (or take a photo in the app).
2. The app extracts purchased items + prices.
3. The app estimates item-level carbon footprint (g CO2) and shows the results in a table.

> Note: Carbon footprint values are estimates based on typical lifecycle assumptions for the detected item category.

## Setup

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Set your OpenAI API key (recommended)
Set an environment variable called `OPENAI_API_KEY`.

**Windows PowerShell**
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

**macOS/Linux**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

You can also paste the key directly in the app sidebar (helpful for demos), but avoid committing keys to code.

### 3) Run the app
```bash
streamlit run receiptscan_exec.py
```

## Optional configuration
- `OPENAI_MODEL` (env var) – set a default model name for the app.
  - Example:
    ```bash
    export OPENAI_MODEL="gpt-4o-mini"
    ```

## Files
- `receiptscan_exec.py` – Streamlit application
- `requirements.txt` – Python dependencies

