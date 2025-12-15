# AI Receipt Scanner (Streamlit + OpenAI)

Upload (or take) a photo of a receipt and the app will:
1) Extract line items (name + price when visible)
2) Estimate a rough carbon footprint per item (g CO₂)
3) Summarize totals (subtotal/tax/total if detected)
4) Generate a short, personalized “environmental interpretation” + suggestions

> Note: Emissions values are directional estimates (category-based), not audited measurements.

---

## Demo

- Upload a receipt image (JPG/PNG) or use your camera
- Click **Scan receipt**
- View extracted items, totals, and the interpretation
- Click **Scan Another Receipt** to reset

---

## Tech Stack

- Streamlit UI (`app.py`) :contentReference[oaicite:0]{index=0}  
- OpenAI calls + parsing helpers (`receipt_ai.py`) :contentReference[oaicite:1]{index=1}  
- Requests + Pandas + Pillow

---

## Setup (Local)

### 1) Clone and install dependencies
git clone <your-repo-url>
cd <your-repo-folder>

### 2) Set your OpenAI API key (recommended: environment variable)
Windows (PowerShell)

python -m pip install --upgrade pip
pip install -r requirements.txt

setx OPENAI_API_KEY "PASTE_YOUR_KEY_HERE"
# Close and reopen PowerShell after running setx

For just the current session:

$env:OPENAI_API_KEY="PASTE_YOUR_KEY_HERE"

macOS / Linux
export OPENAI_API_KEY="PASTE_YOUR_KEY_HERE"

Optional: Use a .env file for local dev
Inside .env file:
OPENAI_API_KEY=PASTE_YOUR_KEY_HERE
OPENAI_MODEL=gpt-5.2

### 3) Run the app
In terminal command: python -m streamlit run app.py
