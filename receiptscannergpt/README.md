to# receiptscannergpt
Use gpt/oai api to determine co2 footprint of consumer purchases via receipt

ADV492 Receipt Carbon Footprint Scan

This project provides a Streamlit-based web application for scanning and estimating the carbon footprint of products listed on a receipt. The application uses the OpenAI GPT-4 API to process the receipt and determine the carbon footprint of each item.

Features:
Upload or take a picture of a receipt
Encode and process the image to extract product information and estimated carbon footprint
Display the results in a structured format

Requirements:
Python 3.7 or later
Streamlit
pandas
requests
Pillow (PIL)
OpenAI API key
Setup

Clone the repository:
git clone https://github.com/your-repo/receipt-carbon-footprint-scan.git

Install the required Python packages:
pip install -r requirements.txt

Set up your OpenAI API key:
Ensure you have your OpenAI API key set as an environment variable:
export OPENAI_API_KEY='your-openai-api-key'

Run the Streamlit application:
streamlit run receiptscan_exec.py

Interact with the app:
Upload an image of a receipt or take a picture using the webcam.
The application will process the image and display the product names, prices, and estimated carbon footprint in grams of CO2.

Notes

Ensure your OpenAI API key is valid and properly set as an environment variable.
The application processes JPEG images; ensure your uploaded or captured image is in the correct format.
The response from the OpenAI API is expected to be structured as specified in the payload content.
License

This project is licensed under the MIT License. See the LICENSE file for details.
