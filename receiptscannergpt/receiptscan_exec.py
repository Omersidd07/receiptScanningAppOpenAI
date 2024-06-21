import base64
import requests
import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image
import os

#run_loop = True

#while(run_loop):
#run_loop = False
# uploaded_file = st.file_uploader("Choose a file", type=['png','jpg','jpeg'])
# if uploaded_file is not None:
#     # To read file as bytes:
#     bytes_data = uploaded_file.getvalue()

api_key = os.getenv("OPENAI_API_KEY") #requires openai gpt-4 key - replace or set local env variable to key

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
st.set_page_config(page_title="ADV492 Receipt Carbon Footprint Scan", page_icon="isee_logo.jpeg", layout="centered")
st.image("isee_logo-removebg-preview-2.png", width=50)
img_camera_buffer = st.camera_input("Take a picture")
img_choosefile_buffer = st.file_uploader("Choose an image file", accept_multiple_files = False, type=['png','jpg','jpeg'])
data_string = "Est. Carbon Footprint - Please upload or photograph your receipt"
st.text(data_string)

img_file_buffer = None
if img_camera_buffer is not None:
    img_file_buffer = img_camera_buffer
else:
    img_file_buffer = img_choosefile_buffer
    
if img_file_buffer is not None:
    # To read image file buffer as bytes:
    bytes_data = img_file_buffer.getvalue()
    image = Image.open(img_file_buffer)
    with open('temp.jpeg', 'wb') as f:
        image.save(f, format='JPEG')
    # Check the type of bytes_data:
    # Should output: <class 'bytes'>
    #st.write(bytes_data)
    # Getting the base64 string
    
    with open("temp.jpeg","rb") as ifff:
        base64_image = base64.b64encode(ifff.read()).decode('utf-8')
    
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "I am going to give you an image of a receipt.  I want you to output NOTHING ELSE except the following based on the image.  Read the receipt and determine the names and prices of itmes in it, and then estimate in grams of CO2 emitted to produce this product.  Then output each ITEM NAME, PURCHASE PRICE, CARBON FOOTPRINT. SEPARATE ITEM PRICE, PURCHASE PRICE, CARBON FOOTPRINT BY COMMAS, DO NOT OUTPUT ANYTHING ELSE. DELIMIT EACH ENTRY WITH A NEW LINE CHARACTER. Carbon Footprint should always be provided in the following unit (g CO2)"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 1500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_json_contents = response.json()
    
    print(response.json())

    data_string = str(response_json_contents["choices"][0]["message"]["content"])
    #data_string = "ZUCCHINI GREEN, $4.66, 341\nBANANA CAVENDISH, $1.32, 80\nPOTATOES BRUSHED, $4.84, 287\nBROCCOLI, $5.15, 204\nBRUSSEL SPROUTS, $0.99, 45\nGRAPES GREEN, $7.03, 338\nPEAS SNOW, $3.27, 129\nTOMATOES GRAPE, $2.99, 195\nLETTUCE ICEBERG, $2.49, 113"
    lines = data_string.split('\n')

    # Split each line into items
    data = [line.split(',') for line in lines]
    
    print("\nDATA AFTER SPLIT: \n", data,"\n")

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Product', 'Price', "g CO2"])
    
    st.text(data_string)


    