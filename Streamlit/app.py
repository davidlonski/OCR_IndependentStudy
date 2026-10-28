import os
import streamlit as st
from PIL import Image
import numpy as np
import google.generativeai as genai
import cv2 as cv
import time as t

genai.configure(api_key=os.environ["API_KEY"])

# Title for the Streamlit app
st.title("This is an OCR application")

# Camera input
camera_input = st.camera_input("Take a picture...")

if camera_input is not None:
    start_time = t.time();
    # Convert the captured image to a PIL Image
    image = Image.open(camera_input)

    # Convert the PIL image to a NumPy array (OpenCV format)
    image_np = np.array(image)

    # Convert RGB to BGR format (OpenCV uses BGR by default)
    image_bgr = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)

    # Define the filename
    tempFile = "output.png"

    # Write the image using OpenCV
    cv.imwrite(tempFile, image_np)

    st.success(f"Image saved as {tempFile}")



    # Upload the image file to the Generative AI model
    myfile = genai.upload_file(tempFile)

    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")

    

    # Generate content with the Gemini model
    result = model.generate_content([myfile, "\n", 'WHAT TEXT IS IN THIS IMAGE'])

    end_time = t.time();

    elapse = end_time - start_time;

    # Correct the result
    correctedVer = model.generate_content("Correct the result [" + result.text + "] Return the corrected version. NOTHING ELSE.")

    st.write("====Result after OCR with Gemini====")
    st.write(result.text)

    st.write("====Correction with Gemini====")
    st.write(correctedVer.text)

    st.write("====Time taken====")
    st.write(elapse)
