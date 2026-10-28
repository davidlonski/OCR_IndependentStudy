import os
import google.generativeai as genai
import numpy as np
import cv2 as cv
import re

# Configure the API key
genai.configure(api_key=os.environ["API_KEY"])
model = genai.GenerativeModel("gemini-1.5-flash")
filePath = "../../../images\captcha.png"

# Load the image
img = cv.imread(filePath)
height, width = img.shape[:2]
print(f"height = {height}")
print(f"width = {width}")

def GenerateCoordinates(filePath):
    myfile = genai.upload_file(filePath)
    result = model.generate_content([
        myfile,
        "\n",
        "Analyze the image and identify the coordinates of the text. Once you find the text, please return the rectangular region of interest in the format [y1:y2, x1:x2]. " +
        f"Make sure the region is within the image bounds which is height:{height}, width:{width}, and adjusts according to the image size. only return one set of coordinates please."
    ])
    
    return result.text

# Generate coordinates
text_coordinates = GenerateCoordinates(filePath)
print(f'Text Coordinates Returned: {text_coordinates}')

# Using regex to extract numbers
numbers = re.findall(r'\d+', text_coordinates)

# Check if we have exactly 4 numbers for coordinates
y1, y2, x1, x2 = map(int, numbers)
    
# Define the scaling factors
scale_percent = 50  # Scale to 50% of the original size
new_width = int(width * scale_percent / 100)
new_height = int(height * scale_percent / 100)

# Resize the image
resized_image = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)

# Scale the coordinates based on the new image dimensions
scaled_x1 = int(x1 * (new_width / width))
scaled_y1 = int(y1 * (new_height / height))
scaled_x2 = int(x2 * (new_width / width))
scaled_y2 = int(y2 * (new_height / height))

# Draw a rectangle around the detected text position in the resized image
cv.rectangle(resized_image, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 2)

# Display the original and resized images
cv.imshow('Resized Image', resized_image)

# Wait until a key is pressed and close the image windows
cv.waitKey(0)
cv.destroyAllWindows()
