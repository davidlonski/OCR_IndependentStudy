import os
import google.generativeai as genai
import numpy as np
import cv2 as cv
import re

# Configure API key
genai.configure(api_key=os.environ["API_KEY"])
# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
filePath = "../../images/U0x2tW.png"


# Load the image
img = cv.imread(filePath)
height, width = img.shape[:2]
print(f"height = {height}")
print(f"width = {width}")


# Generates The cordinates of text
def GenerateCoordinates(filePath, letter):
    myfile = genai.upload_file(filePath)
    prompt = "Return bounding box for the Letter '{Letter}' in the image. Format: [xmin, ymin, xmax, ymax]. Only provide one coordinates."
    result = model.generate_content([myfile, prompt])
    return result.text



# Generate text with caching
def GenerateText(filePath):
    myfile = genai.upload_file(filePath)
    result = model.generate_content([
        myfile,
        "\n",
        "ONLY Return the text in the image, NOTHING ELSE. if there is no text return no"
    ])
    
    return result.text



image_Text = GenerateText(filePath)
#image_Text = "U0x2tW"
print(f'The text in the image is : {image_Text}')

# Generate coordinates for each character
all_coordinates = []
for char in image_Text:
    if char.isalnum():
        text_coordinates = GenerateCoordinates(filePath, char)
        print(f'Coordinates for character "{char}": {text_coordinates}')
        all_coordinates.append(text_coordinates)

print(f'All coordinates collected: {all_coordinates}')

# Process each set of coordinates
for text_coordinates in all_coordinates:
    # Using regex to extract numbers for each coordinate set
    numbers = re.findall(r'\d+', text_coordinates)
    
    # Check if we have enough numbers for coordinates
    if len(numbers) >= 4:
        ymin, xmin, ymax, xmax = map(int, numbers[:4])
        
        # Convert coordinates to original dimensions
        xmin = int(xmin / 1000 * width)
        ymin = int(ymin / 1000 * height)
        xmax = int(xmax / 1000 * width)
        ymax = int(ymax / 1000 * height)
        
        # Draw a rectangle around each detected text position
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

# Display the original and resized images
cv.imshow('rectangle Image', img)

# Wait until a key is pressed and close the image windows
cv.waitKey(0)
cv.destroyAllWindows()
