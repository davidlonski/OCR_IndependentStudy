import os
import google.generativeai as genai
import numpy as np
import cv2 as cv
import re

# Configure API key
genai.configure(api_key=os.environ["API_KEY"])
# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
filePath = "../../images/objects.jpg"


# Load the image
img = cv.imread(filePath)
height, width = img.shape[:2]
print(f"height = {height}")
print(f"width = {width}")


# Generates The cordinates of text
def GenerateCoordinates(filePath, objString):
    myfile = genai.upload_file(filePath)
    prompt = f"Return bounding boxes for '{objString}' in the . Format: [xmin, ymin, xmax, ymax]. Only provide one coordinate."
    print(f"Prompt: {prompt}")
    result = model.generate_content([myfile, prompt])
    return result.text



# Generate text with caching
def GenerateText(filePath):
    myfile = genai.upload_file(filePath)
    result = model.generate_content([
        myfile,
        "\n",
        "ONLY Return a list of each object in the image, NOTHING ELSE. if there is no objects return 'no object'"
    ])
    
    return result.text




image_obj = GenerateText(filePath)
print(f"There is a '{image_obj}' in the image")
print()

objects = image_obj.replace('- ', '').split('\n')
# Remove any empty strings
objects = [obj.strip() for obj in objects if obj.strip()]

print(f"The objects in the image are: {objects}")
print()


for obj in objects:
    # Generate coordinates
    text_coordinates = GenerateCoordinates(filePath, obj)
    
    print(f"The object is being processed: {obj}")
    print(f'Text Coordinates Returned: {text_coordinates}')
    print()



    # Using regex to extract numbers
    numbers = re.findall(r'\d+', text_coordinates)

    # Check if we have exactly 4 numbers for coordinates
    ymin, xmin, ymax, xmax = map(int, numbers)

    # Convert coordinates to original dimensions
    xmin = int(xmin / 1000 * width) - 10
    ymin = int(ymin / 1000 * height) - 10
    xmax = int(xmax / 1000 * width) + 10
    ymax = int(ymax / 1000 * height) + 10


    # Draw a rectangle around the detected text position in the resized image
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the original and resized images
    #cv.imshow('rectangle Image for ' + obj, img)

cv.imshow('Final Image', img)


# Wait until a key is pressed and close the image windows
cv.waitKey(0)
cv.destroyAllWindows()
