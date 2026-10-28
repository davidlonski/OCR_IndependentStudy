import os
import google.generativeai as genai
import numpy as np
import cv2 as cv
import re
import time
from google.api_core import retry

# Configure API key
genai.configure(api_key=os.environ["API_KEY"])
# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-flash")



filePath = "images/noisy_objects_salt_pepper_1.0.jpg"

# Load the image
img = cv.imread(filePath)
height, width = img.shape[:2]


# Generates The cordinates of text
def GenerateCoordinates(filePath, objString, max_retries=3, delay=1):
    """
    Generate coordinates with retry logic and delay
    """
    for attempt in range(max_retries):
        try:
            myfile = genai.upload_file(filePath)
            prompt = f"Return bounding boxes for '{objString}' in the . Format: [xmin, ymin, xmax, ymax]. Only provide one coordinate."
            print(f"Prompt: {prompt}")
            result = model.generate_content([myfile, prompt])
            return result.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if all retries failed



# Generate text with caching
def GenerateText(filePath, max_retries=3, delay=1):
    """
    Generate text with retry logic and delay
    """
    for attempt in range(max_retries):
        try:
            myfile = genai.upload_file(filePath)
            result = model.generate_content([
                myfile,
                "\n",
                "ONLY Return a list of each object in the image, NOTHING ELSE. if there is no objects return 'no object'"
            ])
            return result.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if all retries failed



def DetectingObjects(filePath):
    try:
        image_obj = GenerateText(filePath)
        print(f"There is a '{image_obj}' in the image")
        print()

        objects = image_obj.replace('- ', '').split('\n')
        objects = [obj.strip() for obj in objects if obj.strip()]

        print(f"The objects in the image are: {objects}")
        print()

        for obj in objects:
            # Add delay between object processing
            time.sleep(1)  # Wait 1 second between objects
            
            try:
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

            except Exception as e:
                print(f"Error processing object {obj}: {str(e)}")
                continue  # Skip to next object if there's an error

    except Exception as e:
        print(f"Error in DetectingObjects: {str(e)}")

    cv.imshow('Final Image', img)

    cv.imwrite("images/detected_objects_1.0.jpg", img)


    # Wait until a key is pressed and close the image windows
    cv.waitKey(0)
    cv.destroyAllWindows()


DetectingObjects(filePath)
