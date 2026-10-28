import os
import cv2 as cv
import matplotlib.pyplot as plt
import google.generativeai as genai
import numpy as np
import re

# Configure API key
genai.configure(api_key=os.environ["API_KEY"])
# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-flash")


# Generate text with caching
def GenerateText(filePath):
    myfile = genai.upload_file(filePath)
    result = model.generate_content([
        myfile,
        "\n",
        "ONLY Return the text in the image, NOTHING ELSE. if there is no text return no"
    ])
    return result.text

def generateLineCords(imagePath):
    # Load the image
    img = cv.imread(imagePath)
    height, width = img.shape[:2]
    
    myfile = genai.upload_file(imagePath)
    prompt = (
        "Look at this image and identify ONE letter. "
        "Provide coordinates for a vertical line on either the left or right edge of that letter. "
        f"The image is {width} pixels wide and {height} pixels tall. "
        "Return ONLY a pair of coordinates for a vertical line in this format: "
        "(x, 0), (x, height) "
        "where x is the horizontal position between 0 and {width}."
    )
    
    result = model.generate_content([myfile, "\n", prompt])
    print(result.text)

    # Using regex to extract numbers
    numbers = re.findall(r'\d+', result.text)

    if len(numbers) >= 1:  # We only need one x-coordinate
        x = min(int(numbers[0]), width)  # Ensure x is within image width
        return (x, 0), (x, height)
    else:
        # Return a default line position if parsing fails
        return (width//2, 0), (width//2, height)

def drawLines(imagePath, min_spacing=20):
    """
    Draw multiple vertical lines for each letter in the CAPTCHA
    with minimum spacing between lines
    """
    img = cv.imread(imagePath)
    if img is None:
        raise ValueError(f"Could not read image at {imagePath}")
        
    result_img = img.copy()
    used_x = []
    
    # Try to get multiple lines
    attempts = 0
    max_attempts = 12  # Limit attempts to prevent infinite loops
    
    while len(used_x) < 6 and attempts < max_attempts:
        start_point, end_point = generateLineCords(imagePath)
        x = start_point[0]
        
        # Check if this line is far enough from existing lines
        if not used_x or all(abs(x - used) >= min_spacing for used in used_x):
            used_x.append(x)
            cv.line(result_img, start_point, end_point, (0, 255, 0), 2)
        
        attempts += 1
    
    # Sort lines from left to right
    used_x.sort()
    
    return result_img

# Usage
if __name__ == "__main__":
    imgPath = "../../images/plate.jpg"
    try:
        result_image = drawLines(imgPath)
        cv.imshow("CAPTCHA with Lines", result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        # Save the result
        output_path = '../../images/captcha_with_lines.png'
        cv.imwrite(output_path, result_image)
    except Exception as e:
        print(f"Error processing image: {e}")
