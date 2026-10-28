import os
import google.generativeai as genai
import cv2

filePath = input("Enter file path : ")

# Configure the API key
genai.configure(api_key=os.environ["API_KEY"])

img = cv2.imread(filePath)

img2 = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

cv2.imshow('Image Window with grayscale', img2)
cv2.imshow('Image Window without grayscale', img)


cv2.waitKey(0)

cv2.destroyAllWindows()




# Upload the image file
myfile = genai.upload_file(filePath)

prompt = input("Enter prompt for image : ")

# Generate content with the Gemini model
result = genai.GenerativeModel("gemini-1.5-flash").generate_content([myfile, "\n", prompt])

# Output the result
print(result.text)
