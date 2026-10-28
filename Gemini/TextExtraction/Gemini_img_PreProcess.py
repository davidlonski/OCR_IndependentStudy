import os
import google.generativeai as genai
import cv2 as cv
import numpy as np

genai.configure(api_key=os.environ["API_KEY"])

#filePath = input("Enter file path : ")

img = cv.imread('TestWritten.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)



cv.imshow('Image Window', img)
cv.imshow('Mean Thresholding', th2)
cv.imshow('Gaussian Thresholding', th3)

cv.waitKey(0)

cv.destroyAllWindows()

"""
cv.imwrite('tamperedImage1.png', img)
cv.imwrite('tamperedImage2.png', th2)
cv.imwrite('tamperedImage3.png', th3)

# Upload the image file
myfile1 = genai.upload_file('tamperedImage1.png')
myfile2 = genai.upload_file('tamperedImage2.png')
myfile3 = genai.upload_file('tamperedImage3.png')


# Generate content with the Gemini model
result1 = genai.GenerativeModel("gemini-1.5-flash").generate_content([myfile1, "\n", 'WHAT TEXT IS IN THIS IMAGE'])
result2 = genai.GenerativeModel("gemini-1.5-flash").generate_content([myfile2, "\n", 'WHAT TEXT IS IN THIS IMAGE'])
result3 = genai.GenerativeModel("gemini-1.5-flash").generate_content([myfile3, "\n", 'WHAT TEXT IS IN THIS IMAGE'])



resultArr1 = result1.text.split()
resultArr2 = result2.text.split()
resultArr3 = result3.text.split()


expected = ""

with open('Expected.txt', 'r') as word:
    expected = word.read()

expectedArr = expected.split()


def compare_arrays(expectedArr, resultArr):
    # Initialize count of matched words
    matched_words = 0

    # Create a set of the expected words to compare easily
    expected_set = set(expectedArr)

    # Iterate through resultArr and count matches
    for word in resultArr:
        if word in expected_set:
            matched_words += 1

    # Calculate the percentage of matched words
    if len(expectedArr) == 0:
        return 0

    match_percentage = (matched_words / len(expectedArr)) * 100

    return match_percentage, matched_words, len(expectedArr)

print()
print("====Expected Words====")
print(expectedArr)
print()
print("====Result after OCR with Gemini====")
print(resultArr1)

print()
print("====Result With Mean Thresholding after OCR with Gemini====")
print(resultArr2)

print()
print("====Result With Gaussian Thresholding after OCR with Gemini====")
print(resultArr3)

print()
print("===Results comparing expected with OCR with Gemini")
match_percentage, matched_words, total_words = compare_arrays(expectedArr, resultArr1)
print(f"Matched Words: {matched_words}/{total_words} ({match_percentage:.2f}% match)")

print("===Results comparing expected with OCR with Gemini and Mean Thresholding")
match_percentage, matched_words, total_words = compare_arrays(expectedArr, resultArr2)
print(f"Matched Words: {matched_words}/{total_words} ({match_percentage:.2f}% match)")

print("===Results comparing expected with OCR with Gemini and Gaussian Thresholding")
match_percentage, matched_words, total_words = compare_arrays(expectedArr, resultArr3)
print(f"Matched Words: {matched_words}/{total_words} ({match_percentage:.2f}% match)")
"""
