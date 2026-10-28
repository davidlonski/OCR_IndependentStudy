import os

import google.generativeai as genai

genai.configure(api_key=os.environ["API_KEY"])

filePath = "..\\images\\captcha.png"

# Upload the image file
myfile = genai.upload_file(filePath)


model = genai.GenerativeModel("gemini-1.5-flash")


# Generate content with the Gemini model
result = model.generate_content([myfile, "\n", 'WHAT TEXT IS IN THIS IMAGE'])


print(result.text)