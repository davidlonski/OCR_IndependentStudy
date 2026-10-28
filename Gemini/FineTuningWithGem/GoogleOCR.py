import google.generativeai as genai
import os

# Configure the API key
genai.configure(api_key=os.environ["API_KEY"])

# Function to process a single image and get the region of interest
def process_image(image_path):
    
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Upload the image file to the API
    myfile = genai.upload_file(image_path)
    
    # Generate the content by analyzing the image for text coordinates
    result = model.generate_content([
        myfile,
        "\n",
        "Analyze the image and identify the coordinates of the text. Once you find the text, please return the rectangular region of interest in the format [y1:y2, x1:x2]. " +
        "Make sure the region is within the image bounds and adjusts according to the image size."
    ])
    
    return result

# Directory containing images
image_dir = "ocr_dataset"

# Loop through the images and process each one
for i in range(len(os.listdir(image_dir))):  # Adjust to match the number of images or your specific condition
    image_path = os.path.join(image_dir, f"image_{i}.png")
    
    # Process the image and get the result
    roi_result = process_image(image_path)
    
    # Output or save the result as needed
    print(f"Result for image_{i}.png: {roi_result.text}")




