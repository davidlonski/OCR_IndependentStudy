import os
import cv2 as cv
import matplotlib.pyplot as plt
import google.generativeai as genai
import numpy as np
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

def add_noise(image, noise_type='gaussian'):
    """
    Add noise to an image
    noise_type: 'gaussian' or 'salt_pepper'
    """
    if noise_type == "gaussian":
        mean = 0
        sigma = 25
        noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
        noisy_image = cv.add(image, noise)
    elif noise_type == "salt_pepper":
        prob = 0.05
        noisy_image = np.copy(image)
        # Salt
        salt = np.random.random(image.shape) < prob/2
        noisy_image[salt] = 255
        # Pepper
        pepper = np.random.random(image.shape) < prob/2
        noisy_image[pepper] = 0
    
    return noisy_image

def GetText(filePath):
    # Read the image
    img = cv.imread(filePath)
    assert img is not None, "file could not be read, check with os.path.exists()"

    # Add noisy versions
    gaussian_noise = add_noise(img, 'gaussian')
    salt_pepper_noise = add_noise(img, 'salt_pepper')
    cv.imwrite('../../images/gaussian_noise.png', gaussian_noise)
    cv.imwrite('../../images/salt_pepper_noise.png', salt_pepper_noise)



    # Original image
    try:
        temp = GenerateText(filePath)
        print("original : " + temp)
    except ValueError as ve:
        print("Error with original image: Value error occurred")
    
    # Gaussian noise
    try:
        temp = GenerateText("../../images/gaussian_noise.png")
        print("gaussian noise : " + temp)
    except ValueError as ve:
        print("Error with gaussian noise: Value error occurred")
    
    # Salt & pepper noise
    try:
        temp = GenerateText("../../images/salt_pepper_noise.png")
        print("salt & pepper noise : " + temp)
    except ValueError as ve:
        print("Error with salt & pepper noise: Value error occurred")

    # Include the original image in the list
    titles = ['Original Image', 'Gaussian Noise', 'Salt & Pepper Noise']
    images = [img, gaussian_noise, salt_pepper_noise]

    # Create a subplot with 3 rows, 1 image per row
    plt.figure(figsize=(5, 15))  # Adjusted figure size for vertical layout
    for i in range(len(images)):
        plt.subplot(3, 1, i + 1)  # Changed to 3 rows, 1 column
        plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])



    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

main = "../../images"

#print("UUXVRs")
#imgPath = main + "/UUXVRs.png"
#GetText(imgPath)


#print("4NNECI")
#imgPath = main + "/4NNECI.png"
#GetText(imgPath)


print("U0X2tW")
imgPath = main + "/U0x2tW.png"
GetText(imgPath)
