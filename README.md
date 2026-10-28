# OCR and Object Recognition with a Vision-Language Model

An undergraduate independent study, fall 2024 to March 2025. It tests whether Gemini 1.5 Flash can work as an OCR engine that also locates what it reads, using photos, CAPTCHAs, synthetic text, noise and composite images, with OpenCV for the image processing.

**Research report:** [Research Report 03 (PDF)](https://www.davidlonski.com/research/ocr-object-recognition/report.pdf), which re-analyzes this code and its saved outputs. Project page: [davidlonski.com/research/ocr-object-recognition](https://www.davidlonski.com/research/ocr-object-recognition).

## What the study found

- **Reading text.** The model read product labels, plates and CAPTCHAs. Shown the distorted CAPTCHA "U0x2tW", it returned "U - 0 x 2 tw.", wrong only in the case of the final W.
- **Robustness to noise.** On a ten-step salt-and-pepper ladder, it boxed all nine objects in a test scene up to intensity 0.6, where 88% of pixels carried noise, and still found three at 1.0.
- **Out-of-bounds boxes.** Gemini returns boxes on a 0-to-1,000 scale in `[ymin, xmin, ymax, xmax]` order. The early region scripts used the replies as pixels, so the boxes seemed to fall outside the image. Dividing by 1,000 and multiplying by the image size (added October 30, 2024) is the correct conversion.
- **Unfilled prompts.** Two prompts are plain strings, not f-strings, so the model receives the literal placeholders: the region prompt never states the image size, and the CAPTCHA prompt never names the character (see Known issues).
- **"bapplea".** Recorded at the time as invented text, it most likely read the overlapping words "apple" and "banana" in the uncropped composite image.

## Repository layout

| Path | What it holds |
|---|---|
| `Streamlit/app.py` | The camera OCR app: take a photo, have Gemini read it, then run a second pass to correct the text |
| `Gemini/TextExtraction/` | Text reading, text regions, region rescaling, multi-box detection and per-character CAPTCHA boxes |
| `Gemini/Challenge/` | The noise ladder (`Noising.py`, `ObjectDetection.py`, `NoisingAndDetection.py`) and its 20 saved images |
| `Gemini/CaptchaWithGem/` | CAPTCHA drawing with added noise and lines |
| `Gemini/FineTuningWithGem/` | Synthetic text images and labels (`dataSetCreation.py`), CAPTCHA generation, and a Google Cloud Vision comparison (`GoogleOCR.py`) |
| `Gemini/Simple_Gemini/` | First experiments with the Gemini API |
| `openCv_Studies/` | OpenCV studies: thresholding, regions of interest, blurring, morphology and resizing |
| `Ai_Detector/` | A RoBERTa detector for AI-generated text |
| `IndependentStudy.ipynb` | The final notebook (Colab): CAPTCHA reading, image warping, noise functions, the apple and banana composite, a Gaussian perturbation, FGSM on ResNet-18, and a Tesseract baseline |
| `images/` | The test images |

### Experiments

| Experiment | Date | Code |
|---|---|---|
| Camera OCR and correction | Oct 2, 2024 | `Streamlit/app.py` |
| Text regions | Oct 2–30, 2024 | `Gemini/TextExtraction/Gemini_img_ROIfail.py` |
| Region rescaling | Oct 16, 2024 | `Gemini/TextExtraction/Gemini_img_Scalling.py` |
| Synthetic text set | Oct 16, 2024 | `Gemini/FineTuningWithGem/dataSetCreation.py` |
| CAPTCHA characters | Oct 30, 2024 | `Gemini/TextExtraction/Gemini_img_Boxes.py` |
| Noise ladder | Nov 6–20, 2024 | `Gemini/Challenge/Noising.py`, `ObjectDetection.py` |
| Apple and banana composite | Dec 2024 | `IndependentStudy.ipynb` |
| Gaussian perturbation and FGSM | by Mar 3, 2025 | `IndependentStudy.ipynb` |

## Setup

Python 3.12 was used. Install the packages the scripts import:

```bash
pip install google-generativeai opencv-python numpy matplotlib pillow streamlit captcha requests beautifulsoup4 pytesseract
```

The notebook also uses `torch`, `torchvision` and `torchattacks`; `Ai_Detector` uses `transformers`; `GoogleOCR.py` uses `google-cloud-vision`. `pytesseract` needs the Tesseract program installed.

The scripts read a Gemini API key from the `API_KEY` environment variable:

```bash
export API_KEY="your-key"        # macOS and Linux
setx API_KEY "your-key"          # Windows, then open a new terminal
```

The scripts call `gemini-1.5-flash` through the `google-generativeai` package. Google has since retired the 1.5 models, so change the model name to a current one before running them.

## Running

Image paths are relative to each script's folder, so run scripts from inside their folder:

```bash
cd Gemini/TextExtraction
python Gemini_img_OCR.py
```

Start the camera app with:

```bash
streamlit run Streamlit/app.py
```

`IndependentStudy.ipynb` was written for Google Colab and reads its images from `/content/`.

## Known issues

These are left as they were during the study, so the code matches the report:

- `Gemini/TextExtraction/Gemini_img_ROIfail.py`: the prompt's second line is not an f-string, so the model receives the text `height:{height}, width:{width}` instead of the image size.
- `Gemini/TextExtraction/Gemini_img_Boxes.py`: the prompt is not an f-string, so every request asks for the letter `'{Letter}'` and none names the character.
- `Gemini/TextExtraction/Gemini_img_Scalling.py` uses the model's reply as pixel coordinates without converting from the 0-to-1,000 scale.
- `Gemini_img_ROIfail.py` and `Gemini_img_Scalling.py` point to images three folders up, outside the repository. The images are in `images/` (the license plate is `licensesPlate.jpg`).
- In the notebook, the Gaussian perturbation (`np.random.normal(0, 0.1, ...)` converted to `uint8`) truncates to zero, so the perturbed image is identical to the original.

## Related work

The study continued in summer 2025 as an office-file accessibility and quiz tool: [omni-text](https://github.com/davidlonski/omni-text) (OCR for Office files) and [RAG-Dev](https://github.com/davidlonski/RAG-Dev) (retrieval-augmented quiz generation).

## Author

David Lonski.
