import streamlit as st
import re
from PIL import Image
import pytesseract
import cv2
from pypdf import PdfReader
import io
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# -------------------- CONFIG --------------------
# ✅ Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


# ✅ Gemini API Key
# loads the .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# -------------------- OCR FUNCTIONS --------------------

def ocr_image(image, lang="eng+hin+guj"):
    """OCR for a single image using Tesseract."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess for cleaner OCR
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(gray, lang=lang)
    return text


def ocr_pdf(pdf_bytes, lang="eng+hin+guj"):
    """Extract OCR text page-by-page from a PDF. Returns list of pages."""
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    all_pages = []

    for page in pdf.pages:
        page_text = ""

        # If text is selectable
        extracted = page.extract_text()
        if extracted:
            page_text += extracted

        # OCR each embedded image
        for img in page.images:
            img_data = io.BytesIO(img.data)
            image = Image.open(img_data)
            page_text += "\n" + ocr_image(image, lang)

        all_pages.append(page_text.strip())

    return all_pages

# -------------------- OCR CLEANING --------------------

def clean_ocr(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'-\s+', '-', text)
    return text

# -------------------- FIELD LABEL TRANSLATOR --------------------

def translate_label(label):
    translations = {
        # Gujarati
        "નામ": "Name",
        "પિતાનું નામ": "Father Name",
        "સરનામું": "Address",
        "જન્મ તારીખ": "Date of Birth",
        "ગામ": "Village",
        "જિલ્લો": "District",
        "રાજ્ય": "State",

        # Hindi
        "नाम": "Name",
        "पिता का नाम": "Father Name",
        "पता": "Address",
        "जन्म तिथि": "Date of Birth",
        "ग्राम": "Village",
        "जिला": "District",
        "राज्य": "State"
    }
    return translations.get(label.strip(), label)

# -------------------- OCR QUALITY CHECK --------------------

def validate_ocr_quality(text):
    score = 0

    if len(text) < 40:
        return "LOW", "OCR text too short. Please rescan clearly."

    if re.search(r'[A-Za-z0-9]', text):
        score += 1

    if len(re.findall(r'[!?@#$%^&*{}~]', text)) < 5:
        score += 1

    if len(text.split()) > 15:
        score += 1

    if score == 3:
        return "HIGH", "OCR quality is good."
    elif score == 2:
        return "MEDIUM", "OCR is okay but not perfect."
    else:
        return "LOW", "OCR quality is low — some fields may be incorrect."

# -------------------- GEMINI MAIN FUNCTION --------------------

def get_ai_instructions(text_pages):
    cleaned_pages = [clean_ocr(p) for p in text_pages]
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(cleaned_pages)

    quality, quality_msg = validate_ocr_quality(full_text)


def get_ai_instructions(text, language="gu"):
    """
    Gemini-powered form helper that outputs in English, Hindi, or Gujarati.
    language options: "en" (English), "hi" (Hindi), "gu" (Gujarati)
    """

    # Language-specific instruction blocks
    lang_block = {
        "en": """
Use very simple English. Speak slowly, clearly, and suitable for rural people.
""",
        "hi": """
सब कुछ बहुत आसान हिंदी में लिखो। ग्रामीण लोगों के लिए बिलकुल सरल भाषा उपयोग करो।
लंबे वाक्य मत बनाओ। छोटे-छोटे पॉइंट्स दो।
""",
        "gu": """
બધું બહુ સરળ ગુજરાતી માં લખો. ગામડાના લોકો સરળતાથી સમજી શકે એવી ભાષા રાખો.
લાંબા વાક્યો ન લખતા. નાના-નાના મુદ્દાઓમાં લખો.
""",
    }

    selected_language_instruction = lang_block.get(language.lower(), lang_block["gu"])

    prompt = f"""
You are a helpful multilingual government-form assistant for Indian citizens,
especially rural users. You speak in very simple English and give clear steps.

Your job:
1. Identify the type of form from the OCR text.
2. Extract all fields you can understand.
3. Fill whatever information is visible.
4. For missing things, ask the user in simple, friendly lines.
5. Give short and simple next steps anyone can follow.
6. DO NOT use JSON. DO NOT use code-like formatting. 
7. Keep everything in plain text, easy to read in the language chosen below.

LANGUAGE INSTRUCTION:
{selected_language_instruction}

FORMAT STRICTLY LIKE THIS:

AUTO-FILLED FORM
(list all fields you understood and filled)

NEED INFO FROM USER
(list questions like: “Please tell your date of birth.”)

NEXT STEPS
(one or two simple steps, e.g., “Please sign and attach your Aadhaar card.”)

Tone must be soft, helpful, and easy for rural users.  
Avoid long sentences.  
If any Hindi or Gujarati text appears, show bilingual labels when useful.

OCR TEXT STARTS BELOW:
{text}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# -------------------- STREAMLIT UI --------------------

st.set_page_config(page_title="Smart Form Scanner", page_icon="🧠", layout="centered")

st.title("🧠 Smart Multilingual Form Scanner (ENG • HIN • GUJ)")
st.write("Upload any form (Image/PDF) and get auto-filled structured output.")

uploaded_file = st.file_uploader("📁 Upload Form", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    with st.spinner("🔍 Extracting text..."):
        if uploaded_file.type == "application/pdf":
            pages = ocr_pdf(uploaded_file.read())
            text_display = "\n\n--- PAGE BREAK ---\n\n".join(pages)
            ai_output = get_ai_instructions(pages)

        else:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            text = ocr_image(img)
            text_display = text
            ai_output = get_ai_instructions([text])

    
    st.subheader("🤖Output")
    st.text_area("AI Output", ai_output, height=300)

    st.download_button(
        "💾 Download JSON",
        data=ai_output,
        file_name="form_analysis.json",
        mime="application/json"
    )

else:
    st.info("⬆ Upload a form to get started.")


