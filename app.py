from flask import Flask, render_template, request, jsonify, redirect, session
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import re
import tensorflow_datasets as tfds
import base64 

# Import googletrans for dynamic translation
from googletrans import Translator as GoogleTranslator, LANGUAGES

# Import your custom modules
from nlp_fuzzy import extract_symptoms
from symptom_db import plant_disease_db 

from auth import auth_bp
from admin_routes import admin_bp

# ---------------------------------------------------------
#           TRANSLATION LAYER (API LOGIC)
# ---------------------------------------------------------

# Initialize the Google Translator (using the unofficial library)
google_translator = GoogleTranslator()

class Translator:
    """
    Handles static application labels and calls Google API for dynamic content.
    """

    APP_LABELS = {
        'en': {
            'disease_detected': "Disease Detected",
            'crop': "Crop",
            'confidence': "Confidence",
            'cause': "Cause",
            'treatment': "Treatment",
            'symptoms': "Symptoms",
            'consult_expert': "Consult a local agricultural expert."
        },
        'hi': {
            'disease_detected': "रोग की पहचान",
            'crop': "फ़सल",
            'confidence': "विश्वसनीयता",
            'cause': "कारण",
            'treatment': "उपचार",
            'symptoms': "लक्षण",
            'consult_expert': "स्थानीय कृषि विशेषज्ञ से सलाह लें।"
        },
        'or': {
            'disease_detected': "ରୋଗ ଚିହ୍ନଟ",
            'crop': "ଫସଲ",
            'confidence': "ଆତ୍ମବିଶ୍ୱାସ",
            'cause': "କାରଣ",
            'treatment': "ଚିକିତ୍ସା",
            'symptoms': "ଲକ୍ଷଣ",
            'consult_expert': "ସ୍ଥାନୀୟ କୃଷି ବିଶେଷଜ୍ଞଙ୍କ ସହିତ ପରାମର୍ଶ କରନ୍ତୁ।"
        },
        # --- NEW LANGUAGE LABELS ---
        'bn': { # Bengali
            'disease_detected': "রোগ শনাক্তকরণ",
            'crop': "ফসল",
            'confidence': "বিশ্বাসযোগ্যতা",
            'cause': "কারণ",
            'treatment': "চিকিৎসা",
            'symptoms': "লক্ষণ",
            'consult_expert': "স্থানীয় কৃষি বিশেষজ্ঞের সাথে পরামর্শ করুন।"
        },
        'mr': { # Marathi
            'disease_detected': "रोगाची ओळख",
            'crop': "पीक",
            'confidence': "विश्वासार्हता",
            'cause': "कारण",
            'treatment': "उपचार",
            'symptoms': "लक्षणे",
            'consult_expert': "स्थानिक कृषी तज्ञांचा सल्ला घ्या."
        },
        'te': { # Telugu
            'disease_detected': "వ్యాధి నిర్ధారణ",
            'crop': "పంట",
            'confidence': "విశ్వసనీయత",
            'cause': "కారణం",
            'treatment': "చికిత్స",
            'symptoms': "లక్షణాలు",
            'consult_expert': "స్థానిక వ్యవసాయ నిపుణుడిని సంప్రదించండి."
        },
        'ta': { # Tamil
            'disease_detected': "நோய் கண்டறிதல்",
            'crop': "பயிர்",
            'confidence': "நம்பிக்கை",
            'cause': "காரணம்",
            'treatment': "சிகிச்சை",
            'symptoms': "அறிகுறிகள்",
            'consult_expert': "உள்ளூர் வேளாண் நிபுணரை அணுகவும்."
        },
        'gu': { # Gujarati
            'disease_detected': "રોગની ઓળખ",
            'crop': "પાક",
            'confidence': "વિશ્વાસ",
            'cause': "કારણ",
            'treatment': "સારવાર",
            'symptoms': "લક્ષણો",
            'consult_expert': "સ્થાનિક કૃષિ નિષ્ણાતની સલાહ લો."
        },
        'ur': { # Urdu
            'disease_detected': "بیماری کی شناخت",
            'crop': "فصل",
            'confidence': "اعتماد",
            'cause': "وجہ",
            'treatment': "علاج",
            'symptoms': "علامات",
            'consult_expert': "مقامی زرعی ماہر سے مشورہ کریں۔"
        },
        'kn': { # Kannada
            'disease_detected': "ರೋಗ ಪತ್ತೆ",
            'crop': "ಬೆಳೆ",
            'confidence': "ವಿಶ್ವಾಸಾರ್ಹತೆ",
            'cause': "ಕಾರಣ",
            'treatment': "ಚಿಕಿತ್ಸೆ",
            'symptoms': "ಲಕ್ಷಣಗಳು",
            'consult_expert': "ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."
        },
        'ml': { # Malayalam
            'disease_detected': "രോഗം തിരിച്ചറിയൽ",
            'crop': "വിള",
            'confidence': "വിശ്വാസ്യത",
            'cause': "കാരണം",
            'treatment': "ചികിത്സ",
            'symptoms': "ലക്ഷണങ്ങൾ",
            'consult_expert': "പ്രാദേശിക കാർഷിക വിദഗ്ദ്ധനെ സമീപിക്കുക."
        },
        'pa': { # Punjabi
            'disease_detected': "ਰੋਗ ਦੀ ਪਛਾਣ",
            'crop': "ਫਸਲ",
            'confidence': "ਵਿਸ਼ਵਾਸ",
            'cause': "ਕਾਰਨ",
            'treatment': "ਇਲਾਜ",
            'symptoms': "ਲੱਛਣ",
            'consult_expert': "ਸਥਾਨਕ ਖੇਤੀਬਾੜੀ ਮਾਹਿਰ ਨਾਲ ਸਲਾਹ ਕਰੋ।"
        },
    }
    
    # Static translations (Only for Crop names - ensures consistent translation)
    STATIC_DB_MAP = {
        'hi': {"Potato": "आलू", "Tomato": "टमाटर", "Pepper": "मिर्च", "Apple": "सेब", "Corn": "मक्का", "Grapes": "अंगूर"},
        'or': {"Potato": "ଆଳୁ", "Tomato": "ଟମାଟୋ", "Pepper": "ଲଙ୍କା", "Apple": "ସେଓ", "Corn": "ମକା", "Grapes": "ଅଙ୍ଗୁର"},
        
        # --- NEW LANGUAGE CROP NAMES ---
        'bn': {"Potato": "আলু", "Tomato": "টমেটো", "Pepper": "গোলমরিচ", "Apple": "আপেল", "Corn": "ভুট্টা", "Grapes": "আঙ্গুর"},
        'mr': {"Potato": "बटाटा", "Tomato": "टोमॅटो", "Pepper": "मिरची", "Apple": "सफरचंद", "Corn": "मका", "Grapes": "द्राक्षे"},
        'te': {"Potato": "బంగాళాదుంప", "Tomato": "టొమాటో", "Pepper": "మిరియాలు", "Apple": "యాపిల్", "Corn": "మొక్కజొన్న", "Grapes": "ద్రాక్ష"},
        'ta': {"Potato": "உருளைக்கிழங்கு", "Tomato": "தக்காளி", "Pepper": "மிளகாய்", "Apple": "ஆப்பிள்", "Corn": "சோளம்", "Grapes": "திராட்சை"},
        'gu': {"Potato": "બટાકા", "Tomato": "ટામેટાં", "Pepper": "મરચું", "Apple": "સફરજન", "Corn": "મકાઈ", "Grapes": "દ્રાક્ષ"},
        'ur': {"Potato": "آلو", "Tomato": "ٹماٹر", "Pepper": "مرچ", "Apple": "سیب", "Corn": "مکئی", "Grapes": "انگور"},
        'kn': {"Potato": "ಆಲೂಗಡ್ಡೆ", "Tomato": "ಟೊಮೆಟೊ", "Pepper": "ಮೆಣಸು", "Apple": "ಸೇಬು", "Corn": "ಮೆಕ್ಕೆಜೋಳ", "Grapes": "ದ್ರಾಕ್ಷಿ"},
        'ml': {"Potato": "ഉരുളക്കിഴങ്ങ്", "Tomato": "തക്കാളി", "Pepper": "കുരുമുളക്", "Apple": "ആപ്പിൾ", "Corn": "ചോളം", "Grapes": "മുന്തിരി"},
        'pa': {"Potato": "ਆਲੂ", "Tomato": "ਟਮਾਟਰ", "Pepper": "ਮਿਰਚ", "Apple": "ਸੇਬ", "Corn": "ਮੱਕੀ", "Grapes": "ਅੰਗੂਰ"},
    }

    def get_labels(self, lang_code: str) -> dict:
        return self.APP_LABELS.get(lang_code, self.APP_LABELS['en'])

    def translate_output(self, text: str, target_lang: str) -> str:
        """Translates dynamic output content from English to target language."""
        if target_lang == 'en':
            return text
        
        # 1. Try static translation for known crop names (Faster and consistent)
        static_map = self.STATIC_DB_MAP.get(target_lang, {})
        if text in static_map:
            return static_map[text]
            
        # 2. Use the Google Translate API for all dynamic content
        try:
            translation = google_translator.translate(text, src='en', dest=target_lang)
            return translation.text
        except Exception as e:
            print(f"Translation API error: {e}")
            return text 


translator = Translator()


# ---------------------------------------------------------
#          INPUT TRANSLATION (API-based, essential for NLP)
# ---------------------------------------------------------

def translate_input_to_en(text: str, source_lang: str) -> str:
    """
    Translates user input to English for NLP processing.
    """
    if source_lang == 'en':
        return text

    # If the input is in the native script, translate it using the chosen language code
    if any(ord(char) > 127 for char in text):
        try:
            translation = google_translator.translate(text, src=source_lang, dest='en')
            return translation.text
        except Exception as e:
            print(f"Input Translation API error: {e}")
            return "ERROR_SCRIPT_INPUT_TRANSLATION_FAIL" 

    return text # Pass Romanized text directly


# ---------------------------------------------------------
#       FLASK CONFIGURATION & MODEL LOADING
# ---------------------------------------------------------

app = Flask(__name__, template_folder='templates')
app.secret_key = "super-secret-key"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)


IMG_SIZE = (128, 128)
MODEL_PATH = "trained_plant_disease_multiclass_model.keras"

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH) 
    print(f"TensorFlow Model loaded from {MODEL_PATH}")
except Exception as e:
    MODEL = None
    print(f"Error loading model: {e}")

try:
    ds_info = tfds.builder('plant_village').info
    CLASS_NAMES = ds_info.features['label'].names
    NUM_CLASSES = len(CLASS_NAMES)
except Exception as e:
    CLASS_NAMES = ["Potato___Early_blight", "Potato___Late_blight", "Tomato___Early_blight", "Tomato___Bacterial_spot", "Corn_(maize)___Common_rust_", "Apple___Apple_scab"]
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Error loading class names (using hardcoded fallback): {e}")


# ---------------------------------------------------------
#                   HELPER FUNCTIONS
# ---------------------------------------------------------

def get_plant_from_disease(disease_name):
    parts = disease_name.split('___')
    if parts:
        return parts[0].replace('_(maize)', '')
    return 'Unknown'


def get_plant_from_text(text):
    for plant in plant_disease_db.keys():
        if re.search(r'\b' + re.escape(plant) + r'\b', text, re.IGNORECASE):
            return plant
    return 'Unknown'


# ---------------------------------------------------------
#               IMAGE PREDICTION LOGIC
# ---------------------------------------------------------

def predict_image(image_file, lang_code):
    if not MODEL:
        return {'error': 'Model not loaded.'}

    labels = translator.get_labels(lang_code)

    try:
        # 1. Read the full file content for both encoding and prediction
        image_content = image_file.read()
        
        # 2. Encode to Base64 for returning the image to the frontend
        base64_encoded_image = base64.b64encode(image_content).decode('utf-8')
        image_data_url = f"data:{image_file.mimetype};base64,{base64_encoded_image}"
        
        # 3. Use the content in a BytesIO object for PIL/TensorFlow processing
        img = Image.open(io.BytesIO(image_content)).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_final = np.expand_dims(img_array, axis=0)

        predictions = MODEL.predict(img_final)

        if predictions.size == 0 or predictions.shape[0] != 1 or len(CLASS_NAMES) == 0:
            return {'error': 'Invalid prediction output.'}

        pred_idx = np.argmax(predictions[0])
        if pred_idx >= len(CLASS_NAMES):
            return {'error': 'Label index mismatch.'}

        confidence = float(np.max(predictions[0]) * 100)
        disease_name = CLASS_NAMES[pred_idx]
        plant_name = get_plant_from_disease(disease_name)

        def clean_name(n):
            return n.split('___')[-1].replace('_', ' ').strip().title()

        cnn_key = clean_name(disease_name)

        english_cause = f"Disease detected via image classification: {disease_name}"
        english_treatment = translator.APP_LABELS['en']['consult_expert']
        english_symptoms = []
        best_key = None

        full_cnn_key = f"{plant_name} {cnn_key}".strip()

        if plant_name in plant_disease_db and full_cnn_key in plant_disease_db[plant_name]:
            data = plant_disease_db[plant_name][full_cnn_key]
            best_key = full_cnn_key
            english_cause = data['cause']
            english_treatment = data['treatment']
            english_symptoms = [best_key] 

        else:
            results = extract_symptoms(
                disease_name.replace('___', ' ').replace('_', ' '),
                plant_name,
            )
            if results:
                m = results[0]
                best_key = m['symptom_key']
                english_cause = m['cause']
                english_treatment = m['treatment']
                english_symptoms = [best_key]
            elif plant_name in plant_disease_db:
                english_symptoms = list(plant_disease_db[plant_name].keys())

        # Output Translation
        translated_cause = translator.translate_output(english_cause, lang_code)
        translated_treatment = translator.translate_output(english_treatment, lang_code)
        translated_symptoms = [
            translator.translate_output(s, lang_code) for s in english_symptoms
        ]

        translated_disease = translator.translate_output(
            best_key if best_key else disease_name.replace('___', ' '),
            lang_code
        )
        
        if not best_key:
             translated_disease = translated_disease.replace('___', ' ')

        translated_plant_name = translator.translate_output(plant_name, lang_code)

        return {
            'type': 'image_symptom',
            'disease_name': translated_disease,
            'crop': translated_plant_name, 
            'confidence': confidence,
            'cause': translated_cause,
            'treatment': translated_treatment,
            'symptoms': translated_symptoms,
            'uploaded_image_b64': image_data_url,
            'labels': labels
        }

    except Exception as e:
        return {'error': f'Image processing error: {str(e)}'}


# ---------------------------------------------------------
#                   TEXT ANALYSIS LOGIC
# ---------------------------------------------------------

def analyze_text(text, lang_code):

    labels = translator.get_labels(lang_code)

    # CRITICAL: Translate input to English for NLP processing
    text_proc = translate_input_to_en(text, lang_code)

    def error(message_key):
        
        if message_key == 'SCRIPT_ERROR_TRANSLATION_FAIL':
            # Specific error for API translation failure
            if lang_code == 'hi':
                msg = "क्षमा करें, आपके इनपुट को प्रोसेस करने में अनुवाद विफल रहा। कृपया इसे रोमन लिपि (English characters) में टाइप करने का प्रयास करें।"
            elif lang_code == 'or':
                msg = "କ୍ଷମା କରନ୍ତୁ, ଆପଣଙ୍କ ଇନପୁଟ୍ ପ୍ରକ୍ରିୟାକରଣରେ ଅନୁବାଦ ବିଫଳ ହେଲା। ଦୟାକରି ଏହାକୁ ରୋମାନ୍ ଲିପିରେ ଟାଇପ୍ କରିବାକୁ ଚେଷ୍ଟା କରନ୍ତୁ।"
            else:
                msg = f"Sorry, input translation failed to process your script. Try Romanized English. (Language Code: {lang_code})"

            return {'type': 'error', 'error': msg, 'labels': labels}
        
        if message_key == 'PLANT_NOT_FOUND':
            msg = "Please specify the crop (e.g., Tomato, Pepper, Potato)."
        else:
            msg = "An unknown error occurred during processing."

        msg = translator.translate_output(msg, lang_code)

        return {
            'type': 'text_symptom',
            'disease_name': labels['disease_detected'] + ': N/A',
            'crop': labels['crop'] + ': N/A',
            'confidence': 'N/A',
            'cause': msg,
            'treatment': msg,
            'symptoms': [],
            'labels': labels
        }

    if text_proc == "ERROR_SCRIPT_INPUT_TRANSLATION_FAIL":
        return error('SCRIPT_ERROR_TRANSLATION_FAIL')

    plant = get_plant_from_text(text_proc)

    if plant == 'Unknown':
        return error('PLANT_NOT_FOUND')

    matched = extract_symptoms(text_proc, plant)

    translated_plant = translator.translate_output(plant, lang_code)

    if not matched:
        msg = f"Could not match symptoms for {plant}. Try rephrasing."
        msg = translator.translate_output(msg, lang_code)

        return {
            'type': 'text_symptom',
            'disease_name': labels['disease_detected'] + ': N/A',
            'crop': translated_plant, 
            'confidence': 'N/A',
            'cause': msg,
            'treatment': labels['consult_expert'],
            'symptoms': [],
            'labels': labels
        }

    best = matched[0]

    # Output Translation
    translated_cause = translator.translate_output(best['cause'], lang_code)
    translated_treatment = translator.translate_output(best['treatment'], lang_code)

    eng_sym = [m['symptom_key'] for m in matched]
    tr_sym = [translator.translate_output(s, lang_code) for s in eng_sym]

    translated_disease = translator.translate_output(best['symptom_key'], lang_code)

    return {
        'type': 'text_symptom',
        'disease_name': translated_disease,
        'crop': translated_plant,
        'confidence': best['score'],
        'cause': translated_cause,
        'treatment': translated_treatment,
        'symptoms': tr_sym,
        'labels': labels
    }


# ---------------------------------------------------------
#                   FLASK ROUTES
# ---------------------------------------------------------

@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    lang = request.values.get('lang', 'en') 

    if request.is_json:
        data = request.json
        if data and 'lang' in data:
            lang = data.get('lang', lang)
        
        if data and 'text' in data:
            return jsonify(analyze_text(data['text'], lang))

    if 'image' in request.files:
        image = request.files['image']
        if image.filename != '':
            return jsonify(predict_image(image, lang))

    return jsonify({'error': 'Invalid request format.'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
