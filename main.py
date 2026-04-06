
!pip install deep-translator

# 2. Imports

import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from deep_translator import GoogleTranslator
from functools import lru_cache

# 3. Hinglish + English Dictionary

HINGLISH_WORDS = {
    "kya", "kaise", "hai", "ho", "kyu", "nahi", "haan",
    "bhai", "yaar", "tum", "mera", "tera", "kar", "raha",
    "rahe", "chal", "acha", "theek", "karo", "mat", "kyunki"
}

COMMON_ENGLISH = {
    "hello","hi","hey","ok","yes","no","thanks","bye",
    "what","why","how","when","where","who",
    "good","bad","fine","great","awesome",
    "morning","evening","night",
    "bro","dude","friend",
    "go","come","eat","sleep","run",
    "is","am","are","was","were","be",
    "do","does","did","have","has","had",
    "this","that","these","those",
    "i","you","we","they","he","she","it"
}

# 4. Dataset

data = [
    ("नमस्ते आप कैसे हैं?", "Hindi"),
    ("Hello, how are you?", "English"),
    ("namaste kaise ho", "Hinglish"),
    ("नमस्कार, तुम्ही कसे आहात?", "Marathi"),
    ("হ্যালো, আপনি কেমন আছেন?", "Bengali"),
    ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "Tamil"),
    ("హలో, మీరు ఎలా ఉన్నారు?", "Telugu"),
    ("ಹಲೋ, ನೀವು ಹೇಗಿದ್ದೀರಿ?", "Kannada"),
    ("ഹലോ, നിങ്ങൾക്ക് സുഖമാണോ?", "Malayalam"),
    ("હેલો, તમે કેમ છો?", "Gujarati"),
    ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?", "Punjabi"),
    ("السلام علیکم، آپ کیسے ہیں؟", "Urdu"),
]

# ✅ Add real English sentences
extra_english = [
    ("I am learning machine learning", "English"),
    ("This is a language detection system", "English"),
    ("We are building an AI model", "English"),
    ("He is going to school", "English"),
    ("She likes programming", "English"),
    ("They are playing football", "English"),
    ("Can you help me?", "English"),
    ("What is your name?", "English"),
]
data = data * 40 + extra_english * 30
df = pd.DataFrame(data, columns=["text", "language"])


# 5. Encode Labels

le = LabelEncoder()
y = le.fit_transform(df["language"])



# 6. Vectorization

vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    max_features=20000
)

X = vectorizer.fit_transform(df["text"])

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Evaluation

print("\n📊 Evaluation:\n")
print(classification_report(
    y_test,
    model.predict(X_test),
    target_names=le.classes_
))

# 10. Script Detection

def detect_script(text):
    if re.search(r'[\u0900-\u097F]', text):
        return "Devanagari"
    elif re.search(r'[a-zA-Z]', text):
        return "Latin"
    return "Unknown"

# 11. Hinglish Detection (Improved)

def is_hinglish(text):
    words = re.findall(r'\w+', text.lower())
    if not words:
        return False

    count = sum(1 for w in words if w in HINGLISH_WORDS)
    return count >= 1 and len(words) >= 2

# 12. Predict Language (FINAL)
def predict_language(text, threshold=0.6):
    text_clean = text.lower().strip()
    words = re.findall(r'\w+', text_clean)

    # 🚨 Garbage detection
    if not re.search(r'[a-zA-Z\u0900-\u097F]', text_clean):
        return "Not Detected"

    if len(set(text_clean)) <= 2:
        return "Not Detected"

    # ✅ Common English words
    if text_clean in COMMON_ENGLISH:
        return "English"

    script = detect_script(text_clean)

    # ✅ English fallback rule
    if script == "Latin" and re.fullmatch(r'[a-zA-Z]+', text_clean):
        if len(text_clean) >= 3:
            return "English (rule)"

    vec = vectorizer.transform([text_clean])
    probs = model.predict_proba(vec)[0]

    max_conf = np.max(probs)
    pred_lang = le.inverse_transform([np.argmax(probs)])[0]

    # ✅ Dynamic threshold
    if len(words) == 1:
        threshold = 0.4
    elif len(text_clean) < 5:
        threshold = 0.5

    # 🚨 Low confidence handling
    if max_conf < threshold:
        if script == "Latin":
            return "English (low confidence)"
        return "Not Detected"

    # ✅ Script rule
    if script == "Devanagari":
        return "Hindi"

    # ✅ Hinglish rule
    if script == "Latin":
        if is_hinglish(text_clean):
            return "Hinglish"

    return f"{pred_lang} ({max_conf:.2f})"


# 13. Word-wise Detectio
def detect_mixed(text):
    words = re.findall(r'\w+', text.lower())

    if not words:
        return "No valid words"

    vecs = vectorizer.transform(words)
    probs = model.predict_proba(vecs)

    results = []

    for word, p in zip(words, probs):
        conf = np.max(p)
        lang = le.inverse_transform([np.argmax(p)])[0]

        if word in COMMON_ENGLISH:
            lang = "English"
        elif word in HINGLISH_WORDS:
            lang = "Hinglish"
        elif re.fullmatch(r'[a-zA-Z]+', word):
            lang = "English"
        elif conf < 0.6:
            lang = "Not Detected"

        results.append(f"{word} → {lang} ({conf:.2f})")

    return "\n".join(results)
# =========================================
# 14. Translation
# =========================================
@lru_cache(maxsize=500)
def translate(text, target="en"):
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except:
        return "Translation Error"


# =========================================
# 15. Language Menu
# =========================================
lang_map = {
    "1": ("en", "English"),
    "2": ("hi", "Hindi"),
    "3": ("mr", "Marathi"),
    "4": ("bn", "Bengali"),
    "5": ("ta", "Tamil"),
    "6": ("te", "Telugu"),
    "7": ("kn", "Kannada"),
    "8": ("ml", "Malayalam"),
    "9": ("gu", "Gujarati"),
    "10": ("pa", "Punjabi"),
    "11": ("ur", "Urdu"),
}


# =========================================
# 16. Sample Test
# =========================================
samples = [
    "hello",
    "developer",
    "kya kar rahe ho",
    "hello bhai",
    "नमस्ते",
    "asdjkl"
]

print("\n🔍 Sample Output:\n")
for s in samples:
    print("Input:", s)
    print("Detected:", predict_language(s))
    print("Word-wise:\n", detect_mixed(s))
    print("Translated:", translate(s))
    print("-" * 40)


# =========================================
# 17. User Input Loop
# =========================================
print("\n💬 Type text (or 'exit')")

while True:
    text = input("\nEnter text: ").strip()

    if text.lower() == "exit":
        print("👋 Exiting...")
        break

    if not text:
        print("⚠️ Enter some text")
        continue

    print("\n🌍 Detected:", predict_language(text))
    print("\n🔎 Word-wise:\n", detect_mixed(text))

    print("\n🌐 Convert to:")
    for k, (_, name) in lang_map.items():
        print(f"{k}. {name}")

    choice = input("Choose (1-11): ")
    target = lang_map.get(choice, ("en", "English"))

    print(f"\n✅ {target[1]}:", translate(text, target[0]))
