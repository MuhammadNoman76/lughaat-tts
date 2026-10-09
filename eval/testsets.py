"""Test sentences from the plan appendices (A: Urdu, A2: code-switched, A3: English, B: gold words)."""
from __future__ import annotations

APPENDIX_A = [
    "پاکستان ایک خوبصورت ملک ہے۔",
    "میں نے کل بازار سے تین کتابیں خریدیں۔",
    "بھائی، گھر میں دھوپ بہت تیز ہے۔",
    "وزیرِ اعظم نے آج قوم سے خطاب کیا۔",
    "لڑکی نے اپنی چھوٹی بہن کو کہانی پڑھ کر سنائی۔",
    "ڈاکٹر صاحب نے کہا کہ تم جلد ٹھیک ہو جاؤ گے۔",
    "ہاں، مجھے یاد ہے، ہم وہاں نہیں گئے تھے۔",
    "اس کی قیمت ۲۵۰۰ روپے ہے۔",
    "میری سالگرہ 14 اگست کو ہوتی ہے۔",
    "ٹرین صبح ۷:۴۵ پر روانہ ہوئی۔",
    "غالب کا شعر آج بھی ہر زبان پر ہے۔",
    "محبت، خلوص اور اعتماد زندگی کے ستون ہیں۔",
    "کیا آپ میری بات سمجھ رہے ہیں؟",
    "شکریہ! آپ کا دن اچھا گزرے۔",
    "قلم، خط اور غلطی جیسے الفاظ میں ق، خ اور غ کی آواز واضح ہونی چاہیے۔",
    "میں نے اپنا laptop آن کیا اور email چیک کی۔",
    "ڈاکٹر عبدالقدیر خان پاکستان کے مشہور سائنسدان تھے۔",
    "یہ سڑک بہت چوڑی ہے۔",
    "انہوں نے کہا تھا کہ وہ جمعرات کو آئیں گے۔",
    "اردو ایک شیریں زبان ہے۔",
    "۱۹۴۷ء میں پاکستان آزاد ہوا۔",
    "درجۂ حرارت ۳۸ اعشاریہ ۵ ڈگری تھا۔",
]

APPENDIX_A2 = [
    "آج کی meeting کینسل ہو گئی ہے۔",
    "میں نے اپنا laptop آن کیا اور email چیک کی۔",
    "Please اپنا password کسی کو share نہ کریں۔",
    "ہمارا next step یہ ہے کہ client کو proposal بھیج دیں۔",
    "یہ app بہت user friendly ہے لیکن battery جلدی ختم کرتی ہے۔",
    "Thank you so much، آپ نے بہت help کی۔",
    "Office میں آج internet کا problem تھا۔",
    "اس project کی deadline اگلے Monday تک ہے۔",
    "میں weekend پر Karachi جا رہا ہوں۔",
    "Doctor نے کہا کہ یہ normal infection ہے، worry نہ کریں۔",
    "ہمیں AI اور machine learning پر focus کرنا چاہیے۔",
    "Update کے بعد phone ٹھیک سے کام کر رہا ہے۔",
    "یہ meeting 3 pm پر Zoom پر ہو گی۔",
    "Customer service نے میری complaint register کر لی۔",
    "آج weather بہت pleasant ہے، let's go for a walk۔",
]

APPENDIX_A3 = [
    "Good morning, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Please send me the report by Thursday afternoon.",
    "Artificial intelligence is changing the way we work.",
    "I think this is the best decision for our team.",
    "The meeting has been moved to three thirty.",
    "Thank you for your patience and understanding.",
    "Karachi is the largest city in Pakistan.",
]

# Appendix B gold pronunciations (canonical set). Note: the plan lists ڈاکٹر as ɖɑːktər; the
# letter ٹ is retroflex, so ɖɑːkʈər is used (consonants are authoritative; see DECISIONS.md).
APPENDIX_B = {
    "پاکستان": "pɑːkɪstɑːn", "خوبصورت": "xuːbsuːrət", "پڑھنا": "pəɽʰnɑː", "بھائی": "bʰɑːiː", "گھر": "ɡʰər",
    "نہیں": "nəhĩː", "ہاں": "hɑ̃ː", "قلم": "qələm", "غلط": "ɣələt", "خط": "xət", "کتاب": "kɪtɑːb", "لڑکی": "ləɽkiː",
    "چھوٹا": "ʧʰoːʈɑː", "ٹھیک": "ʈʰiːk", "ڈاکٹر": "ɖɑːkʈər", "شکریہ": "ʃʊkrijɑː", "آج": "ɑːʤ", "زندگی": "zɪndəɡiː",
    "دوست": "doːst", "ایک": "eːk", "محبت": "mʊhəbbət", "وزیرِ اعظم": "ʋəziːreː ɑːzəm",
}

# expected language tags for Appendix A2 token-by-token (words only; punctuation omitted)
APPENDIX_A2_TAGS = {
    "آج کی meeting کینسل ہو گئی ہے۔": ["ur", "ur", "en", "ur", "ur", "ur", "ur"],
    "Please اپنا password کسی کو share نہ کریں۔": ["en", "ur", "en", "ur", "ur", "en", "ur", "ur"],
    "ہمیں AI اور machine learning پر focus کرنا چاہیے۔": ["ur", "en_acronym", "ur", "en", "en", "ur", "en", "ur", "ur"],
    "یہ meeting 3 pm پر Zoom پر ہو گی۔": ["ur", "en", "num_en", "en", "ur", "en", "ur", "ur", "ur"],
}

PAKISTANI_ANCHORS = {
    "laptop": "lɛːpʈɑːp", "meeting": "miːʈɪŋ", "email": "iːmeːl", "thank you": "tʰɛːŋk juː",
    "problem": "prɑːbləm", "cancel": "kɛːnsəl", "update": "əpɖeːʈ",
}

__all__ = ["APPENDIX_A", "APPENDIX_A2", "APPENDIX_A3", "APPENDIX_B", "APPENDIX_A2_TAGS", "PAKISTANI_ANCHORS"]
