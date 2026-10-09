from lughaat_tts.normalize import normalize, normalize_chars, expand_numbers, split_sentences, normalize_punctuation


def test_char_unification():
    assert normalize_chars("كتاب يه") == "کتاب یہ"
    assert normalize_chars("ه") == "ہ"
    assert normalize_chars("بھائی") == "بھائی"          # ھ stays
    assert normalize_chars("کتـــاب") == "کتاب"          # tatweel removed


def test_numbers_in_urdu_context():
    assert normalize("اس کی قیمت ۲۵۰۰ روپے ہے۔") == "اس کی قیمت دو ہزار پانچ سو روپے ہے۔"
    assert normalize("میری سالگرہ 14 اگست کو ہوتی ہے۔") == "میری سالگرہ چودہ اگست کو ہوتی ہے۔"
    assert normalize("ٹرین صبح ۷:۴۵ پر روانہ ہوئی۔") == "ٹرین صبح پونے آٹھ پر روانہ ہوئی۔"
    assert normalize("۱۹۴۷ء میں پاکستان آزاد ہوا۔") == "انیس سو سینتالیس میں پاکستان آزاد ہوا۔"
    assert normalize("درجۂ حرارت ۳۸ اعشاریہ ۵ ڈگری تھا۔") == "درجۂ حرارت اڑتیس اعشاریہ پانچ ڈگری تھا۔"
    assert normalize("۲۰۲۶ میں 25 لوگ") == "دو ہزار چھبیس میں پچیس لوگ"
    assert normalize("50% لوگ") == "پچاس فیصد لوگ"
    assert normalize("Rs. 500 کا بل") == "پانچ سو روپے کا بل"
    assert normalize("14/08/2026 کو") == "چودہ اگست دو ہزار چھبیس کو"


def test_numbers_in_english_context_are_left_for_misaki():
    assert expand_numbers("یہ meeting 3 pm پر ہے") == "یہ meeting 3 pm پر ہے"
    assert expand_numbers("call me at 5 pm") == "call me at 5 pm"


def test_sentence_split_and_punct():
    assert split_sentences("پہلا جملہ۔ دوسرا جملہ؟ تیسرا!") == ["پہلا جملہ۔", "دوسرا جملہ؟", "تیسرا!"]
    assert normalize_punctuation("ہاں، ٹھیک ہے۔ کیا؟") == "ہاں, ٹھیک ہے. کیا?"


def test_deterministic():
    s = "وزیرِ اعظم نے ۲۰۲۶ میں 25 لوگوں سے ملاقات کی۔"
    assert normalize(s) == normalize(s)
