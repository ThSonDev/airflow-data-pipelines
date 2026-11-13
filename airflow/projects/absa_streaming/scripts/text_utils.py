"""Text cleaning and encoding. Must stay identical for training and serving."""
import re
import unicodedata

REMOVE_TONE = False
REMOVE_NOISE = True


def remove_vietnamese_tone(text):
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = unicodedata.normalize('NFC', text)
    return text


def remove_non_vietnamese_chars(text):
    return re.sub(
        r"[^a-zàáạảãăắằặẳẵâầấậẩẫèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ"
        r"ỳýỵỷỹđ0-9\s]",
        " ",
        text
    )


def clean(text):
    # NFC first: NFD input (base letter + combining mark) would lose its accents below
    text = unicodedata.normalize('NFC', str(text)).lower().strip()
    if REMOVE_TONE:
        text = remove_vietnamese_tone(text)
    if REMOVE_NOISE:
        text = remove_non_vietnamese_chars(text)
    text = re.sub(r"\s+", " ", text)
    return text


def encode(text, vocab):
    """Cleaned text -> token ids. Unknown words map to 1 (<UNK>); empty text gives [1]."""
    return [vocab.get(t, 1) for t in text.split()] or [1]
