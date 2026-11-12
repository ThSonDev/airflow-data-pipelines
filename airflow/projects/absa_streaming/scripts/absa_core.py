"""Shared code for the ABSA pipeline: constants, text cleaning, encoding, model, settings.

Used by preprocess, train, eval, save_postgres and the Spark consumer, so training and
serving always run the same code.
"""
import os
import re
import unicodedata
from types import SimpleNamespace

import torch
import torch.nn as nn

# Architecture constants
ASPECTS = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']
EMBED_DIM = 128
NUM_FILTERS = 192
WORD_WINDOW = 5
NUM_CLASSES = 4

# Label mapping (training labels are CSV labels + 1)
PREDICTION_MAP = {0: "NONE", 1: "NEG", 2: "POS", 3: "NEU"}

# Text cleaning options
REMOVE_TONE = False
REMOVE_NOISE = True


def get_settings():
    """Read settings from environment variables. Defaults are the Docker container values."""
    env = os.environ.get
    models_dir = env("MODELS_DIR", "/opt/airflow/models")
    return SimpleNamespace(
        postgres_host=env("POSTGRES_HOST", "postgres"),
        postgres_port=int(env("POSTGRES_PORT", "5432")),
        postgres_db=env("POSTGRES_DB", "airflow"),
        postgres_user=env("POSTGRES_USER", "airflow"),
        postgres_password=env("POSTGRES_PASSWORD", "airflow"),
        kafka_bootstrap=env("KAFKA_BOOTSTRAP", "kafka:9092"),
        kafka_topic=env("KAFKA_TOPIC", "absa-reviews"),
        models_dir=models_dir,
        data_dir=env("DATA_DIR", "/opt/airflow/projects/absa_streaming/data"),
        tmp_dir=env("TMP_DIR", f"{models_dir}/tmp"),
    )


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
    text = str(text).lower().strip()
    if REMOVE_TONE:
        text = remove_vietnamese_tone(text)
    if REMOVE_NOISE:
        text = remove_non_vietnamese_chars(text)
    text = re.sub(r"\s+", " ", text)
    return text


def encode(text, vocab):
    """Cleaned text -> list of token ids. Unknown words map to 1 (<UNK>)."""
    return [vocab.get(t, 1) for t in text.split()]


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, NUM_FILTERS, kernel_size=WORD_WINDOW, padding=int(WORD_WINDOW // 2))
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.shared_dense = nn.Linear(NUM_FILTERS, 128)
        self.dropout = nn.Dropout(0.5)
        self.output_heads = nn.ModuleList([
            nn.Linear(128, NUM_CLASSES) for _ in range(num_labels)
        ])

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        x = torch.relu(self.shared_dense(x))
        x = self.dropout(x)
        outputs = []
        for head in self.output_heads:
            outputs.append(head(x))
        return torch.stack(outputs, dim=1)
