"""Shared code for the ABSA pipeline: constants, settings, logging, padding, model.

Used by preprocess, train, eval, save_postgres and the Spark consumer, so training and
serving always run the same code.
"""
import logging
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from text_utils import clean, encode, remove_vietnamese_tone  # noqa: F401  (re-exported)

# Architecture constants
ASPECTS = ['Price', 'Shipping', 'Outlook', 'Quality', 'Size', 'Shop_Service', 'General', 'Others']
EMBED_DIM = 128
NUM_FILTERS = 192
WORD_WINDOW = 5
NUM_CLASSES = 4

# Label mapping (training labels are CSV labels + 1)
PREDICTION_MAP = {0: "NONE", 1: "NEG", 2: "POS", 3: "NEU"}


def get_logger(name):
    """One log format for all scripts. Level comes from LOG_LEVEL (default INFO)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return logger


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


def pad_batch(id_lists):
    """Pad token id lists to the longest one, at least WORD_WINDOW wide (the conv kernel size)."""
    width = max(WORD_WINDOW, max((len(ids) for ids in id_lists), default=0))
    out = torch.zeros(len(id_lists), width, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


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
