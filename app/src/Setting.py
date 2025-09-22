import logging
from typing_extensions import Final
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Directory per le trascrizioni
TRANSCRIPTIONS_DIR: Final[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)),"transcriptions")

if not os.path.exists(TRANSCRIPTIONS_DIR):
    os.makedirs(TRANSCRIPTIONS_DIR)


ALLOWED_EXTENSIONS: set = {'mp3', 'wav', 'm4a', 'ogg', 'flac'}

# Lista lingue supportate
SUPPORTED_LANGUAGES: Final[dict] = {
    "auto": "Rileva automaticamente",
    "it": "Italiano",
    "en": "Inglese",
    "es": "Spagnolo",
    "fr": "Francese",
    "de": "Tedesco",
    "pt": "Portoghese",
    "ru": "Russo",
    "ja": "Giapponese",
    "zh": "Cinese",
    "ar": "Arabo"
}

# Lista modelli supportati
SUPPORTED_MODELS: Final[dict] = {
    #"turbo": "Turbo (veloce, meno accurato)",
    "tiny": "Tiny (veloce, meno accurato)",
    "base": "Base (equilibrato)",
    "small": "Small (buona accuratezza)",
    "medium": "Medium (molto accurato)",
    "large-v3": "Large-v3 (massima accuratezza)"
}