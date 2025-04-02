import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from torch.serialization import add_safe_globals

# Add XttsConfig to safe globals
add_safe_globals([XttsConfig])

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
text = "Technology has transformed the way we live, work, and communicate. From smartphones to smart homes, innovation continues to reshape our daily routines. As artificial intelligence advances, we're beginning to see machines understand language, recognize images, and even generate creative content. The future is unfolding faster than ever before."

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
wav = tts.tts(text=text, speaker_wav="xtts.wav", language="en")
# Text to speech to a file
tts.tts_to_file(text=text, speaker_wav="xtts.wav", language="en", file_path="output.wav")