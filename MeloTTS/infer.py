from melo.api import TTS
import time

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available
text = "Technology has transformed the way we live, work, and communicate. From smartphones to smart homes, innovation continues to reshape our daily routines. As artificial intelligence advances, we're beginning to see machines understand language, recognize images, and even generate creative content. The future is unfolding faster than ever before."
start_time = time.time()
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id
# Default accent
output_path = 'melotts.wav'
model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed)
end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")
