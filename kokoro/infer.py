from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import time

pipeline = KPipeline(lang_code='a')
text = '''
Technology has transformed the way we live, work, and communicate. From smartphones to smart homes, innovation continues to reshape our daily routines. As artificial intelligence advances, we're beginning to see machines understand language, recognize images, and even generate creative content. The future is unfolding faster than ever before.'''

start_time = time.time()
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'kokoro.wav', audio, 24000)
end_time = time.time()
print(f"Total inference time: {end_time - start_time:.2f} seconds")