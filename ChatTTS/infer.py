import ChatTTS
import torch
import torchaudio
import time

chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

texts = ["Technology has transformed the way we live, work, and communicate. From smartphones to smart homes, innovation continues to reshape our daily routines. As artificial intelligence advances, we're beginning to see machines understand language, recognize images, and even generate creative content. The future is unfolding faster than ever before."]

start_time = time.time()
wavs = chat.infer(texts)

for i in range(len(wavs)):
    """
    In some versions of torchaudio, the first line works but in other versions, so does the second line.
    """
    try:
        torchaudio.save(f"chattts{i}.wav", torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
    except:
        torchaudio.save(f"chattts{i}.wav", torch.from_numpy(wavs[i]), 24000)

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")