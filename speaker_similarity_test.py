from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path

# 初始化 encoder
encoder = VoiceEncoder()

# 文件路径配对
pairs = [
    ("1.wav", "/scratch/s6029388/CosyVoice/ESD_split/test/0006/Surprise/0006_001425.wav"),
    ("2.wav", "/scratch/s6029388/CosyVoice/ESD_split/test/0006/Happy/0006_000723.wav"),
    ("3.wav", "/scratch/s6029388/CosyVoice/ESD_split/test/0006/Neutral/0006_000009.wav"),
    ("4.wav", "/scratch/s6029388/CosyVoice/ESD_split/test/0008/Surprise/0008_001401.wav"),
    ("5.wav", "/scratch/s6029388/CosyVoice/ESD_split/test/0008/Happy/0008_000705.wav"),
    ("6.wav", "/scratch/s6029388/CosyVoice/ESD_split/test/0008/Neutral/0008_000045.wav"),
]

# 所有输出
print("Speaker Similarity Results (Cosine):\n")

for gen_file, ref_file in pairs:
    try:
        gen_wav = preprocess_wav(Path(f"/scratch/s6029388/CosyVoice/{gen_file}"))
        ref_wav = preprocess_wav(Path(ref_file))
        
        gen_embed = encoder.embed_utterance(gen_wav)
        ref_embed = encoder.embed_utterance(ref_wav)

        cosine_sim = np.dot(gen_embed, ref_embed) / (np.linalg.norm(gen_embed) * np.linalg.norm(ref_embed))
        print(f"{gen_file} vs {Path(ref_file).name}: {cosine_sim:.4f}")
    except Exception as e:
        print(f"Error comparing {gen_file} and {ref_file}: {e}")