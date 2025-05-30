from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path

# Initialize speaker encoder
encoder = VoiceEncoder()

# Define file pair: (generated, reference)
pairs = [
    ("generated_sample.wav", "reference_speaker/emotion/utterance.wav")
]

# Set root directories
generated_root = Path("/path/to/generated_audio")
reference_root = Path("/path/to/reference_dataset")

print("Speaker Similarity (Cosine Distance):\n")

for gen_filename, ref_rel_path in pairs:
    try:
        gen_path = generated_root / gen_filename
        ref_path = reference_root / ref_rel_path

        gen_wav = preprocess_wav(gen_path)
        ref_wav = preprocess_wav(ref_path)

        gen_embed = encoder.embed_utterance(gen_wav)
        ref_embed = encoder.embed_utterance(ref_wav)

        cosine_sim = np.dot(gen_embed, ref_embed) / (
            np.linalg.norm(gen_embed) * np.linalg.norm(ref_embed)
        )

        print(f"{gen_filename} vs {ref_path.name}: {cosine_sim:.4f}")
    except Exception as e:
        print(f"[Error] Failed to compare {gen_filename} and {ref_rel_path}: {e}")
