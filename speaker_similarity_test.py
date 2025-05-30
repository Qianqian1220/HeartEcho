from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
from pathlib import Path

# Initialize the voice encoder
encoder = VoiceEncoder()

# Define paired audio files: (generated, reference)
pairs = [
    ("gen_1.wav", "ref_speakerA/emotion1/utt_001.wav"),
    ("gen_2.wav", "ref_speakerA/emotion2/utt_002.wav"),
    ("gen_3.wav", "ref_speakerA/emotion3/utt_003.wav"),
    ("gen_4.wav", "ref_speakerB/emotion1/utt_004.wav"),
    ("gen_5.wav", "ref_speakerB/emotion2/utt_005.wav"),
    ("gen_6.wav", "ref_speakerB/emotion3/utt_006.wav"),
]

# Define root paths
generated_root = Path("/path/to/generated_audio")
reference_root = Path("/path/to/reference_audio")

print("Speaker Similarity (Cosine Distance):\n")

for gen_file, ref_rel_path in pairs:
    try:
        gen_path = generated_root / gen_file
        ref_path = reference_root / ref_rel_path

        gen_wav = preprocess_wav(gen_path)
        ref_wav = preprocess_wav(ref_path)

        gen_embed = encoder.embed_utterance(gen_wav)
        ref_embed = encoder.embed_utterance(ref_wav)

        similarity = np.dot(gen_embed, ref_embed) / (
            np.linalg.norm(gen_embed) * np.linalg.norm(ref_embed)
        )

        print(f"{gen_file} vs {ref_path.name}: {similarity:.4f}")
    except Exception as err:
        print(f"[Error] Failed to compare {gen_file} and {ref_rel_path}: {err}")
