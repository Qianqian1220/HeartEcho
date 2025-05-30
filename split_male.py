import os
import random
import shutil

random.seed(42)

src_root = "/scratch/s6029388/CosyVoice/ESD"
dst_root = "/scratch/s6029388/CosyVoice/ESD_split"
train_root = os.path.join(dst_root, "train")
test_root = os.path.join(dst_root, "test")

os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

# Specify target speakers
speakers = ["0004", "0005", "0006", "0008", "0010"]

# Collect all valid samples
all_samples = []

for speaker in speakers:
    speaker_path = os.path.join(src_root, speaker)

    for emotion in os.listdir(speaker_path):
        emotion_path = os.path.join(speaker_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        for fname in os.listdir(emotion_path):
            if fname.endswith(".wav"):
                base = os.path.splitext(fname)[0]
                wav_path = os.path.join(emotion_path, base + ".wav")
                ori_txt = os.path.join(emotion_path, base + ".original.txt")
                norm_txt = os.path.join(emotion_path, base + ".normalized.txt")

                if os.path.exists(ori_txt) and os.path.exists(norm_txt):
                    all_samples.append((base, speaker, emotion, wav_path, ori_txt, norm_txt))

# Shuffle and split into 70% train / 30% test
random.shuffle(all_samples)
split_idx = int(len(all_samples) * 0.7)
train_samples = all_samples[:split_idx]
test_samples = all_samples[split_idx:]

def process_and_copy(samples, target_root):
    for base, speaker, emotion, wav_path, ori_txt, norm_txt in samples:
        out_dir = os.path.join(target_root, speaker, emotion)
        os.makedirs(out_dir, exist_ok=True)

        # Copy .wav file
        shutil.copyfile(wav_path, os.path.join(out_dir, os.path.basename(wav_path)))

        # Process .original.txt and .normalized.txt
        for txt_path in [ori_txt, norm_txt]:
            if not os.path.exists(txt_path):
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()

            fname = os.path.basename(txt_path)
            utt_id = os.path.splitext(fname)[0]  # e.g., 0004_000365
            spk_id = utt_id[:4]

            if '<|endofprompt|>' in line:
                _, text = line.split('<|endofprompt|>', 1)
                new_line = f"{spk_id}<|endofprompt|>{text.strip()}"
            else:
                print(f"[Warning] Unexpected format: {txt_path}")
                continue

            # Write to target file
            new_txt_path = os.path.join(out_dir, os.path.basename(txt_path))
            with open(new_txt_path, 'w', encoding='utf-8') as f:
                f.write(new_line + '\n')

        print(f"Processed: {base} → {out_dir}")

# Run processing
process_and_copy(train_samples, train_root)
process_and_copy(test_samples, test_root)

print(f"Done. {len(train_samples)} training samples, {len(test_samples)} testing samples.")
