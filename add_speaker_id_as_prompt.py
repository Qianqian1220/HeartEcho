import os
from glob import glob
#Step1:Generate two kinds of txt format files for data process
#Step2:Write prompt, such as speaker_id
# Root directory containing data splits
root_dir = "/path/to/dataset"
splits = ["train", "test"]

for split in splits:
    split_dir = os.path.join(root_dir, split)

    for speaker in os.listdir(split_dir):
        speaker_path = os.path.join(split_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue

        for emotion in os.listdir(speaker_path):
            emotion_path = os.path.join(speaker_path, emotion)
            if not os.path.isdir(emotion_path):
                continue

            # Locate all .original.txt and .normalized.txt files
            txt_files = glob(os.path.join(emotion_path, "*.original.txt")) + \
                        glob(os.path.join(emotion_path, "*.normalized.txt"))

            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()

                # Replace prefix with speaker ID
                fname = os.path.basename(txt_file)
                utt_id = fname.split('.')[0]  # e.g., spk001_000123
                spk_id = utt_id[:4]

                if '<|endofprompt|>' in line:
                    _, text = line.split('<|endofprompt|>', 1)
                    new_line = f"{spk_id}<|endofprompt|>{text.strip()}"
                else:
                    print(f"[Warning] Unexpected format: {txt_file}")
                    continue

                # Overwrite the file
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(new_line + '\n')

                print(f"Processed: {txt_file}")
