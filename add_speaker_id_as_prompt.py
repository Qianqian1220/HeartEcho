import os
from glob import glob

# 根目录
root_dir = "/scratch/s6029388/CosyVoice/ESD_split"
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

            # 处理 .original.txt 和 .normalized.txt 文件
            txt_files = glob(os.path.join(emotion_path, "*.original.txt")) + \
                        glob(os.path.join(emotion_path, "*.normalized.txt"))

            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()

                # 替换前缀 ID 为 speaker ID（即文件名前四位）
                fname = os.path.basename(txt_file)
                utt_id = fname.split('.')[0]  # e.g. 0004_000365
                spk_id = utt_id[:4]

                if '<|endofprompt|>' in line:
                    _, text = line.split('<|endofprompt|>', 1)
                    new_line = f"{spk_id}<|endofprompt|>{text}"
                else:
                    print(f"⚠️ 格式错误: {txt_file}")
                    continue

                # 覆盖写入
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(new_line + '\n')

                print(f"✅ 处理完成: {txt_file}")
