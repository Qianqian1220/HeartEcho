import sys
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 CosyVoice 模块
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# ========== 参数配置（来自问卷表格） ==========
output_wav = "roleA_happy_01.wav"
text_input = "你猜我今天见到谁了？我居然在街上遇到了我高中同学！"
speaker_id = "0006"
prompt_path = "/scratch/s6029388/CosyVoice/ESD_split/test/0006/Happy/0006_000723.wav"
system_prompt = (
    "冷峻守礼系：克制沉稳、低音磁性、礼貌周全。外表冷酷、不善言辞，却在关键时刻展现出细腻与体贴。"
)

# ========== 加载 Yi 模型 ==========
print("🤖 Loading Yi-1.5-6B-Chat model...")
model_id = "01-ai/Yi-1.5-6B-Chat"
cache_dir = "/scratch/s6029388/huggingface"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    cache_dir=cache_dir
)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    cache_dir=cache_dir
)

# ========== 构造清洁 Prompt ==========
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"你是一位冷峻守礼的男朋友，简短自然地回复女生这句话。\n\n女生说：{text_input}"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)
outputs = llm_model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

reply_raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
reply = reply_raw.split("assistant")[-1].strip()
reply_clean = reply.split("。")[0] + "。"

print(f"\n🧡 模型生成回复：{reply_clean}")

# ========== 加载 CosyVoice 模型 ==========
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    fp16=False,
    use_flow_cache=False
)

prompt_audio = load_wav(prompt_path, 16000)

# ========== 合成语音 ==========
print("🎤 合成语音中...")
for i, j in enumerate(cosyvoice.inference_instruct2(reply_clean, speaker_id, prompt_audio, stream=False)):
    torchaudio.save(output_wav, j['tts_speech'], cosyvoice.sample_rate)
    print(f"✅ 合成完成：{output_wav}")