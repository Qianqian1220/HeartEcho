import sys
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Add CosyVoice2 to Python path
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# ========== Configuration ==========
output_wav = "output/roleA_surprise_01.wav"
text_input = "I just opened the door and saw that you ordered food for me! What a surprise!"
speaker_id = "0006"
prompt_path = "data/test/0006/Surprise/example_prompt.wav"

# System prompt describing the character style
system_prompt = (
    "Reserved Gentleman: Calm, low-pitched voice with polite and restrained tone. "
    "Appears cold and quiet, but shows warmth and attentiveness in key moments."
)

# ========== Load LLM ==========
model_id = "01-ai/Yi-1.5-6B-Chat"
cache_dir = "cache/huggingface"

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

# ========== Construct prompt and generate reply ==========
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"You are a reserved boyfriend. Reply naturally and briefly to the following:\n\nShe says: {text_input}"}
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
reply_clean = reply.split("。")[0] + "。" if "。" in reply else reply

print(f"Generated reply: {reply_clean}")

# ========== Load CosyVoice2 model ==========
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=False,
    load_trt=False,
    fp16=False,
    use_flow_cache=False
)

prompt_audio = load_wav(prompt_path, 16000)

# ========== Synthesize speech ==========
print("Synthesizing speech...")

Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

for i, result in enumerate(cosyvoice.inference_instruct2(reply_clean, speaker_id, prompt_audio, stream=False)):
    torchaudio.save(output_wav, result['tts_speech'], cosyvoice.sample_rate)
    print(f"Synthesis complete: {output_wav}")
