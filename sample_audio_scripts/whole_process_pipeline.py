import sys
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Add CosyVoice2 to Python import path
sys.path.append('path/to/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# Set paths and input text
output_wav = "output/xxx.wav"
text_input = "Your input text here."
speaker_id = "speaker_id"
prompt_path = "path/to/prompt.wav"

# Define character prompt for the language model
system_prompt = "Character description for the language model."

# Load language model and tokenizer
print("Loading language model...")
model_id = "01-ai/Yi-1.5-6B-Chat"
cache_dir = "path/to/cache"

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

# Generate a character-specific response from the language model
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"You are a character with the above traits. Reply naturally to this message:\n\nUser: {text_input}"}
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

# Load pretrained CosyVoice2 model
cosyvoice = CosyVoice2(
    'path/to/pretrained_model',
    load_jit=False,
    load_trt=False,
    fp16=False,
    use_flow_cache=False
)

# Load reference audio for style prompting
prompt_audio = load_wav(prompt_path, 16000)

# Run speech synthesis
print("Synthesizing speech...")
Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

for i, result in enumerate(cosyvoice.inference_instruct2(reply_clean, speaker_id, prompt_audio, stream=False)):
    torchaudio.save(output_wav, result['tts_speech'], cosyvoice.sample_rate)
    print(f"Synthesis complete: {output_wav}")
