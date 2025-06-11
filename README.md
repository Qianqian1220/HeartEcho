# Otome Game Talking NPCs

This project gives voice to your game boyfriend.

No more canned voice lines — each response is generated and spoken in real time, adapting to what the player says. Think of it as a flirty NPC that actually listens.

## What It Does

You talk → He listens → He thinks → He talks back  
All in Mandarin, all in character.

Behind the scenes:

- **Whisper (ASR)** — turns your voice into text  
- **Yi-1.5-6B-Chat (LLM)** — writes the perfect in-character reply  
- **CosyVoice2 (TTS)** — speaks it with the right emotion and voice style

## Why It's Fun

- Every line feels personal — no two responses are exactly the same  
- Supports multiple character types with distinct speaking styles  
- Emotions like *happy*, *calm*, and *surprised* are prompt-controlled  
- Designed for Mandarin otome game scenarios  
- Runs in real time (on GPU) — perfect for live interaction

## Data Used

Fine-tuned on a Mandarin emotional speech dataset with multiple male speakers and five basic emotions. Each voice is aligned to a fictional persona with prompt-based emotion control.

## Try It

Set your mic, pick your NPC, say something.  
He'll talk back. Sweet or sarcastic — depends on who you chose.

---

Built for players who want more than "Press X to romance".
