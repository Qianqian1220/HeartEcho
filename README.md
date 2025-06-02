## 🧩 Modules

- **ASR – Whisper (Base, zh)**  
  Transcribes Mandarin player speech into text using the base version of Whisper with Chinese language setting.

- **LLM – Yi-1.5-6B-Chat**  
  Generates context-aware, personality-consistent responses based on two inputs: a predefined character description and the transcribed player utterance.

- **TTS – CosyVoice2**  
  Synthesizes expressive Mandarin speech from text, using a speaker ID and reference audio prompt to match target emotion and voice profile.

---

## 📚 Dataset

- **Fine-Tuning Set**  
  Mandarin subset of the *Emotional Speech Dataset (ESD)*, cleaned to ~4.86 hours per speaker. Used to fine-tune two male emotional profiles for TTS synthesis.

- **Emotion Categories**  
  Neutral, Happy, Angry, Sad, Surprised — each with parallel utterances for consistent emotion modeling.

- **Test Set**  
  Curated for balance across character identity and emotion, used for both subjective (MOS-I) and objective (speaker similarity) evaluations.

---

## 📊 Evaluation

### 🎧 Subjective Evaluation: MOS-I (Mean Opinion Score for Identity)

- **Participants**: 32 valid human raters  
- **Conditions**: 6 main settings (3 emotions × 2 characters) + 2 mismatch controls  
- **Findings**:  
  - Character-consistent samples scored 3.55 on average  
  - Mismatched samples scored below 2.8, indicating effective identity modeling

### 🎙️ Objective Evaluation: Speaker Similarity

- **Method**: Cosine similarity computed using `resemblyzer` embeddings  
- **Results**:  
  - Speaker A (real vs. synthesized): **0.81**  
  - Speaker B (real vs. synthesized): **0.84**  
  - Confirms high similarity between natural and generated voices within the same speaker identity
