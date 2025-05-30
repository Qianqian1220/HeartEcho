### Modules

- **ASR (Whisper Base)**: Converts Mandarin player speech to text.
- **LLM (Yi-1.5-6B-Chat)**: Generates a context-aware, personality-driven response based on character prompt + recognized text.
- **TTS (CosyVoice2)**: Synthesizes expressive Mandarin speech using speaker ID + emotion prompts.

---

## 📁 Dataset

- **Training Set**: Mandarin ESD (Emotional Speech Dataset)
  - ~4.86 hours per speaker after cleaning
  - Used for fine-tuning two emotional male voice profiles
- **Emotion Labels**: Neutral, Happy, Angry, Sad, Surprised
- **Test Set**: Balanced across emotion and character identity for MOS-I and similarity evaluations

---

## 🧪 Evaluation

### Subjective Test: MOS-I (Mean Opinion Score for Identity)

- Evaluated with 32 valid human raters
- Conditions: 3 emotional styles × 2 characters + 2 mismatch controls
- Results show strong character consistency (avg. MOS-I > 3.5) and clear mismatch detection (avg. MOS-I < 2.8)

### Objective Test: Speaker Similarity

- Measured using `resemblyzer`
- Cross-checked between real and synthetic audio per speaker
- Achieved similarity scores of 0.5967 and 0.5799 for cross-speaker TTS

---

## 🛠 Installation

**Dependencies**:
```bash
pip install -r requirements.txt
