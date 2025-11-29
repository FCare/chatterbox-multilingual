#!/usr/bin/env python3
"""
Test script using local sources instead of installed chatterbox module.
This bypasses the need to install the package and uses sources directly.
"""

import sys
import os
import torchaudio as ta
import torch

# Add src directory to Python path to import from local sources
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import from local sources
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
print("Loading models from local sources...")

# Test 1: English TTS
print("\n=== Test 1: English TTS ===")
model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
print(f"Generating: {text}")
wav = model.generate(text)
ta.save("test-local-1.wav", wav, model.sr)
print("✅ Saved to: test-local-1.wav")

# Test 2: Multilingual TTS (French)
print("\n=== Test 2: Multilingual TTS (French) ===")
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
print(f"Generating: {text}")
wav = multilingual_model.generate(text, language_id="fr")
ta.save("test-local-2.wav", wav, multilingual_model.sr)
print("✅ Saved to: test-local-2.wav")

# Test 3: English with voice prompt (if available)
print("\n=== Test 3: English with voice prompt ===")
# Only test if we have a valid audio prompt file
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
if os.path.exists(AUDIO_PROMPT_PATH):
    print(f"Using audio prompt: {AUDIO_PROMPT_PATH}")
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
    ta.save("test-local-3.wav", wav, model.sr)
    print("✅ Saved to: test-local-3.wav")
else:
    print(f"⚠️  Audio prompt file not found: {AUDIO_PROMPT_PATH}")
    print("Skipping voice prompt test...")

print("\n🎉 All tests completed successfully!")
print(f"Generated audio files are saved in: {os.getcwd()}")