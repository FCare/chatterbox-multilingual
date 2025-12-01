#!/usr/bin/env python3
"""
Test script using local sources with streaming support.
This version includes both regular generate() and generate_stream() tests.
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
from chatterbox.vc import ChatterboxVC
from chatterbox.models.s3gen import S3GEN_SR  # Import sample rate constant

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
print("Loading models from local sources...")

def t3_to(model, dtype):
    model.t3.to(dtype=dtype)
    model.conds.t3.to(dtype=dtype)
    torch.cuda.empty_cache()
    return model

# Initialize models
try:
    model = t3_to(ChatterboxTTS.from_pretrained(device=device), torch.bfloat16)
    print("✅ ChatterboxTTS loaded successfully")
except Exception as e:
    print(f"❌ Failed to load ChatterboxTTS: {e}")
    model = None

try:
    multilingual_model = t3_to(ChatterboxMultilingualTTS.from_pretrained(device=device), torch.bfloat16)
    print("✅ ChatterboxMultilingualTTS loaded successfully")
except Exception as e:
    print(f"❌ Failed to load ChatterboxMultilingualTTS: {e}")
    multilingual_model = None

try:
    vc_model = ChatterboxVC.from_pretrained(device=device)
    print("✅ ChatterboxVC loaded successfully")
except Exception as e:
    print(f"❌ Failed to load ChatterboxVC: {e}")
    vc_model = None

# Test 1: English TTS
print("\n=== Test 1: English TTS (Regular) ===")
if model:
    try:
        text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
        print(f"Generating: {text}")
        wav = model.generate(text, n_timesteps=5)
        ta.save("test-local-1-regular.wav", wav, S3GEN_SR)
        print("✅ Saved to: test-local-1-regular.wav")
    except Exception as e:
        print(f"❌ Error in Test 1: {e}")

# Test 2: Multilingual TTS (French)
print("\n=== Test 2: Multilingual TTS French (Regular) ===")
if multilingual_model:
    try:
        text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
        print(f"Generating: {text}")
        wav = multilingual_model.generate(text, language_id="fr", n_timesteps=5)
        ta.save("test-local-2-regular.wav", wav, multilingual_model.sr)
        print("✅ Saved to: test-local-2-regular.wav")
    except Exception as e:
        print(f"❌ Error in Test 2: {e}")

# Test 3: English TTS Streaming
print("\n=== Test 3: English TTS (Streaming) ===")
if model:
    try:
        text = "This is a streaming test with multiple chunks being generated progressively. Each chunk will be produced in real-time as the model processes the text."
        print(f"Generating streaming: {text}")
        
        # Collect all audio chunks from the stream
        audio_chunks = []
        chunk_count = 0
        
        # Debug: Try streaming with different parameters
        print("Starting streaming...")
        for chunk_audio, metrics in model.generate_stream(text, print_metrics=True, n_timesteps=5):
            if chunk_audio is not None:
                audio_chunks.append(chunk_audio)
                chunk_count += 1
                # print(f"  📦 Chunk {chunk_count}: shape {chunk_audio.shape}")
            else:
                print("  ⚠️  Received None chunk")
        
        print(f"Total chunks received: {chunk_count}")
        
        if audio_chunks:
            # Concatenate all chunks
            full_audio = torch.cat(audio_chunks, dim=-1)
            ta.save("test-local-3-streaming.wav", full_audio.cpu(), model.sr)
            print(f"✅ Saved streaming audio to: test-local-3-streaming.wav ({chunk_count} chunks)")
        else:
            print("❌ No audio chunks were generated during streaming")
            
    except Exception as e:
        print(f"❌ Error in Test 3: {e}")
        import traceback
        traceback.print_exc()

# Test 4: Multilingual TTS Streaming (French)
print("\n=== Test 4: Multilingual TTS French (Streaming) ===")
if multilingual_model:
    try:
        text = "Ceci est un test de streaming multilingue avec plusieurs chunks générés progressivement. Chaque segment audio est produit en temps réel."
        print(f"Generating streaming: {text}")
        
        # Collect all audio chunks from the stream
        audio_chunks = []
        chunk_count = 0
        
        # Debug: Try streaming with different parameters
        print("Starting multilingual streaming...")
        #RTX 5060 TI 16 GB, timesteps=5
        # 5 = 160 ms playback -> 0.313ms TTf
        # 10 = 360 ms playback -> 0.351ms TTF
        # 20 = 760 ms playback -> 0.404ms TTf
        # 50 = 1960 ms playback -> 0.549ms TTf
        # 100= 3960 ms playback -> 0.860ms TTf
        for chunk_audio, metrics in multilingual_model.generate_stream(text, language_id="fr", print_metrics=True, n_timesteps=5):
            if chunk_audio is not None:
                audio_chunks.append(chunk_audio)
                # if (chunk_count == 0):
                #     full_audio = torch.cat(audio_chunks, dim=-1)
                #     ta.save("test-local-4-streaming-first.wav", full_audio.detach().cpu(), multilingual_model.sr)
                chunk_count += 1
                # print(f"  📦 Chunk {chunk_count}: shape {chunk_audio.shape}")
            else:
                print("  ⚠️  Received None chunk")
        
        print(f"Total chunks received: {chunk_count}")
        
        if audio_chunks:
            # Concatenate all chunks
            full_audio = torch.cat(audio_chunks, dim=-1)
            ta.save("test-local-4-streaming.wav", full_audio.detach().cpu(), multilingual_model.sr)
            print(f"✅ Saved streaming audio to: test-local-4-streaming.wav ({chunk_count} chunks)")
        else:
            print("❌ No audio chunks were generated during streaming")
            
    except Exception as e:
        print(f"❌ Error in Test 4: {e}")
        import traceback
        traceback.print_exc()

# Test 5: English with voice prompt (if available)
print("\n=== Test 5: English with voice prompt ===")
if model:
    try:
        # Only test if we have a valid audio prompt file
        AUDIO_PROMPT_PATH = "sample_cloning.wav"
        if os.path.exists(AUDIO_PROMPT_PATH):
            text = "This uses a custom voice prompt for personalized speech synthesis."
            print(f"Using audio prompt: {AUDIO_PROMPT_PATH}")
            wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH, n_timesteps=5)
            ta.save("test-local-5-prompt.wav", wav, model.sr)
            print("✅ Saved to: test-local-5-prompt.wav")
        else:
            print(f"⚠️  Audio prompt file not found: {AUDIO_PROMPT_PATH}")
            print("Skipping voice prompt test...")
    except Exception as e:
        print(f"❌ Error in Test 5: {e}")

# Test 6: English with voice prompt (Streaming)
print("\n=== Test 6: English with voice prompt (Streaming) ===")
if model:
    try:
        # Only test if we have a valid audio prompt file
        AUDIO_PROMPT_PATH = "sample_cloning.wav"
        if os.path.exists(AUDIO_PROMPT_PATH):
            text = "This uses a custom voice prompt for personalized speech synthesis with streaming capabilities. Each chunk will be generated progressively."
            print(f"Generating streaming with audio prompt: {AUDIO_PROMPT_PATH}")
            print(f"Text: {text}")
            
            # Collect all audio chunks from the stream
            audio_chunks = []
            chunk_count = 0
            
            print("Starting voice prompt streaming...")
            for chunk_audio, metrics in model.generate_stream(text, audio_prompt_path=AUDIO_PROMPT_PATH, print_metrics=True, n_timesteps=5):
                if chunk_audio is not None:
                    audio_chunks.append(chunk_audio)
                    chunk_count += 1
                else:
                    print("  ⚠️  Received None chunk")
            
            print(f"Total chunks received: {chunk_count}")
            
            if audio_chunks:
                # Concatenate all chunks
                full_audio = torch.cat(audio_chunks, dim=-1)
                ta.save("test-local-6-prompt-streaming.wav", full_audio.cpu(), model.sr)
                print(f"✅ Saved streaming voice prompt audio to: test-local-6-prompt-streaming.wav ({chunk_count} chunks)")
            else:
                print("❌ No audio chunks were generated during voice prompt streaming")
        else:
            print(f"⚠️  Audio prompt file not found: {AUDIO_PROMPT_PATH}")
            print("Skipping voice prompt streaming test...")
    except Exception as e:
        print(f"❌ Error in Test 6: {e}")
        import traceback
        traceback.print_exc()

# Test 7: Multilingual with voice prompt (Regular)
print("\n=== Test 7: Multilingual with voice prompt (Regular) ===")
if multilingual_model:
    try:
        # Only test if we have a valid audio prompt file
        AUDIO_PROMPT_PATH = "sample_cloning.wav"
        if os.path.exists(AUDIO_PROMPT_PATH):
            text = "Ceci utilise un échantillon vocal personnalisé pour la synthèse vocale multilingue personnalisée."
            print(f"Using audio prompt: {AUDIO_PROMPT_PATH}")
            print(f"Text: {text}")
            wav = multilingual_model.generate(text, language_id="fr", audio_prompt_path=AUDIO_PROMPT_PATH, n_timesteps=5)
            ta.save("test-local-7-multilingual-prompt.wav", wav, multilingual_model.sr)
            print("✅ Saved to: test-local-7-multilingual-prompt.wav")
        else:
            print(f"⚠️  Audio prompt file not found: {AUDIO_PROMPT_PATH}")
            print("Skipping multilingual voice prompt test...")
    except Exception as e:
        print(f"❌ Error in Test 7: {e}")

# Test 8: Multilingual with voice prompt (Streaming)
print("\n=== Test 8: Multilingual with voice prompt (Streaming) ===")
if multilingual_model:
    try:
        # Only test if we have a valid audio prompt file
        AUDIO_PROMPT_PATH = "sample_cloning.wav"
        if os.path.exists(AUDIO_PROMPT_PATH):
            text = "Ceci utilise un échantillon vocal personnalisé pour la synthèse vocale multilingue avec capacités de streaming. Chaque chunk sera généré progressivement."
            print(f"Generating multilingual streaming with audio prompt: {AUDIO_PROMPT_PATH}")
            print(f"Text: {text}")
            
            # Collect all audio chunks from the stream
            audio_chunks = []
            chunk_count = 0
            
            print("Starting multilingual voice prompt streaming...")
            for chunk_audio, metrics in multilingual_model.generate_stream(text, language_id="fr", audio_prompt_path=AUDIO_PROMPT_PATH, print_metrics=True, n_timesteps=5):
                if chunk_audio is not None:
                    audio_chunks.append(chunk_audio)
                    chunk_count += 1
                else:
                    print("  ⚠️  Received None chunk")
            
            print(f"Total chunks received: {chunk_count}")
            
            if audio_chunks:
                # Concatenate all chunks
                full_audio = torch.cat(audio_chunks, dim=-1)
                ta.save("test-local-8-multilingual-prompt-streaming.wav", full_audio.cpu(), multilingual_model.sr)
                print(f"✅ Saved multilingual streaming voice prompt audio to: test-local-8-multilingual-prompt-streaming.wav ({chunk_count} chunks)")
            else:
                print("❌ No audio chunks were generated during multilingual voice prompt streaming")
        else:
            print(f"⚠️  Audio prompt file not found: {AUDIO_PROMPT_PATH}")
            print("Skipping multilingual voice prompt streaming test...")
    except Exception as e:
        print(f"❌ Error in Test 8: {e}")
        import traceback
        traceback.print_exc()

# Test 9: Voice Conversion (Regular)
print("\n=== Test 9: Voice Conversion (Regular) ===")
if vc_model:
    try:
        # Only test if we have valid audio files
        SOURCE_AUDIO_PATH = "cloning_input.wav"
        TARGET_VOICE_PATH = "sample_cloning.wav"
        
        if os.path.exists(SOURCE_AUDIO_PATH) and os.path.exists(TARGET_VOICE_PATH):
            print(f"Converting voice from: {SOURCE_AUDIO_PATH}")
            print(f"Using target voice: {TARGET_VOICE_PATH}")
            
            converted_wav = vc_model.generate(
                audio=SOURCE_AUDIO_PATH,
                target_voice_path=TARGET_VOICE_PATH
            )
            ta.save("test-local-9-vc-regular.wav", converted_wav, vc_model.sr)
            print("✅ Saved to: test-local-9-vc-regular.wav")
        else:
            print(f"⚠️  Audio files not found:")
            print(f"   - Source audio: {SOURCE_AUDIO_PATH}")
            print(f"   - Target voice: {TARGET_VOICE_PATH}")
            print("Skipping voice conversion test...")
    except Exception as e:
        print(f"❌ Error in Test 9: {e}")

# Test 10: Voice Conversion (Streaming) - placeholder for future implementation
print("\n=== Test 10: Voice Conversion (Streaming) ===")
if vc_model:
    try:
        # Only test if we have valid audio files
        SOURCE_AUDIO_PATH = "cloning_input.wav"
        TARGET_VOICE_PATH = "sample_cloning.wav"
        
        if os.path.exists(SOURCE_AUDIO_PATH) and os.path.exists(TARGET_VOICE_PATH):
            print(f"Converting voice from: {SOURCE_AUDIO_PATH} (streaming mode)")
            print(f"Using target voice: {TARGET_VOICE_PATH}")
            
            # Note: generate_stream method doesn't exist yet for VC
            # This is a placeholder for future implementation
            if hasattr(vc_model, 'generate_stream'):
                # Collect all audio chunks from the stream
                audio_chunks = []
                chunk_count = 0
                
                print("Starting voice conversion streaming...")
                for chunk_audio, metrics in vc_model.generate_stream(
                    audio=SOURCE_AUDIO_PATH,
                    target_voice_path=TARGET_VOICE_PATH,
                    print_metrics=True
                ):
                    if chunk_audio is not None:
                        audio_chunks.append(chunk_audio)
                        chunk_count += 1
                    else:
                        print("  ⚠️  Received None chunk")
                
                print(f"Total chunks received: {chunk_count}")
                
                if audio_chunks:
                    # Concatenate all chunks
                    full_audio = torch.cat(audio_chunks, dim=-1)
                    ta.save("test-local-10-vc-streaming.wav", full_audio.cpu(), vc_model.sr)
                    print(f"✅ Saved streaming VC to: test-local-10-vc-streaming.wav ({chunk_count} chunks)")
                else:
                    print("❌ No audio chunks were generated during VC streaming")
            else:
                print("⚠️  generate_stream method not yet implemented for ChatterboxVC")
                print("Using regular generate() method as fallback...")
                converted_wav = vc_model.generate(
                    audio=SOURCE_AUDIO_PATH,
                    target_voice_path=TARGET_VOICE_PATH
                )
                ta.save("test-local-10-vc-fallback.wav", converted_wav, vc_model.sr)
                print("✅ Saved fallback VC to: test-local-10-vc-fallback.wav")
        else:
            print(f"⚠️  Audio files not found:")
            print(f"   - Source audio: {SOURCE_AUDIO_PATH}")
            print(f"   - Target voice: {TARGET_VOICE_PATH}")
            print("Skipping voice conversion streaming test...")
    except Exception as e:
        print(f"❌ Error in Test 10: {e}")

print("\n🎉 All tests completed!")
print(f"Generated audio files are saved in: {os.getcwd()}")
print("\nFiles created:")
print("📁 Regular TTS:")
print("  - test-local-1-regular.wav (English)")
print("  - test-local-2-regular.wav (French)")
print("📁 Streaming TTS:")
print("  - test-local-3-streaming.wav (English)")
print("  - test-local-4-streaming.wav (French)")
print("📁 Voice Prompt (English):")
print("  - test-local-5-prompt.wav (TTS with voice prompt)")
print("  - test-local-6-prompt-streaming.wav (TTS with voice prompt streaming)")
print("📁 Voice Prompt (Multilingual):")
print("  - test-local-7-multilingual-prompt.wav (Multilingual TTS with voice prompt)")
print("  - test-local-8-multilingual-prompt-streaming.wav (Multilingual TTS with voice prompt streaming)")
print("📁 Voice Conversion:")
print("  - test-local-9-vc-regular.wav (VC Regular)")
print("  - test-local-10-vc-streaming.wav (VC Streaming) or test-local-10-vc-fallback.wav (VC Fallback)")
print("\n💡 Note: Watermarking was disabled for this test.")
print("🔧 Note: This test uses local sources directly without package installation.")
print("📝 Note: Test 10 requires generate_stream() method implementation in ChatterboxVC.")
