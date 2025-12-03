from dataclasses import dataclass
from pathlib import Path
import time
from typing import Generator, Tuple, Optional, List
import os
from collections import OrderedDict

import librosa
import numpy as np
import torch
import gc
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.watermarker import OptimizedPerthImplicitWatermarker


REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text

@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS generation"""
    latency_to_first_chunk: Optional[float] = None
    rtf: Optional[float] = None
    total_generation_time: Optional[float] = None
    total_audio_duration: Optional[float] = None
    chunk_count: int = 0

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = OptimizedPerthImplicitWatermarker(device=device)
        
        # Cache system for conditionals
        self._conditionals_cache = OrderedDict()  # Cache LRU
        self._cache_max_size = 50  # Limite du nombre d'entrées dans le cache

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen(use_fp16=False)
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
        )
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main", 
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    def _get_cache_key(self, wav_fpath, exaggeration):
        """Génère une clé unique pour le cache basée sur le path et l'exaggeration."""
        # Convertir le path en path absolu pour éviter les problèmes de paths relatifs
        abs_path = Path(wav_fpath).resolve()
        
        # Créer une clé unique combinant le path et l'exaggeration
        cache_key = f"{abs_path}:{exaggeration}"
        
        # Optionnel: ajouter la taille du fichier et la date de modification
        # pour détecter si le fichier a changé
        try:
            stat = abs_path.stat()
            cache_key += f":{stat.st_size}:{stat.st_mtime}"
        except OSError:
            pass
            
        return cache_key
    
    def _manage_cache_size(self):
        """Gère la taille du cache en supprimant les entrées les plus anciennes."""
        while len(self._conditionals_cache) >= self._cache_max_size:
            # Supprime l'entrée la plus ancienne (FIFO)
            oldest_key = next(iter(self._conditionals_cache))
            del self._conditionals_cache[oldest_key]
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        # Générer la clé de cache
        cache_key = self._get_cache_key(wav_fpath, exaggeration)
        
        # Vérifier si on a déjà calculé ces conditionals
        if cache_key in self._conditionals_cache:
            # Cache hit - récupérer depuis le cache
            self.conds = self._conditionals_cache[cache_key]
            # Déplacer vers la fin pour LRU
            self._conditionals_cache.move_to_end(cache_key)
            print(f"Cache hit for {wav_fpath} with exaggeration={exaggeration}")
            return
            
        # Cache miss - calculer les conditionals
        print(f"Cache miss for {wav_fpath} with exaggeration={exaggeration}")
        
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        # Handle tensor ou numpy array retournés par Voice Encoder optimisé
        if isinstance(ve_embed, np.ndarray):
            ve_embed = torch.from_numpy(ve_embed)
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=ve_embed.dtype),
        ).to(device=self.device)
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        
        # Ajouter au cache
        self._manage_cache_size()  # S'assurer qu'on ne dépasse pas la limite
        self._conditionals_cache[cache_key] = self.conds

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        # cache optimization params
        max_new_tokens=1000, 
        max_cache_len=1500, # Affects the T3 speed, hence important
        # t3 sampling params
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        n_timesteps = 5,
        t3_params={},
        print_metrics=False,
    ):
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                **t3_params,
            )

            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]
            def drop_bad_tokens(tokens):
                # Use torch.where instead of boolean indexing to avoid sync
                mask = tokens < 6561
                # Count valid tokens without transferring to CPU
                valid_count = torch.sum(mask).item()
                # Create output tensor of the right size
                result = torch.zeros(valid_count, dtype=tokens.dtype, device=tokens.device)
                # Use torch.masked_select which is more CUDA-friendly
                result = torch.masked_select(tokens, mask)
                return result

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = drop_bad_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
                n_timesteps=n_timesteps,
            )
            watermarked_wav = self.watermarker.apply_watermark(wav.squeeze(0), sample_rate=self.sr).unsqueeze(0)
        gc.collect()
        torch.cuda.empty_cache()
        return watermarked_wav.detach().cpu()

    def _process_token_buffer(
        self,
        token_buffer,
        all_tokens_so_far,
        context_window,
        start_time,
        metrics,
        print_metrics,
        fade_duration=0.02,  # seconds to apply linear fade-in on each chunk
        n_timesteps = 5,
    ):

        # Combine buffered chunks of tokens
        new_tokens = torch.cat(token_buffer, dim=-1)

        # Build tokens_to_process by including a context window
        if len(all_tokens_so_far) > 0:
            context_tokens = (
                all_tokens_so_far[-context_window:]
                if len(all_tokens_so_far) > context_window
                else all_tokens_so_far
            )
            tokens_to_process = torch.cat([context_tokens, new_tokens], dim=-1)
            context_length = len(context_tokens)
        else:
            tokens_to_process = new_tokens
            context_length = 0

        # Drop any invalid tokens and move to the correct device
        clean_tokens = drop_invalid_tokens(tokens_to_process).to(self.device)
        if len(clean_tokens) == 0:
            return None, 0.0, False
        def drop_bad_tokens(tokens):
            # Use torch.where instead of boolean indexing to avoid sync
            mask = tokens < 6561
            # Count valid tokens without transferring to CPU
            valid_count = torch.sum(mask).item()
            # Create output tensor of the right size
            result = torch.zeros(valid_count, dtype=tokens.dtype, device=tokens.device)
            # Use torch.masked_select which is more CUDA-friendly
            result = torch.masked_select(tokens, mask)
            return result

        clean_tokens = drop_bad_tokens(clean_tokens)
        if len(clean_tokens) == 0:
            return None, 0.0, False


        # Run S3Gen inference to get a waveform (1 × T)
        audio_chunk, _ = self.s3gen.inference(
            speech_tokens=clean_tokens,
            ref_dict=self.conds.gen,
            n_timesteps=n_timesteps,
        )

        # If we have context tokens, crop out the samples corresponding to them
        if context_length > 0:
            samples_per_token = audio_chunk.shape[-1] / len(clean_tokens)
            skip_samples = int(context_length * samples_per_token)
            audio_chunk = audio_chunk[..., skip_samples:]

        if audio_chunk.shape[-1] == 0:
            return None, 0.0, False

        # Apply a short linear fade-in on the new chunk to smooth boundaries
        fade_samples = int(fade_duration * self.sr)
        if fade_samples > 0:
            if fade_samples > audio_chunk.shape[-1]:
                fade_samples = audio_chunk.shape[-1]
            # GPU-native fade-in avec torch.linspace
            fade_in = torch.linspace(0.0, 1.0, fade_samples,
                                   dtype=audio_chunk.dtype,
                                   device=audio_chunk.device)
            audio_chunk[..., :fade_samples] *= fade_in

        # Compute audio duration and watermark
        audio_duration = audio_chunk.shape[-1] / self.sr
        watermarked_chunk = self.watermarker.apply_watermark(audio_chunk.squeeze(0), sample_rate=self.sr).unsqueeze(0)

        # Update first‐chunk latency metric
        if metrics.chunk_count == 0:
            metrics.latency_to_first_chunk = time.time() - start_time
            if print_metrics:
                print(f"Latency to first chunk: {metrics.latency_to_first_chunk:.3f}s")

        metrics.chunk_count += 1
        return watermarked_chunk, audio_duration, True
    
    def generate_stream(
        self,
        text,
        language_id, # for API compatibility; not used in this model
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        # streaming parameters
        stream_chunk_size: List[int] = [20,50,100],  # Tokens per chunk - Ramp up 
        context_window = 50,
        fade_duration=0.02,  # seconds to apply linear fade-in on each chunk
        print_metrics: bool = True,
        # cache optimization params
        max_new_tokens=1000, 
        max_cache_len=1500, # Affects the T3 speed, hence important
        # t3 sampling params
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        n_timesteps = 5,
        t3_params={},
    ) -> Generator[Tuple[torch.Tensor, StreamingMetrics], None, None]:
        start_time = time.time()
        metrics = StreamingMetrics()
        print("Multi language streaming in generation")
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        try:
            text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        except TypeError:
            text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        total_audio_length = 0.0
        all_tokens_processed = []  # Keep track of all tokens processed so far

        with torch.inference_mode():
            for token_chunk in self.t3.inference_stream(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                stream_chunk_size=stream_chunk_size,
                **t3_params
            ):
                # Extract only the conditional batch.

                # Process each chunk immediately
                wav, audio_duration, success = self._process_token_buffer(
                    [token_chunk], all_tokens_processed, context_window, 
                    start_time, metrics, print_metrics, fade_duration, n_timesteps
                )
                
                if success:
                    total_audio_length += audio_duration
                    yield wav.detach().cpu(), metrics
                
                # Update all_tokens_processed with the new tokens
                if len(all_tokens_processed) == 0:
                    all_tokens_processed = token_chunk
                else:
                    all_tokens_processed = torch.cat([all_tokens_processed, token_chunk], dim=-1)

            # Final metrics calculation
            metrics.total_generation_time = time.time() - start_time
            metrics.total_audio_duration = total_audio_length
            if total_audio_length > 0:
                metrics.rtf = metrics.total_generation_time / total_audio_length
                if print_metrics:
                    print(f"Total generation time: {metrics.total_generation_time:.3f}s")
                    print(f"Total audio duration: {metrics.total_audio_duration:.3f}s")
                    print(f"RTF (Real-Time Factor): {metrics.rtf:.3f}")
                    print(f"Total chunks yielded: {metrics.chunk_count}")
        gc.collect()
        torch.cuda.empty_cache()

    def clear_conditionals_cache(self):
        """Vide le cache des conditionals."""
        self._conditionals_cache.clear()
        
    def get_cache_info(self):
        """Retourne des informations sur l'état du cache."""
        return {
            'size': len(self._conditionals_cache),
            'max_size': self._cache_max_size,
            'keys': list(self._conditionals_cache.keys())
        }
