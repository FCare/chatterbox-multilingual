import torch
import numpy as np
import torchaudio.functional as F_audio

from perth.perth_net.perth_net_implicit.model.perth_net import PerthNet
from perth.perth_net import PREPACKAGED_MODELS_DIR
from perth.watermarker import WatermarkerBase


def _to_tensor(x, device):
    """Convert input to tensor, keeping it on the same device if already a tensor"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.copy())
    elif isinstance(x, torch.Tensor):
        # If already a tensor, keep it as is (don't force copy)
        pass
    return x.to(dtype=torch.float, device=device)


class OptimizedPerthImplicitWatermarker(WatermarkerBase):
    """
    GPU-optimized version of PerthImplicitWatermarker that uses torchaudio.functional.resample
    instead of librosa.resample to avoid GPU->CPU->GPU transfers.
    """
    
    def __init__(self, run_name: str = "implicit", models_dir=PREPACKAGED_MODELS_DIR,
                 device="cpu", perth_net=None):
        assert (run_name is None) or (perth_net is None)
        if perth_net is None:
            self.perth_net = PerthNet.load(run_name, models_dir).to(device)
        else:
            self.perth_net = perth_net.to(device)

    def apply_watermark(self, signal, sample_rate, **_):
        """
        Apply watermark to audio signal, keeping everything on GPU when possible.
        
        Args:
            signal: Audio signal (can be torch.Tensor or np.ndarray)
            sample_rate: Sample rate of the input signal
            
        Returns:
            torch.Tensor: Watermarked signal (stays on same device as input)
        """
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        
        # Convert to tensor and keep on device
        signal = _to_tensor(signal, self.perth_net.device)
        
        # GPU-native resampling using torchaudio
        if change_rate:
            signal = F_audio.resample(
                signal, 
                orig_freq=sample_rate, 
                new_freq=self.perth_net.hp.sample_rate
            )

        # split signal into magnitude and phase
        magspec, phase = self.perth_net.ap.signal_to_magphase(signal)

        # encode the watermark
        magspec = magspec[None].to(self.perth_net.device)
        wm_magspec, _mask = self.perth_net.encoder(magspec)
        wm_magspec = wm_magspec[0]

        # assemble back into watermarked signal
        wm_signal = self.perth_net.ap.magphase_to_signal(wm_magspec, phase)
        
        # GPU-native resampling back to original rate if needed
        if change_rate:
            wm_signal = F_audio.resample(
                wm_signal, 
                orig_freq=self.perth_net.hp.sample_rate, 
                new_freq=sample_rate
            )
            
        return wm_signal

    def get_watermark(self, wm_signal, sample_rate, round=True, **_):
        """
        Extract watermark from signal, GPU-optimized version.
        
        Args:
            wm_signal: Watermarked signal
            sample_rate: Sample rate
            round: Whether to round the watermark prediction
            
        Returns:
            np.ndarray: Extracted watermark
        """
        change_rate = sample_rate != self.perth_net.hp.sample_rate
        
        # Convert to tensor and keep on device
        wm_signal = _to_tensor(wm_signal, self.perth_net.device)
        
        # GPU-native resampling
        if change_rate:
            wm_signal = F_audio.resample(
                wm_signal, 
                orig_freq=sample_rate, 
                new_freq=self.perth_net.hp.sample_rate
            )
            
        wm_magspec, _phase = self.perth_net.ap.signal_to_magphase(wm_signal)
        wm_magspec = wm_magspec.to(self.perth_net.device)
        wmark_pred = self.perth_net.decoder(wm_magspec[None])[0]
        wmark_pred = wmark_pred.clamp(0., 1.)
        wmark_pred = wmark_pred.round() if round else wmark_pred
        
        return wmark_pred.detach().cpu().numpy()