import os
import numpy as np
from ...io import wavread

def extract_audio_from_folder(root: str, n_sources: int = 3) -> Tuple[np.ndarray, int]:
    sample_rate = 16000  # Only 16kHz is supported
    max_samples = int(sample_rate * max_duration)
    
    source_paths = [os.path.join(root, f"src_{i}.wav") for i in range(1, n_sources + 1)]
    
    waveform_src_img = []

    for source_path in source_paths:
        data, _ = wavread(source_path, return_2d=False)
        waveform_src_img.append(data[:max_samples])

    waveform_src_img = np.stack(waveform_src_img, axis=1)  # (n_channels, n_sources, n_samples)

    return waveform_src_img, sample_rate
