import os
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T

def audio_to_mel(audio, sample_rate):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    ).to(audio.device)
    mel = mel_spectrogram(audio)
    # 将Power Spectrogram转为DB单位
    mel_db = T.AmplitudeToDB()(mel)
    return mel_db

def batch_stft(audio_data, window_length=None):

    window = torch.hann_window(window_length) 
    batch_size, num_channels, _ = audio_data.shape

    # STFT
    stft_data = torch.stft(
        audio_data.reshape(-1, audio_data.size(-1)), # Flatten batch and channel
        n_fft=window_length,
        hop_length=window_length//4,
        win_length=window_length,
        window=window.to(audio_data.device),
        normalized=True, 
        onesided=True,
        return_complex=True,
        center=False
    )

    # Reshape back to (B, C, F, T) format
    _, freq_bins, time_frames = stft_data.shape
    stft_data = stft_data.view(batch_size, num_channels, freq_bins, time_frames)
    
    return stft_data

def batch_mel(audio_data, sample_rate=44100, window_length=None, n_mels=128, f_min=0.0, f_max=None):

    window = torch.hann_window(window_length).to(audio_data.device)
    batch_size, num_channels, _ = audio_data.shape

    # Compute STFT
    stft_data = torch.stft(
        audio_data.reshape(-1, audio_data.size(-1)),  # Flatten batch and channel
        n_fft=window_length,
        hop_length=window_length // 4,
        win_length=window_length,
        window=window,
        normalized=True,
        onesided=True,
        return_complex=True,
        center=False
    )

    # Convert STFT to power spectrogram
    power_spec = stft_data.abs() ** 2

    # Create Mel filter
    mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max, n_stft=window_length//2 + 1,norm='slaney').to(audio_data.device)
    
    # Apply Mel filter
    mel_spec = mel_scale(power_spec)

    # Reshape back to (B, C, F, T) format
    _, n_mels, time_frames = mel_spec.shape
    mel_spec = mel_spec.view(batch_size, num_channels, n_mels, time_frames)
    
    return mel_spec

def L1Loss(x,y):
    loss = F.l1_loss(x, y)
    return loss

def multi_scale_stft_loss(x: torch.Tensor, 
                          y: torch.Tensor, 
                          window_lengths: List[int] = [2048, 512], 
                          clamp_eps: float = 1e-5, 
                          mag_weight: float = 1.0, 
                          log_weight: float = 1.0, 
                          pow: float = 1.0) -> torch.Tensor:
    loss = 0.0
    for w in window_lengths: 
        x_stft = batch_stft(x,w)
        y_stft = batch_stft(y,w)
        x_magnitude = torch.abs(x_stft)
        y_magnitude = torch.abs(y_stft)
        log_magnitude_loss = F.l1_loss(
            x_magnitude.clamp(min=clamp_eps).pow(pow).log10(),
            y_magnitude.clamp(min=clamp_eps).pow(pow).log10(),
        )
        magnitude_loss = F.l1_loss(x_magnitude, y_magnitude)
        loss += log_weight * log_magnitude_loss + mag_weight * magnitude_loss
        
    return loss
    
def mel_spectrogram_loss(
    x: torch.Tensor, y: torch.Tensor, 
    sample_rate: int = 16000,
    n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320], 
    window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
    clamp_eps: float = 1e-5, 
    mag_weight: float = 0.0, 
    log_weight: float = 1.0, 
    pow: float = 1.0, 
    mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0], 
    mel_fmax: List[float] = None,):

    if mel_fmax is None:
        mel_fmax = [sample_rate / 2] * len(n_mels)
    
    loss = 0.0
    for n_mel, win_len, fmin, fmax in zip(n_mels, window_lengths, mel_fmin, mel_fmax):
        
        x_mel = batch_mel(x,sample_rate=sample_rate,window_length=win_len,n_mels=n_mel,f_min=fmin,f_max=fmax)
        y_mel = batch_mel(y,sample_rate=sample_rate,window_length=win_len,n_mels=n_mel,f_min=fmin,f_max=fmax)

        log_loss = F.l1_loss(
            x_mel.clamp(min=clamp_eps).pow(pow).log10(),
            y_mel.clamp(min=clamp_eps).pow(pow).log10(),
        )

        mag_loss = F.l1_loss(
            x_mel, y_mel
        )

        loss += log_weight * log_loss + mag_weight * mag_loss
    
    return loss

def sisdr_loss(x: torch.Tensor, y: torch.Tensor, scaling: bool = True, zero_mean: bool = True, clip_min: float = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio (SISDR) between two audio signals.

    Args:
        x (torch.Tensor): Reference audio signal (B x T).
        y (torch.Tensor): Estimated audio signal (B x T).
        scaling (bool): Whether to apply scale-invariant normalization. Defaults to True.
        zero_mean (bool): Whether to zero-mean the signals before computation. Defaults to True.
        clip_min (float): Minimum possible loss value to clip. Defaults to None.
        eps (float): Small constant to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: The computed SISDR loss (negative SISDR in dB).
    """
    # Ensure inputs have the same shape
    assert x.shape == y.shape, "Input tensors x and y must have the same shape"

    # Reshape tensors for batch processing
    references = x.reshape(x.size(0), 1, -1).permute(0, 2, 1)  # B x T x 1
    estimates = y.reshape(y.size(0), 1, -1).permute(0, 2, 1)   # B x T x 1

    # Zero-mean the signals if specified
    if zero_mean:
        references = references - references.mean(dim=1, keepdim=True)
        estimates = estimates - estimates.mean(dim=1, keepdim=True)

    # Projection of estimates onto references
    references_projection = (references**2).sum(dim=1, keepdim=True) + eps
    references_on_estimates = (references * estimates).sum(dim=1, keepdim=True) + eps

    # Scale normalization
    if scaling:
        scale = references_on_estimates / references_projection
    else:
        scale = 1

    # True and residual components
    e_true = scale * references
    e_res = estimates - e_true

    # Signal and noise energies
    signal = (e_true**2).sum(dim=1)
    noise = (e_res**2).sum(dim=1)

    # Compute SISDR
    sisdr = 10 * torch.log10(signal / (noise + eps) + eps)

    # Negative SISDR for loss
    sisdr_loss = sisdr

    # Apply minimum clipping if specified
    if clip_min is not None:
        sisdr_loss = torch.clamp(sisdr_loss, min=clip_min)

    # Return mean SISDR loss across the batch
    return sisdr_loss.mean()


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor, sample_rate: int) -> dict:
    """
    Compute audio metrics (mel loss and STFT loss) between original and reconstructed audio.

    Args:
        original (torch.Tensor): Original audio tensor (B x T).
        reconstructed (torch.Tensor): Reconstructed audio tensor (B x T).
        sample_rate (int): Sampling rate of the audio.

    Returns:
        dict: Dictionary containing the computed metrics.
    """
    return {
        "mel_loss": mel_spectrogram_loss(original, reconstructed, sample_rate=sample_rate).item(),
        "stft_loss": multi_scale_stft_loss(original, reconstructed).item(),
        "sisdr_loss":sisdr_loss(original.squeeze(dim=0),reconstructed.squeeze(dim=0))
    }


def process_audio_files(original_dir: str, reconstructed_dir: str, sample_rate: int) -> dict:
    """
    Compute average metrics for all audio files in the original and reconstructed directories.

    Args:
        original_dir (str): Path to the directory containing original audio files.
        reconstructed_dir (str): Path to the directory containing reconstructed audio files.
        sample_rate (int): Sampling rate for the audio files.

    Returns:
        dict: Dictionary containing the average metrics.
    """
    # Find all original and reconstructed files
    original_files = sorted(list(Path(original_dir).glob("*.wav")))
    reconstructed_files = sorted(list(Path(reconstructed_dir).glob("*.wav")))

    assert len(original_files) == len(
        reconstructed_files
    ), "Mismatch between the number of original and reconstructed files."

    total_metrics = {"mel_loss": 0.0, "stft_loss": 0.0,"sisdr_loss":0.0}
    num_files = len(original_files)

    for original_path, reconstructed_path in zip(original_files, reconstructed_files):
        print(f"Processing: {original_path.name}")
        # Load audio files
        original_audio, _ = torchaudio.load(original_path)
        reconstructed_audio, _ = torchaudio.load(reconstructed_path)

        # Compute metrics
        metrics = compute_metrics(original_audio.unsqueeze(dim=0), reconstructed_audio.unsqueeze(dim=0), sample_rate=sample_rate)
        total_metrics["mel_loss"] += metrics["mel_loss"]
        total_metrics["stft_loss"] += metrics["stft_loss"]
        total_metrics["sisdr_loss"] += metrics["sisdr_loss"]

    # Compute averages
    average_metrics = {k: v / num_files for k, v in total_metrics.items()}
    return average_metrics


if __name__ == "__main__":
    # Specify directories and sample rate
    original_dir = "/original/"  # Replace with the actual path
    reconstructed_dir = "/reconstructed/"  # Replace with the actual path
    sample_rate = 16000  # Replace with your desired sample rate

    # Process audio files and compute average metrics
    avg_metrics = process_audio_files(original_dir, reconstructed_dir, sample_rate)

    # Print the average metrics
    print("\nAverage Metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
