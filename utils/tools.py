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

import pywt

def batch_wdt(audio_data, wavelet='db1', max_level=3,mode='symmetric'):
    """
    对每个音频信号进行小波包变换并返回所有频率分量。

    Args:
        audio_data (torch.Tensor): 形状为 (B, C, T) 的音频数据张量
        wavelet (str): 小波类型，默认为 'db1'（Haar 小波）
        max_level (int): 小波包变换的最大层数

    Returns:
        torch.Tensor: 形状为 (B, C, T1) 的张量，包含所有分量的信号
    """
    B, C, T = audio_data.shape
    all_components = []

    # 对每个音频信号进行小波包变换
    for b in range(B):
        components = []
        for c in range(C):
            signal = audio_data[b, c].cpu().numpy()  # 将每个信号转换为 numpy 数组
            wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode=mode, maxlevel=max_level)
            # 获取小波包变换的频率节点
            nodes = wp.get_level(max_level, 'freq')
            
            # 获取所有频率分量
            for node in nodes:
                components.append(torch.tensor(node.data))  # 转换每个分量为 Tensor
        # 将每个 batch 的所有分量合并
        all_components.append(torch.stack(components, dim=0))  # 按照频率维度进行堆叠

    # 将所有分量转换为形状 (B, C, T1)
    return torch.stack(all_components, dim=0).to(audio_data.device)