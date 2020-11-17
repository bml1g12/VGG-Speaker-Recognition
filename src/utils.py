# Third Party
import librosa
import numpy as np
import time as timelib
import scipy
import soundfile as sf
import scipy.signal as sps
from scipy import interpolate
import librosa
from scipy import signal


# ===============================================
#       code from Arsha for loading data.
# ===============================================
def read_audio(current_file, sample_rate=None, mono=True):
    """Read audio file
    Parameters
    ----------
    current_file : str
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.
    Returns
    -------
    y : (n_samples, n_channels) np.array
        Audio samples.
    sample_rate : int
        Sampling rate.
    Notes
    -----
    """

    y, file_sample_rate = sf.read(
        current_file, dtype="float32", always_2d=True
    )

    # convert to mono
    if mono and y.shape[1] > 1:
        y = np.mean(y, axis=1, keepdims=True)

    # resample if sample rates mismatch
    if (sample_rate is not None) and (file_sample_rate != sample_rate):
        if y.shape[1] == 1:
            # librosa expects mono audio to be of shape (n,), but we have (n, 1).
            y = librosa.core.resample(y[:, 0], file_sample_rate, sample_rate)[:, None]
        else:
            y = librosa.core.resample(y.T, file_sample_rate, sample_rate).T
    else:
        sample_rate = file_sample_rate

    return y, sample_rate

def load_wav(vid_path, sr, mode='train'):

    wav, sr_ret = read_audio(vid_path, sample_rate=sr, mono=True)
    
    print("finish loading wav", timelib.time()-t1)
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time-spec_len)
            spec_mag = mag_T[:, randtime:randtime+spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


