import numpy as np
import scipy
from scipy.signal import spectrogram
import wave
import matplotlib
import matplotlib.pyplot as plt

########################################################################
# preprocessing
########################################################################

"""
contruct mel filter bank
"""

# Mel spectrum constants and functions
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 2595.0


def hz_to_mel(hertz):
    """Converts frequencies to mel scale using HTK formula
    -------
    Args:
                              hertz: scalar or numpy.array of frequencies in hertz
    -------
    Returns:
        Object of same size as hertz containing corresponding values on the mels scale
        """

    return _MEL_HIGH_FREQUENCY_Q * np.log10(1.0 + (hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def mel_to_hz(mels):
    """Converts mel scale to frequencies using HTK formula
    -------
    Args:
                              hertz: scalar or numpy.array of frequencies in mels
    -------
    Returns:
        Object of same size as mel containing corresponding values in hertz
        """

    return _MEL_BREAK_FREQUENCY_HERTZ * (10.0 ** (mels / _MEL_HIGH_FREQUENCY_Q) - 1.0)


def mel_frequencies(n_mels=128, *, fmin=0.0, fmax=11025.0):
    """Compute an array of acoustic frequencies tuned to the mel scale.
    -------
    Args:
    n_mels : int > 0 [scalar]
        Number of mel bins.
    fmin : float >= 0 [scalar]
        Minimum frequency (Hz).
    fmax : float >= 0 [scalar]
        Maximum frequency (Hz).
    -------
    Returns:
    bin_frequencies : ndarray [shape=(n_mels,)]
        Vector of ``n_mels`` frequencies in Hz which are uniformly spaced on the Mel
        axis.
    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels)


def fft_frequencies(*, sr=22050, n_fft=2048):
    """
    Args:
    sr : number > 0 [scalar]
        Audio sampling rate
    n_fft : int > 0 [scalar]
        FFT window size
    -------
    Returns:
    freqs : numpy.ndarray [shape=(1 + n_fft/2,)]
        Frequencies ``(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)``
    """

    return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel(
        *,
        sr,
        n_fft,
        n_mels=128,
        fmin=0.0,
        fmax=8000.0,
        dtype=np.float32,
):
    """Create a Mel filter-bank.
    This produces a linear transformation matrix to project
    FFT bins onto Mel-frequency bins.
    -------
    Args
    sr : number > 0 [scalar]
        sampling rate of the incoming signal
    n_fft : int > 0 [scalar]
        number of FFT components
    n_mels : int > 0 [scalar]
        number of Mel bands to generate
    fmin : float >= 0 [scalar]
        lowest frequency (in Hz)
    fmax : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use ``fmax = sr / 2.0``
    dtype : numpy.dtype
        The data type of the output basis.
        By default, uses 32-bit (single-precision) floating point.
    -------
    Returns:
    M : numpy.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix
    """

    if fmax is None:
        fmax = float(sr) / 2

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights


def log_mel_spectrogram(data,
                        audio_sample_rate,
                        log_offset,
                        hop_length,
                        fft_length,
                        n_mels):
    """Convert waveform to a log magnitude mel-frequency spectrogram.
    -------
    Args:
        data: 1D numpy.array of waveform data.
        audio_sample_rate: The sampling rate of data.
        log_offset: Add this to values when taking log to avoid -Infs.
        window_length: length of each window to analyze.
        hop_length: length between successive analysis windows.
        fft_length: length of fft to analyze
    -------
    Returns:
        2D numpy.array of (num_frames, num_mel_bins) consisting of log mel filterbank
        magnitudes for successive frames.
    """
    f, t, spectrogram = scipy.signal.spectrogram(x=data,
                                                 fs=audio_sample_rate,
                                                 nperseg=fft_length,
                                                 noverlap=hop_length,
                                                 nfft=fft_length)
   
    mel_spectrogram = np.matmul(mel(n_fft=fft_length, sr=audio_sample_rate, n_mels=n_mels),
                                   spectrogram)

    return np.log(mel_spectrogram + log_offset)

def data_to_vector_array(data,
                         n_mels=128,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         downsample=True,
                         input_dim=64):
    """
    convert 1darray data to a vector array.

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    
    # 01 calculate the number of dimensions
    
    dims = n_mels * frames
    
    # 02 generate melspectrogram using log_mel_spectrogram
    
    mel_spectrogram = log_mel_spectrogram(data=data.flatten(),
                                          audio_sample_rate=16000,
                                          log_offset=0.0,
                                          hop_length=hop_length,
                                          fft_length=n_fft,
                                          n_mels=n_mels)

    # 03 trim to 200 time data points
    
    mel_spectrogram = mel_spectrogram[:, 0:200]

    
    # 04 calculate total vector size 
    
    vector_array_size = len(mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips (fail safe)
    
    if vector_array_size < 1:
        return np.empty((0, input_dim))

    
    # 06 generate feature vectors by concatenating multiframes (downsample mel spectrogram)
    if downsample:
        new_mels = 32
        new_frames = int(input_dim / new_mels)
        increment = int(n_mels / new_mels)  # value by which to sample the full 128 mels for each discrete time instant in each frame.
        offset = n_mels - new_mels * increment  # ensures all vector arrays have equal size
        assert (input_dim % new_mels == 0)

        vector_array = np.zeros((vector_array_size, new_mels * new_frames))

        for t in range(new_frames):
            new_vec = mel_spectrogram[:, t: t + vector_array_size].T
            vector_array[:, new_mels * t: new_mels * (t + 1)] = new_vec[:, offset::increment]

        return vector_array
    else:
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = mel_spectrogram[:, t: t + vector_array_size].T
        return vector_array


def file_to_vector_array(wav_path,
                         n_mels=128,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         downsample=True,
                         input_dim=64):
    """
    convert wav file to a vector array.

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    

    # 01 loading file with audio module
    
    with wave.open(wav_path, 'r') as wav_file:
        raw_frames = wav_file.readframes(-1)
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()

    temp_buffer = np.empty((num_frames, num_channels, 4), dtype=np.uint8)
    raw_bytes = np.frombuffer(raw_frames, dtype=np.uint8)
    temp_buffer[:, :, :sample_width] = raw_bytes.reshape(-1, num_channels, sample_width)
    temp_buffer[:, :, sample_width:] = (temp_buffer[:, :, sample_width-1:sample_width] >> 7) * 255
    data = temp_buffer.view('<i4').reshape(temp_buffer.shape[:-1])
    
    
    
    return data_to_vector_array(data=data[:, 0],
                                n_mels=128,
                                frames=5,
                                n_fft=1024,
                                hop_length=512,
                                downsample=True,
                                input_dim=64)

def prep(data):
    
        return log_mel_spectrogram(data=data,
                                   audio_sample_rate=16000,
                                   log_offset=0.0,
                                   hop_length=512,
                                   fft_length=1024,
                                   n_mels=128)
    
def transform(mel_spectrogram,
              n_mels=128,
              frames=2,
              downsample=True,
              input_dim=64):
    """
    transform mel spectrogram to a vector array.

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    
    # 01 calculate the number of dimensions
    
    dims = n_mels * frames

    
    # 02 calculate total vector size
    
    vector_array_size = len(mel_spectrogram[0, :]) - frames + 1
    
    
    # 03 skip too short clips (fail safe)
    
    if vector_array_size < 1:
        print('recording too short!')
        return np.zeros(1, input_dims)
    
    # 04 generate feature vectors by concatenating multiframes (downsample mel spectrogram)
    
    if downsample:
        new_mels = 32
        new_frames = int(input_dim / new_mels)
        increment = int(n_mels / new_mels)  # value by which to sample the full 128 mels for each discrete time instant in each frame.
        offset = n_mels - new_mels * increment  # ensures all vector arrays have equal size
        assert (input_dim % new_mels == 0)

        vector_array = np.zeros((vector_array_size, new_mels * new_frames))

        for t in range(new_frames):
            new_vec = mel_spectrogram[:, t: t + vector_array_size].T
            vector_array[:, new_mels * t: new_mels * (t + 1)] = new_vec[:, offset::increment]

        return vector_array
    else:
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = mel_spectrogram[:, t: t + vector_array_size].T
        return vector_array



def score(score_buffer):

    y_pred = np.mean(score_buffer)
    # just print score for now
    print(y_pred, end='\r')
    
'''  
    if y_pred < 446.041404:
        print("normal,    anomaly score: %f" %y_pred, end="\r")
    else:
        print("anomalous, anomaly score: %f" %y_pred, end="\r")
'''