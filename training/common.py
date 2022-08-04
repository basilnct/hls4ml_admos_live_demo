"""
 @file   common.py
 @brief  Commonly used script
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# default
import csv
import glob
import argparse
import itertools
import re
import sys
import os

# additional
import numpy
import wave
import scipy
from scipy.signal import spectrogram
import librosa
import librosa.core
import librosa.feature
import yaml
from tqdm import tqdm

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"


########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-c', '--config', type=str, default="ad08.yml", help="specify yml config")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("HLS4ML TOYADMOS ANOMALY DETECTION\nversion {}".format(__versions__))
        print("===============================\n")
    return args


########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param


########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( int )
    """
    
    try:
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
    	return temp_buffer.view('<i4').reshape(temp_buffer.shape[:-1])
   
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))



def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def save_dat(data, filename):
    numpy.savetxt(filename, data, delimiter=' ', newline='\n', fmt='%g')


########################################################################


########################################################################
# feature extractor
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

    return _MEL_HIGH_FREQUENCY_Q * numpy.log10(1.0 + (hertz / _MEL_BREAK_FREQUENCY_HERTZ))


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

    mels = numpy.linspace(min_mel, max_mel, n_mels)

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

    return numpy.fft.rfftfreq(n=n_fft, d=1.0 / sr)


def mel(
        *,
        sr,
        n_fft,
        n_mels=128,
        fmin=0.0,
        fmax=8000.0,
        dtype=numpy.float32,
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
    weights = numpy.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

    fdiff = numpy.diff(mel_f)
    ramps = numpy.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = numpy.maximum(0, numpy.minimum(lower, upper))

    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, numpy.newaxis]

    return weights


def log_mel_spectrogram(data,
                        audio_sample_rate,
                        log_offset,
                        window_length,
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
                                                 nperseg=window_length,
                                                 noverlap=hop_length,
                                                 nfft=fft_length)
    mel_spectrogram = numpy.matmul(mel(n_fft=fft_length, sr=audio_sample_rate, n_mels=n_mels),
                                   spectrogram)

    return numpy.log(mel_spectrogram + log_offset)


def file_to_vector_array(file_path,
                         n_mels=128,
                         frames=2,
                         n_fft=1024,
                         hop_length=512,
                         downsample=True,
                         input_dim=64):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using log_mel_spectrogram
    y, sr = file_load(file_path)
    mel_spectrogram = log_mel_spectrogram(data=y,
                                          audio_sample_rate=sr,
                                          log_offset=0.0,
                                          window_length=n_fft,
                                          hop_length=hop_length,
                                          fft_length=n_fft,
                                          n_mels=n_mels)
    # 03.5 trim to 50 to 250
    mel_spectrogram = mel_spectrogram[:, 50:250]

    # 04 calculate total vector size
    vector_array_size = len(mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes

    # downsample mel spectrogram
    if downsample:
        new_mels = 32
        new_frames = int(input_dim / new_mels)
        increment = int(n_mels / new_mels)  # value by which to sample the full 128 mels for each discrete time instant in each frame.
        offset = n_mels - new_mels * increment  # ensures all vector arrays have equal size
        assert (input_dim % new_mels == 0)

        vector_array = numpy.zeros((vector_array_size, new_mels * new_frames))

        for t in range(new_frames):
            new_vec = mel_spectrogram[:, t: t + vector_array_size].T
            vector_array[:, new_mels * t: new_mels * (t + 1)] = new_vec[:, offset::increment]

        return vector_array
    else:
        vector_array = numpy.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = mel_spectrogram[:, t: t + vector_array_size].T
        return vector_array


# load dataset
def select_dirs(param):
    """
    param : dict
        baseline.yaml data

    return :
        dirs :  list [ str ]
            load base directory list of dev_data
    """
    logger.info("load_directory <- development")
    dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    dirs = sorted(glob.glob(dir_path))
    return dirs


########################################################################


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    print('dir_path: {}'.format(dir_path))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    logger.info("target_dir : {}".format(target_dir + "_" + id_name))

    # development
    normal_files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                             dir_name=dir_name,
                                                                             prefix_normal=prefix_normal,
                                                                             id_name=id_name,
                                                                             ext=ext)))
    normal_labels = numpy.zeros(len(normal_files))
    anomaly_files = sorted(
        glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                              dir_name=dir_name,
                                                                              prefix_anomaly=prefix_anomaly,
                                                                              id_name=id_name,
                                                                              ext=ext)))
    anomaly_labels = numpy.ones(len(anomaly_files))
    files = numpy.concatenate((normal_files, anomaly_files), axis=0)
    labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
    logger.info("test_file  num : {num}".format(num=len(files)))
    if len(files) == 0:
        logger.exception("no_wav_file!!")
    print("\n========================================")

    return files, labels


########################################################################

def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=128,
                         frames=2,
                         n_fft=1024,
                         hop_length=512,
                         downsample=True):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames
    print(dims)

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            downsample=downsample)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        logger.exception("no_wav_file!!")

    logger.info("train_file num : {num}".format(num=len(files)))
    return files
########################################################################
