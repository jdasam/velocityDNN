import numpy as np
import librosa
import os
import multiprocessing
import time
import argparse
import scipy


def wav2stft(y, n_fft, hop_length, c=1000):
    """ convert wave file into logarithmic compressed magnitude of stft.
    the signal `y` is padded so that center of frame `stft[:, t]` is corresponds to `y[t * hop_length]`.
    Hamming window is used.

    Args:
    y: monophonic waveform signal
    n_fft: fft window size
    hop_length: hop size

    Return: log amplitude of stft frames size of ( n_fft // 2 , t)
    """
    stft = librosa.stft(y, n_fft, hop_length, window=scipy.signal.hamming(n_fft))
    stft = np.abs(stft)
    return librosa.logamplitude((c * stft) + 1)


def fft2midi_mat(fft_size, sr, midi_min, midi_max, midi_res, del_duplicate=False):
    """ make triangular mapping matrix M
    that midiscale_spectrogram = midimap * stft

    Args:
    n_fft: fft window size
    sr: sample rate of input
    midi_min: cut-off midi number(minimum). lowest key of piano is 21
    dele_duplicate: if True, delete duplicated or empty midi bins due to sparse FFT bins in low freqency.

    Return: mapping matrix of size (n_midi_bin, n_fft/2)
    """
    n_midi_bin = int((midi_max - midi_min) / midi_res + 1)
    map_mx = np.zeros(shape=(n_midi_bin, fft_size / 2), dtype='float32')
    freq_resolution = sr/fft_size
    freq_fft = np.array(range(fft_size/2))*freq_resolution
    for fft_bin in range(1, fft_size/2):
        midinum = librosa.core.hz_to_midi(freq_fft[fft_bin])
        index_floor = int(midinum / midi_res)
        midi_frac = midinum/midi_res - index_floor
        if midi_min <= index_floor*midi_res <= midi_max:
            map_mx[index_floor-int(midi_min/midi_res), fft_bin] = 1-midi_frac
        if midi_min <= index_floor*midi_res+1 <= midi_max:
            map_mx[index_floor+1-int(midi_min/midi_res), fft_bin] = midi_frac

    last_nonzero_freq = np.argwhere(map_mx.sum(axis=0) != 0)[-1][0]  # delete all-zero coef-frequencies
    map_mx = map_mx[:, 1:last_nonzero_freq+1]  # ignore f=0 bin
    for n in range(n_midi_bin):
        if np.sum(map_mx[n, :]) != 0:
            map_mx[n, :] = map_mx[n, :] / np.sum(map_mx[n, :])
    if del_duplicate:
        delete_midi=[]
        for n in range(n_midi_bin-1):
            if np.sum(map_mx[n, :]) == 0 or np.array_equal(map_mx[n, :],map_mx[n+1, :]):
                delete_midi.append(n)
        map_mx = np.delete(map_mx, delete_midi,axis=0)

    return map_mx


def build_input(filelist, save_dir_root, n_process=4, map=None, n_fft=2048, sr=16000, hop=1600):
    for n in xrange(len(filelist)):
        if n % 10 == 0:
            print "process#{:d} progress: #file {:d}".format(n_process, n)
        filename = filelist[n].split('/')[-1].strip('.mp3')
        if os.path.isfile(save_dir_root + '/' + filename + '.npy'):
            pass
        else:
            y, sr = librosa.load(filelist[n], sr=sr)
            log_STFT = wav2stft(y, n_fft, hop)

            if map is None:
                spec = log_STFT
            else:
                spec = np.dot(map, log_STFT[1: map.shape[1] + 1, :])  # ignore f=0 bin
            spec = spec.T  # shape of (time, frequency)
            if not os.path.isdir(save_dir_root):
                os.makedirs(save_dir_root)
            np.save(save_dir_root + '/' + filename + '.npy', spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help='output folder name')
    parser.add_argument("-sr", "--sample_rate", type=int, default=44100, help="fft_length")
    parser.add_argument("-f", "--n_fft", type=int, default=8192, help="fft_length")
    parser.add_argument("-hop", "--hop_size", type=int, default=441, help="hop_size")
    parser.add_argument("-l", "--midi_comp", action='store_true', help="midi_scale_compression")
    parser.add_argument("-d", "--del_dup", action='store_true', help="delete_duplicate_fft_bin")
    parser.add_argument("-low", "--lowest_bin", type=int, default=28, help="lower midi range")  # 39
    parser.add_argument("-up", "--highest_bin", type=int, default=71, help="upper midi range")  # 108
    args = parser.parse_args()

    num_processer = 4
    t = time.time()
    target_folder = "synth"
    save_dir_root = "data/" + args.name

    filelist = [os.path.join(target_folder, el) for el in os.listdir(target_folder)]

    if args.midi_comp:
        midi_map = fft2midi_mat(args.n_fft, args.sample_rate, args.lowest_bin, args.highest_bin, 1, args.del_dup)
    else:
        midi_map = None
    pool = multiprocessing.Pool(processes=num_processer)
    for n_process in range(num_processer):
        p = pool.apply_async(build_input, (filelist[len(filelist)*n_process//4: len(filelist)*(n_process+1)//4],
                                           save_dir_root, n_process, midi_map, args.n_fft, args.sample_rate, args.hop_size))
    pool.close()
    pool.join()
    t_end = time.time()-t
    print "duration: {:f} sec".format(t_end)