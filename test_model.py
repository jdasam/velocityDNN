from __future__ import division

import tensorflow as tf
import numpy as np
import extract_spectrogram as ex
import make_records as mr
import os
import librosa
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt


MODEL_DIR = 'results/type1_10_batch32/10_39-108/cnn_type_1/adam_1.000e-03/fold_1/'
NORM_DIR = 'input/10_39-108/norm.npz'
SEG_LEN = 100
SR = 10

c = tf.ConfigProto()
c.gpu_options.visible_device_list = '0'
sess = tf.Session(config=c)

dir(tf.contrib)  # https://github.com/tensorflow/tensorflow/issues/10130
saver = tf.train.import_meta_graph(MODEL_DIR + 'epoch-32.meta')
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess, MODEL_DIR + 'epoch-32')

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("input/input:0")
pred = graph.get_tensor_by_name("output/pred:0")
sigmoid_pred = graph.get_tensor_by_name("prediction:0")

learning_rate = graph.get_tensor_by_name('lr:0')
is_train = graph.get_tensor_by_name('is_train:0')
keep_prob = graph.get_tensor_by_name('dropout_prob:0')

# load norm_factor
norm = np.load(NORM_DIR)
mean = norm['mean']
std = norm['std']

wav_list = ['real/' + el for el in os.listdir('real')]

midi_map = ex.fft2midi_mat(2048, 16000, 39, 108, 1, True)


def wav2spec(filename):
    y, sr = librosa.load(filename, sr=16000)
    log_STFT = ex.wav2stft(y, 2048, 1600)
    spec = np.dot(midi_map, log_STFT[1: midi_map.shape[1] + 1, :])  # ignore f=0 bin
    spec = spec.T  # shape of (time, frequency)
    return spec

specs = [wav2spec(el) for el in wav_list]
specs = [mr.normalize(mr.pad2d(el, SEG_LEN), mean, std) for el in specs]

preds = []

for n in xrange(len(wav_list)):
    spec = specs[n]
    spec_segs = mr.slice(spec, SEG_LEN)
    pred_segs = []
    for m in range(spec_segs.shape[0]):
        sig_pred = sess.run([sigmoid_pred],
                            feed_dict={input: spec_segs[m:m+1, :, :],
                                       keep_prob: np.ones(2),
                                       is_train: False})
        sig_pred = np.pad(sig_pred[0][0, :, :], ((2, 2), (0, 0)), 'constant')
        pred_segs.append(sig_pred)
    preds.append(np.concatenate(pred_segs, axis=0))


note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def midi_name(midi_num):
    octave = str(midi_num // 12)
    note = note_names[(midi_num + 3) % 12]
    return octave, note


def draw(file_name, spec, pred, sr=10):
    plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    plt.subplot(gs[0])
    plt.title(file_name)
    tick_range = range(spec.shape[0]//sr)
    plt.xticks([el*sr for el in tick_range], tick_range)
    plt.xlabel('time(sec)')
    plt.imshow(spec.T, interpolation='nearest', aspect='auto',
               origin='lower', cmap=plt.get_cmap('gray_r'))
    mpl.rcParams.update({'font.size': 10})
    plt.subplot(gs[1])
    plt.imshow(pred.T, interpolation='nearest', aspect='auto',
               origin='lower', cmap=plt.get_cmap('gray_r'))
    edge = range(8, 44, 12)
    for el in edge:
        plt.plot([0, spec.shape[0]], [el, el], color='red', linewidth=1, linestyle="--", alpha=0.8)
    midi_ticks = [midi_name(el)[1] + midi_name(el)[0] for el in range(40, 84)]
    plt.yticks(range(44), midi_ticks)
    plt.xticks([el * sr for el in tick_range], tick_range)
    plt.xlabel('time(sec)')
    plt.savefig('images/type1_10{}.png'.format(file_name), bbox_inches='tight')


for n in xrange(len(wav_list)):
    file_name = wav_list[n].split('/')[-1].replace('.wav', '')
    spec = specs[n]
    pred = preds[n]
    draw(file_name, spec, pred, sr=10)