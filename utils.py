import re
import pretty_midi
import numpy as np
import mido
import math
from numpy.lib.stride_tricks import as_strided
import os
import json
import datetime
import matplotlib.pyplot as plt


def mid2piano_roll(midipath, pedal=False, onset=False, fs=100):
    # TODO: parse with mido may contains error if multiple set_temtpo signal exist
    assert (pedal and onset) is not True, 'pedal + onset is not reasonable'
    # read total length
    mid = mido.MidiFile(midipath)
    t = 0
    for message in mid:
        t += message.time
    l = int(math.ceil(t * fs))
    y = np.zeros((l, 88))
    if onset:
        mid = mido.MidiFile(midipath)
        t = 0
        for message in mid:
            t += message.time
            if message.type == 'note_on':
                frame = int(t * fs)
                y[frame, message.note - 21] = 1
    else:
        pretty_midi.pretty_midi.MAX_TICK = 1e10
        midi_data = pretty_midi.PrettyMIDI(midipath)
        midi_roll = midi_data.get_piano_roll(fs=fs)
        midi_roll = midi_roll.T  # (time, feature)
        midi_roll = midi_roll[:, 21:109]
        midi_roll = (midi_roll >= 1).astype(np.int)
        if midi_roll.shape[0] <= y.shape[0]:
            y[:midi_roll.shape[0], :] = midi_roll
        else:
            y = midi_roll[:y.shape[0], :]

    if pedal:
        mid = mido.MidiFile(midipath)
        t = 0
        pedal = []
        for message in mid:
            t += message.time
            if message.type == 'control_change' and message.control == 64:
                if message.value == 127:
                    pedal_on = 1
                elif message.value == 0:
                    pedal_on = 0
                pedal.append([pedal_on, t])

        pedal_array = np.zeros((l))
        temp_on = 0
        temp_frame = 0

        for el in pedal:
            frame = int(el[1] * fs)
            if temp_on == 1:
                pedal_array[temp_frame:frame] = 1
            temp_on = el[0]
            temp_frame = frame
        if temp_on == 1:
            pedal_array[temp_frame:] = 1

        for n in xrange(y.shape[0] - 1):
            if pedal_array[n] == 1 and pedal_array[n + 1] == 1:
                y[n + 1, :] = (y[n, :] + y[n + 1, :] >= 1).astype(np.int)
    return y


def piano_roll2chroma_roll(piano_roll):
    chroma_roll = np.zeros((piano_roll.shape[0], 12)) # (time, class)
    for n in range(piano_roll.shape[1]):
        chroma_roll[:, n % 12] += piano_roll[:, n]
    chroma_roll = (chroma_roll >= 1).astype(np.int)
    return chroma_roll


def mid2chroma_roll(midipath, pedal=False, onset=False):
    piano_roll = mid2piano_roll(midipath, pedal=pedal, onset=onset)
    chroma_roll = piano_roll2chroma_roll(piano_roll)
    return chroma_roll


def array2stack(array, length, hop=1):
    assert (array.shape[0]-length)%hop == 0, 'length of array is not fit. l={:d}, length={:d}, hop={:d}'\
        .format(array.shape[0], length, hop)
    strides = array.strides
    stack = as_strided(array, ((array.shape[0] - length)//hop + 1, length, array.shape[1]),
                       (strides[0]*hop, strides[0], strides[1]))
    return stack


def overlap_stack2array(stack):
    # TODO: what if hop != stack.shape[1]//2 ?
    hop = stack.shape[1] // 2
    length = (stack.shape[0] + 1) * hop
    array = np.zeros((length, stack.shape[2]))
    array[:hop//2, :] = stack[0, :hop//2,:]
    for n in xrange(stack.shape[0]):
        array[n*hop + hop//2: n*hop + 3*hop//2, :] = stack[n, hop//2: 3*hop//2, :]
    array[(stack.shape[0]-1)*hop + 3*hop//2:, :] = stack[stack.shape[0]-1, 3*hop//2:, :]

    return array


def onset2delayed(onset, delay_len=10):
    rolled_onset = np.zeros(onset.shape)
    for k in range(delay_len):
        temp = np.roll(onset, k, axis=0)
        temp[0, :] = 0
        weight = math.sqrt((delay_len - k) / float(delay_len))
        rolled_onset += temp * weight
    rolled_onset[rolled_onset > 1] = 1
    return rolled_onset


def save_config(config,):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    param_path = os.path.join(config.save_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def record_as_text(config, text):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    record_txt = config.save_dir + '/' + 'summary.txt'
    f = open(record_txt, 'a')
    f.write(text)
    f.close()


def plot_piano_roll(piano_roll, plot_range=None, segment_len=100):
    if not plot_range:
        plot_range = [0, piano_roll.shape[0]]
    plt.imshow(piano_roll[plot_range[0]: plot_range[1]].T, interpolation='nearest', aspect='auto', origin='lower', vmin=0, vmax=1, cmap=plt.get_cmap('gray_r'))
    x_ticks_sec = range(plot_range[0] // 100, plot_range[1] // 100)
    plt.xticks([el * 100 for el in x_ticks_sec], x_ticks_sec)
    octaves = range(piano_roll.shape[-1] // 12)
    plt.yticks(octaves, [str(el + 21) for el in octaves])
    plt.colorbar(ticks=[0, 1], pad=0.01, aspect=10)
    edges = range(segment_len * plot_range[0] // segment_len, segment_len * plot_range[1] // segment_len)
    for el in edges:
        plt.plot([el, el], [-0.5, piano_roll.shape[-1] - 0.5], color='red', linewidth=1, linestyle="--", alpha=0.8)


def plot_spectrogram(spectrogram, plot_range=None, segment_len=100):
    if not plot_range:
        plot_range = [0, spectrogram.shape[0]]
        plt.imshow(spectrogram[plot_range[0]: plot_range[1]].T, interpolation='nearest', aspect='auto', origin='lower', cmap=plt.get_cmap('gray_r'))
    x_ticks_sec = range(plot_range[0] // 100, plot_range[1] // 100)
    plt.xticks([el * 100 for el in x_ticks_sec], x_ticks_sec)
    plt.colorbar(pad=0.01, aspect=10)
    edges = range(segment_len * plot_range[0] // segment_len, segment_len * plot_range[1] // segment_len)
    for el in edges:
        plt.plot([el, el], [-0.5, spectrogram.shape[-1] -0.5], color='red', linewidth=1, linestyle="--", alpha=0.8)