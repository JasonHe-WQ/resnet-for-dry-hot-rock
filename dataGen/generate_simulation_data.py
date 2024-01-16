from __future__ import print_function
from matplotlib import pyplot as plt
# from import_data_label_mod import input_seismic

# import tensorflow as tf
import numpy as np
# import os
import csv

# from array import *

# sample_num = 40
# sample_length = 1600
sample_num = 500
sample_length = 5000
# wavelet_length = 200
wavelet_length = 2500
frequ_start = 20
frequ_end = 300
ampl_start = 200
ampl_end = 300
SNR = [0]
save_path = '.'


def ricker_wavelet(f, a, length):
    dt = 0.001
    length = length * dt
    wave_time = np.linspace(-length / 2, (length - dt) / 2, length / dt)
    ricker = (1. - 2. * (np.pi ** 2) * (f ** 2) * (wave_time ** 2)) * np.exp(
        -(np.pi ** 2) * (f ** 2) * (wave_time ** 2)) * a
    return ricker


def generate_ricker(frequence_seq, amplitude_seq, num):
    for i in range(num):
        data = ricker_wavelet(frequence_seq[i], amplitude_seq[i] / 100, wavelet_length)
        if i == 0:
            gen_data = data
        else:
            gen_data = np.concatenate((gen_data, data), axis=0)
    return gen_data


def get_sim_label(ori_data):
    time = len(ori_data)
    sim_label = np.full((time, 2), 0)
    for i in range(time):
        if i == 0 or i == time - 1 or i == 1 or i == time - 2:
            sim_label[i, 0] = 1
        else:
            if abs(ori_data[i - 2]) < 0.001 and abs(ori_data[i - 1]) < 0.001 and abs(ori_data[i + 2]) < 0.001 and abs(
                    ori_data[i + 1]) < 0.001 and abs(ori_data[i]) < 0.001:
                sim_label[i, 0] = 1
            else:
                sim_label[i, 1] = 1
    return sim_label


def add_nosie(original_data, a_SNR):
    power = np.linalg.norm(original_data, ord=2)
    std = power / (10 ** (0.05 * a_SNR))
    noise = np.random.normal(0, std, len(original_data)) / 100
    noise_data = original_data + noise
    # print(power)
    # print(np.linalg.norm(noise, ord=2))
    return noise_data


def main():
    frequence = range(frequ_start, frequ_end + 1, int((frequ_end - frequ_start + 1) / (sample_length / wavelet_length)))
    amplitude = range(ampl_start, ampl_end + 1, int((ampl_end - ampl_start + 1) / (sample_length / wavelet_length)))
    wavelet_num = int(sample_length / wavelet_length)
    data_orig = generate_ricker(frequence, amplitude, wavelet_num)
    label = get_sim_label(data_orig)
    print(np.linalg.norm(label[:, 1], 0))
    for snr_n in SNR:
        result_sequence = []
        for i in range(sample_num):
            data_noise = add_nosie(data_orig, snr_n)
            result_sequence.append(data_noise)
            # if i==0 and snr_n==-10:
            #    test_data=data_noise
        with open(save_path + str(snr_n) + 'dB.csv', 'w', newline='') as save_file:
            wr = csv.writer(save_file, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            for data in result_sequence:
                wr.writerow(data)
    with open(save_path + 'original.csv', 'w', newline='') as origal_file:
        wr = csv.writer(origal_file, quoting=csv.QUOTE_ALL)
        wr.writerow(data_orig)
    with open(save_path + 'label.csv', 'w', newline='') as label_file:
        wr = csv.writer(label_file, quoting=csv.QUOTE_ALL)
        for tim in range(sample_length):
            wr.writerow(label[tim, :])

    with open(save_path + '-10dB.csv', 'r', newline='') as f:
        rd = csv.reader(f, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
        result_read = list(rd)
        ret = []
        for i in range(sample_length):
            ret.append(float(result_read[0][i]))
    squence_array = np.arange(1, sample_length + 1)
    # plt.plot(squence_array, ret-test_data)
    plt.plot(squence_array, ret)
    # plt.plot(squence_array, data_orig)

    # with open(save_path+'label.csv', 'r', newline='') as f:
    #    rd = csv.reader(f, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
    #    result_read = list(rd)
    #    ret=[]
    #    for i in range(sample_length):
    #        ret.append(result_read[i][1])
    # squence_array=np.arange(1,sample_length+1)
    # plt.plot(squence_array, ret)
    # plt.plot(squence_array, data_orig)
    # plt.show()


if __name__ == '__main__':
    main()
