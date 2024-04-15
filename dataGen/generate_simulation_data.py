import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 设置参数
row_length = 1600       # 每行的数据点数
num_wavelets_per_row = 8  # 每行包含的波形数
wavelet_length = int(row_length / num_wavelets_per_row)  # 每个波形的长度
frequ_start = 20
frequ_end = 300
ampl_start = 200
ampl_end = 300
SNR = -5  # 调整信噪比为20dB，更大的SNR值表示较小的噪声
save_path = './data/'

# 创建保存路径
if not os.path.exists(save_path):
    os.makedirs(save_path)

def ricker_wavelet(f, a, length):
    dt = 0.001
    num_points = length  # 直接使用计算得到的整数波形长度
    wave_time = np.linspace(-length / 2 * dt, (length / 2 - 1) * dt, num_points)
    ricker = (1. - 2. * (np.pi ** 2) * (f ** 2) * (wave_time ** 2)) * np.exp(
        -(np.pi ** 2) * (f ** 2) * (wave_time ** 2)) * a
    return ricker

def add_noise(original_data, snr):
    signal_power = np.mean(original_data ** 2)
    signal_db = 10 * np.log10(signal_power)
    noise_db = signal_db - snr
    noise_power = 10 ** (noise_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), len(original_data))
    noisy_data = original_data + noise
    return noisy_data

# 生成频率和振幅序列
frequence = np.linspace(frequ_start, frequ_end, num_wavelets_per_row)
amplitude = np.linspace(ampl_start, ampl_end, num_wavelets_per_row)

# 创建CSV文件
with open(os.path.join(save_path, f'{SNR}dB_data.csv'), 'w', newline='') as file:
    writer = csv.writer(file)

    # 生成多行数据
    for _ in range(10):  # 示例：生成10行数据
        data_row = np.array([])
        for f, a in zip(frequence, amplitude):
            wavelet = ricker_wavelet(f, a, wavelet_length)
            data_row = np.append(data_row, wavelet)
        noisy_data_row = add_noise(data_row, SNR)
        writer.writerow(noisy_data_row)

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(data_row, label='Original Wavelet')
plt.plot(noisy_data_row, label='Noisy Wavelet', linestyle='--')
plt.title(f'Ricker Wavelet with Noise : SNR = {SNR} dB')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.savefig(f'./{SNR}dB_wavelet.png')
plt.show()

