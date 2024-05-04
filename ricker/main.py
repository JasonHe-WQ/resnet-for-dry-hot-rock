import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker

# 定义参数
sampleLen = 1000  # 波形采样点数
fs = 1000  # 采样频率 Hz
fm = 75  # 中心频率 Hz
ap = 0.7  # 子波峰值位置
no = -25  # 加入白噪声 dB
timeDic = np.linspace(0, (sampleLen - 1) / fs, sampleLen) * 1000  # 时间向量转化为毫秒

# 生成 Ricker 小波
width = fm * (1 / fs) * np.sqrt(2 * np.pi)
OriData = ricker(sampleLen, width)

# 添加白噪声
noise = np.random.normal(0, 10 ** (no / 20), sampleLen)
noisedData = OriData + noise

# FFT 变换和功率谱计算
N = len(OriData)
f = np.fft.fftfreq(N, d=1/fs)
waveData2 = np.fft.fft(OriData)
Opower = np.abs(waveData2)**2

noisedData2 = np.fft.fft(noisedData)
NDpower = np.abs(noisedData2)**2

noise2 = np.fft.fft(noise)
Npower = np.abs(noise2)**2

# 绘图
plt.figure(figsize=(12, 10))
titleStr = f'fs = {fs}, fm = {fm}, no = {no}, sampleLen = {sampleLen}'

# 原始 Ricker 小波
plt.subplot(3, 2, 1)
plt.plot(timeDic, OriData)
plt.title('Ricker 小波')
plt.xlabel('时间 /ms')
plt.ylabel('振幅 /mV')

# 原始 Ricker 小波频谱
plt.subplot(3, 2, 2)
plt.plot(f[:N//2], Opower[:N//2])
plt.title('Ricker 小波频谱')
plt.xlabel('频率 /Hz')
plt.ylabel('功率')

# 带噪声的 Ricker 小波
plt.subplot(3, 2, 3)
plt.plot(timeDic, noisedData)
plt.title('带噪声的 Ricker 小波')
plt.xlabel('时间 /ms')
plt.ylabel('振幅 /mV')

# 带噪声的 Ricker 小波频谱
plt.subplot(3, 2, 4)
plt.plot(f[:N//2], NDpower[:N//2])
plt.title('带噪声的 Ricker 小波频谱')
plt.xlabel('频率 /Hz')
plt.ylabel('功率')

# 噪声
plt.subplot(3, 2, 5)
plt.plot(timeDic, noise)
plt.title('噪声')
plt.xlabel('时间 /ms')
plt.ylabel('振幅 /mV')

# 噪声频谱
plt.subplot(3, 2, 6)
plt.plot(f[:N//2], Npower[:N//2])
plt.title('噪声频谱')
plt.xlabel('频率 /Hz')
plt.ylabel('功率')

plt.tight_layout()
plt.show()
