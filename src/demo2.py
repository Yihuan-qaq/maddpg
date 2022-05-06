import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
import ASR
import librosa.display


def mfcc_2(filename):
    y, sr = librosa.load(filename, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)  # n_mfcc为返回的mfcc数据维度
    mfccs = np.array(mfccs)  # 转成numpy类型
    # 返回 帧数*39 的mfccs参数
    return mfccs.T


def specgram(path):
    x, sr = soundfile.read(path)
    ft = librosa.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    ft_abs = np.abs(ft)
    D = librosa.amplitude_to_db(ft_abs, ref=np.max)
    plt.figure(dpi=300)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(ASR.asr_api(path, 'google'))
    plt.show()


# specgram(r'../example/si836.wav')
# specgram(r'temp.wav')

# mfcc = mfcc_2(r'../example/si836.wav')
# print()