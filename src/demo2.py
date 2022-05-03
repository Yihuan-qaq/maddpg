import librosa
import numpy as np


def mfcc_2(filename):
    y, sr = librosa.load(filename, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39)  # n_mfcc为返回的mfcc数据维度
    mfccs = np.array(mfccs)#转成numpy类型
    # 返回 帧数*39 的mfccs参数
    return mfccs.T


mfcc = mfcc_2(r'../example/si836.wav')
print()