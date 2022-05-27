import copy
import soundfile
import librosa
import numpy as np
import ASR
from jiwer import wer
import time


class Env_F(object):

    def __init__(self, freq_band, source_path_wav, source_path_phn):
        """

        :param freq_band:[[start1, end1], [start2, end2],]
        """
        self.freq_band = freq_band
        self.source_path_wav = source_path_wav
        self.source_path_phn = source_path_phn

        flag = 0
        if flag == 0:
            self.source_result = ASR.asr_api(self.source_path_wav, 'deepspeech')
            self.temp_source_result = self.source_result
            flag = 1
            print("----source result :{}".format(self.source_result))
        else:
            self.source_result = self.temp_source_result

        '''Init FFT param'''
        self.n_fft = 512
        self.win_length = self.n_fft
        self.hope_length = self.win_length // 4

        self.FLAG_EMPTY = []  # 记录没有出现的音素位置
        self.FLAG_VALUE = -100  # 标记音素没有出现位置的值
        self.bound_high = 2
        self.bound_low = 0
        self.STATE_HIGH_BOUND = 2
        self.STATE_LOW_BOUND = 0
        self.DONE_REWARD = 45

        self.process_index = self.find_band()
        self.s_dim = len(self.process_index)
        self.a_dim = self.s_dim

    def process_audio(self, source_path):
        x, sr = soundfile.read(source_path)
        ft = librosa.stft(x, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hope_length)
        pha = np.exp(1j * np.angle(ft))
        ft_abs = np.abs(ft)
        return ft_abs, pha, sr

    def find_band(self):
        start_index = []
        end_index = []
        process_index_dict = []
        for i_ in range(len(self.freq_band)):
            start_index.append(self.freq_band[i_][0])
            end_index.append(self.freq_band[i_][1])
        temp_start = np.zeros(len(start_index), dtype=int)
        temp_end = np.zeros(len(end_index), dtype=int)
        for i in range(len(start_index)):
            temp_start[i] = int(start_index[i] // (8000 / (self.n_fft // 2 + 1)))
            temp_end[i] = int(end_index[i] // (8000 / (self.n_fft // 2 + 1)) + 1)
            process_index_dict.append([temp_start[i], temp_end[i]])
        return process_index_dict

    def low_filter(self, ft_matrix, threshold):
        ft_filter = np.zeros(shape=(len(ft_matrix), len(ft_matrix[0])), dtype=float)
        for i in range(len(ft_matrix)):
            for j in range(len(ft_matrix[0])):
                if ft_matrix[i][j] < threshold:
                    ft_filter[i][j] = 0
                else:
                    ft_filter[i][j] = ft_matrix[i][j]
        # ft_matrix[ft_matrix < threshold] = 0
        return ft_filter

    # def normalize(self, data):
    #     normalized = data.ravel() * 1.0 / np.amax(np.abs(data.ravel()))
    #     magnitude = np.abs(normalized)
    #     return magnitude

    def calculate_MSE(self, audio1, audio2):
        # Normalize
        # n_audio1 = self.normalize(audio1)
        # n_audio2 = self.normalize(audio2)

        audio_len = min(len(audio1), len(audio2))
        n_audio1 = audio1[:audio_len]
        n_audio2 = audio2[:audio_len]

        # Diff
        diff = n_audio1 - n_audio2
        abs_diff = np.abs(diff)
        overall_change = sum(abs_diff)
        average_change = overall_change / len(audio1)
        return average_change

    def calculate_reward(self, source_result, processed_result, source_path, phn_hat, threshold):
        # """ 对那些超出阈值范围的状态进行大力度惩罚 """
        # s = s[0]
        # a = a[0]
        threshold_reward = np.zeros(shape=(len(threshold)), dtype=float)
        # for i in range(len(threshold)):
        #     if threshold[i] == self.FLAG_VALUE:
        #         threshold_reward[i] = 0
        #     elif s[i] <= self.STATE_LOW_BOUND:
        #         if a[i] <= 0:
        #             threshold_reward[i] = np.abs(threshold[i]) * 10
        #         else:
        #             threshold_reward[i] = np.abs(threshold[i]) * 2
        #     elif s[i] >= self.STATE_HIGH_BOUND and a[i] >= 0:
        #         threshold_reward[i] = threshold[i] * 5
        #     else:
        #         threshold_reward[i] = threshold[i]

        # threshold_reward = threshold
        for i in range(len(threshold)):
            if threshold[i] == self.FLAG_VALUE:
                threshold_reward[i] = 0
            else:
                threshold_reward[i] = threshold[i]
        """ 计算MSE分数"""
        if source_result == "RequestError":
            r = 0
            return r
        else:
            wer_value = wer(source_result, processed_result)

        global_ft_abs, source_pha, sr = self.process_audio(source_path)
        global_ft_abs_filter = self.low_filter(global_ft_abs, max(threshold_reward))
        global_ft = global_ft_abs_filter * source_pha
        global_ft_hat = librosa.istft(global_ft, hop_length=self.hope_length, win_length=self.win_length)

        source_hat, _ = soundfile.read(source_path)

        MSE1 = self.calculate_MSE(source_hat, phn_hat)
        MSE2 = self.calculate_MSE(source_hat, global_ft_hat)

        if MSE2 == 0.0:
            return -100
        else:
            MSE_ratio = MSE1 / MSE2

        """计算总的reward"""
        # total_threshold =
        mean_threshold = np.sum(threshold_reward) / (len(threshold_reward) - len(self.FLAG_EMPTY))
        # mean_threshold = np.mean(threshold_reward)
        r = wer_value * 100 - MSE_ratio * 70 - mean_threshold * 60
        return r

    def step(self, a):
        """
        :input: 动作a
        计算当前状态s加上动作a后的下一状态s_;
        用这个s_进行一次滤波，并转录结果，判断是否攻击成功;
        攻击成功：奖励r=1，结束标志done=True；
        攻击成功：奖励r=0，结束标志done=Flase；
        :return: s_,r,done;
        """
        done = False
        r = 0
        # s_ = list(map(lambda x: x[0] + x[1], zip(s, a)))
        # s_ = s + a
        # threshold = s_[0]
        # threshold = s_
        s_ = copy.deepcopy(a)
        """限制阈值范围"""
        for i_ in range(len(s_)):
            if s_[i_] > self.STATE_HIGH_BOUND:
                s_[i_] = self.STATE_HIGH_BOUND
            elif s_[i_] < self.STATE_LOW_BOUND:
                s_[i_] = self.STATE_LOW_BOUND

        for i_ in range(len(self.FLAG_EMPTY)):
            s_[self.FLAG_EMPTY[i_]] = self.FLAG_VALUE
        threshold = s_
        """处理音频"""
        ft_abs, pha, sr = self.process_audio(self.source_path_wav)
        process_index = self.process_index
        '''滤波'''
        for index, i in zip(process_index, range(len(process_index))):
            start_index = index[0]
            end_index = index[1]
            ft_abs[start_index - 1:end_index - 1, :] = \
                self.low_filter(ft_abs[start_index - 1:end_index - 1, :], threshold[i])

        '''重建滤波后的音频'''
        ft = ft_abs * pha
        y_hat = librosa.istft(ft, hop_length=self.hope_length, win_length=self.win_length)
        temp_wirte_path = r'temp.wav'
        soundfile.write(temp_wirte_path, y_hat, samplerate=sr)
        t0 = time.time()
        trans_result = ASR.asr_api(temp_wirte_path, 'deepspeech')
        t1 = time.time()
        r = self.calculate_reward(self.source_result, trans_result, self.source_path_wav, phn_hat=y_hat,
                                  threshold=threshold)
        if r > self.DONE_REWARD:
            done = True
        return s_, r, done, t1 - t0

    def reset(self):
        """
        初始化状态
        :return: 状态s
        """
        # Max = 0.5  # 随机生成小数的最大值
        # s = list(np.random.rand(self.s_dim) * Max)
        s = np.zeros(shape=self.s_dim)
        return s

    def action_space_high(self):
        return self.bound_high

    def action_space_low(self):
        return self.bound_low

    def get_s_dim(self):
        return self.s_dim

    def get_a_dim(self):
        return self.a_dim

    def get_FLAG_EMPTY(self):
        return self.FLAG_EMPTY