import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
from scipy.io.wavfile import read
from sklearn.metrics import precision_recall_fscore_support as score



class ExtractFeature:

    @classmethod #classmethod is the decorator
    def __calculate_delta(self,array):

        rows, cols = array.shape
        deltas = np.zeros((rows, 20)) #20 columns
        N = 2
        for i in range(rows):
            index = []
            j = 1
            while j <= N:
                if i - j < 0:
                    first = 0
                else:
                    first = i - j
                if i + j > rows - 1:
                    second = rows - 1
                else:
                    second = i + j
                index.append((second, first))
                j += 1
            deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
        return deltas

    @classmethod
    def extract_features(self,audio_path):

        mfcc_feature = np.asarray

        rate, audio = read(audio_path)
        mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        delta = self.__calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined

