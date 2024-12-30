
import os
import pickle
import numpy as np
from scipy.io.wavfile import read
import time
import sys
from ExtractFeature import ExtractFeature

def testPredict(audio_path):

    modelpath = "speakers_model/"

    ef = ExtractFeature

    gmm_files = [os.path.join(modelpath, fname) for fname in
                os.listdir(modelpath) if fname.endswith('.gmm')]

    speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]


    models   = [pickle.load(open(gmm_file,'rb')) for gmm_file in gmm_files]

    feature = ef.extract_features(audio_path)

    score_of_individual_comparision = np.zeros(len(models))
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(feature))
        score_of_individual_comparision[i] = scores.sum()

    winner = np.argmax(score_of_individual_comparision)

    speaker_detected = speakers[winner]

    return speaker_detected



def predict(file_name):

    speaker_predicted = testPredict(file_name)
    return speaker_predicted

if __name__ == "__main__":
    predict_dir_path = 'dataset/predict/'
    file_name = sys.argv[-1]
    predicted =  predict(predict_dir_path+file_name)
    print(predicted)


