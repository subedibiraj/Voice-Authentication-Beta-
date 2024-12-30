import pickle
from sklearn.mixture import GaussianMixture
from ExtractFeature import ExtractFeature
import numpy as np

def makeModel(pipelined_data_frame):

    ef = ExtractFeature

    stacked_feature = np.asarray(())

    first_audio_location_in_frame = pipelined_data_frame["audio_path"].iloc[0]
    stacked_feature = ef.extract_features(first_audio_location_in_frame)

    for index, row in pipelined_data_frame.iterrows():
        if index != 0:
            ef = ExtractFeature
            currently_fetched_feature = ef.extract_features(row['audio_path'])
            stacked_feature = np.vstack((stacked_feature, currently_fetched_feature))


    model_save_path = "speakers_model/"
    model_name = pipelined_data_frame["target_speaker"].iloc[0]+".gmm"

    gmm = GaussianMixture(n_components=16, covariance_type='diag', max_iter=500, n_init=3, verbose=1)
    gmm.fit(stacked_feature)

    pickle.dump(gmm, open(model_save_path + model_name, 'wb'))
