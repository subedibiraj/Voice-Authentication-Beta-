import os
import pickle
import numpy as np
import json
import sys
from ExtractFeature import ExtractFeature

def testPredict(audio_path):
    modelpath = "speakers_model/"
    ef = ExtractFeature()  # Corrected: Now properly creating an instance

    # Load all GMM models
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
    speakers = [os.path.splitext(os.path.basename(fname))[0] for fname in gmm_files]
    models = [pickle.load(open(gmm_file, 'rb')) for gmm_file in gmm_files]

    # Extract features from the input audio file
    feature = ef.extract_features(audio_path)

    # Debugging: Print feature shape and sample values
    # print(f"Feature shape: {feature.shape}")
    # print(f"Feature sample (first 5 values): {feature[:5]}")

    score_of_individual_comparison = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(feature))
        score_of_individual_comparison[i] = scores.sum()

    # Normalize scores to prevent overconfidence
    score_range = np.max(score_of_individual_comparison) - np.min(score_of_individual_comparison)
    if score_range == 0:  # Avoid division by zero
        probabilities = np.ones(len(score_of_individual_comparison)) / len(score_of_individual_comparison)
    else:
        scores_exp = np.exp(score_of_individual_comparison - np.max(score_of_individual_comparison))
        probabilities = scores_exp / np.sum(scores_exp)

    winner = np.argmax(probabilities)
    confidence = probabilities[winner] * 100  # Convert to percentage

    speaker_detected = speakers[winner]

    # Debugging: Print results for verification
    # print(f"Scores: {score_of_individual_comparison}")
    # print(f"Probabilities: {probabilities}")
    # print(f"Predicted Speaker: {speaker_detected} with Confidence: {confidence:.2f}%")

    # Remove the processed file
    os.remove(audio_path)

    return speaker_detected, confidence

def predict(file_name, threshold=80):
    speaker_predicted, confidence = testPredict(file_name)

    result = {
        "predicted_user": speaker_predicted,
        "confidence": round(confidence, 2),
        "match": "true" if confidence >= threshold else "false"
    }

    return json.dumps(result)

if __name__ == "__main__":
    predict_dir_path = 'dataset/predict/'
    file_name = sys.argv[-1]
    
    # Ensure the file path is correctly formatted
    file_path = os.path.join(predict_dir_path, os.path.basename(file_name))
    
    result_json = predict(file_path)
    print(result_json)
