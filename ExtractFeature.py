import os
import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc
from scipy.io.wavfile import read
import wave
import subprocess

class ExtractFeature:

    @classmethod
    def __calculate_delta(self, mfcc_features):
        """
        Calculate delta features for MFCC (first derivative of MFCC coefficients).
        Args:
            mfcc_features: numpy array of MFCC features (2D array: time frames x MFCC coefficients)
        Returns:
            delta: numpy array of delta features
        """
        num_frames, num_coeffs = mfcc_features.shape
        delta = np.zeros_like(mfcc_features)  # Create a zero array for delta features
        
        # Compute delta features for each frame
        for t in range(num_frames):
            for n in range(num_coeffs):
                prev_frame = max(0, t - 1)
                next_frame = min(num_frames - 1, t + 1)
                
                # Delta computation: (frame(t+1) - frame(t-1)) / 2
                delta[t, n] = (mfcc_features[next_frame, n] - mfcc_features[prev_frame, n]) / 2
        
        return delta

    @classmethod
    def convert_to_wav(self, input_path, output_path):
        """ Convert file to a proper WAV format using ffmpeg """
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "1", "-sample_fmt", "s16", output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return output_path
        except Exception as e:
            print("Error converting to WAV:", e)
            return None

    @classmethod
    def is_valid_wav(self, file_path):
        """ Check if the file is a valid WAV format """
        try:
            with wave.open(file_path, "rb") as wf:
                return True
        except wave.Error:
            return False

    @classmethod
    def extract_features(self, audio_path):
        """ Validate and extract features from a WAV file """
        
        # Ensure the file is a valid WAV file
        if not self.is_valid_wav(audio_path):
            print("Invalid WAV file detected. Attempting to convert...")
            fixed_path = audio_path.replace(".wav", "_fixed.wav")
            audio_path = self.convert_to_wav(audio_path, fixed_path)

            if not self.is_valid_wav(audio_path):
                raise ValueError(f"Failed to process {audio_path}. File is not a valid WAV format.")

        # Read audio and extract features
        try:
            rate, audio = read(audio_path)
            mfcc_features = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
            
            # Normalize the MFCC features to ensure consistency
            mfcc_features = preprocessing.scale(mfcc_features)  
            
            # Calculate delta features (first derivative of MFCC features)
            delta_features = self.__calculate_delta(mfcc_features)
            
            # Combine the MFCC and delta features
            combined_features = np.hstack((mfcc_features, delta_features))
            return combined_features
        
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
