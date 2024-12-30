import MakeModel as mm
from GetFiles import GetFiles
import sys

def train(speaker_name):
    print("Training "+ speaker_name+"'s model")
    gf = GetFiles(dataset_path="dataset")
    pandas_frame = gf.getTrainFiles(flag="train", train_speaker_folder=speaker_name)
    mm.makeModel(pandas_frame)
    print("Training finished.")


speaker_name = sys.argv[1]
train(speaker_name)