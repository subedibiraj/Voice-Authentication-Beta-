
import glob,os
import pandas as pd


class GetFiles:

    def __init__(self,dataset_path):
        self.dataset_path = dataset_path


    def getTrainFiles(self,flag,train_speaker_folder):

        flag = "train"

        data_frame_row = []

        sub_files = os.listdir(self.dataset_path+"/"+flag+"/"+train_speaker_folder)
        for files in sub_files:
            path_to_audio =  self.dataset_path+"/"+flag+"/"+train_speaker_folder+"/"+files
            data_frame_row.append([path_to_audio,train_speaker_folder])
        data_frame = pd.DataFrame(data_frame_row,columns=['audio_path', 'target_speaker'])

        return data_frame


    def getTestFiles(self):

        data_frame_row = []

        data_frame = pd.DataFrame()

        flag = "test"

        speaker_audio_folder = os.listdir(self.dataset_path+"/"+flag)


        for folders in speaker_audio_folder:

            audio_files = os.listdir(self.dataset_path+"/"+flag+"/"+folders)

            for files in audio_files:
                path_to_audio =  self.dataset_path+"/"+flag+"/"+folders+"/"+files
                data_frame_row.append([path_to_audio,folders])

            data_frame = pd.DataFrame(data_frame_row,columns=['audio_path','actual'])

        return data_frame
