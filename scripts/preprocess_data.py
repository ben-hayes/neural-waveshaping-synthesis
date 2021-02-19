import os

from nts.data.utils.preprocess_audio import preprocess_audio

if __name__ == "__main__":
    files = [
        f for f in os.listdir("/import/c4dm-datasets/URMP/Dataset/08_Spring_fl_vn/") 
        if ".wav" in f
    ]
    preprocess_audio(files)