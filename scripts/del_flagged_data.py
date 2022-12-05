import os

import pandas as pd

if __name__ == "__main__":
    path = "/home/ychau001/Desktop/FYP/output/off-class-labels.csv"
    df = pd.read_csv(path)
    file_paths = df["path"].values

    for file_path in file_paths:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"{file_path} removed")
