import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

def rle2mask(rle, width, height):
    mask = np.zeros(width * height, dtype=np.uint8)
    if rle != '-1':
        rle = np.array([int(x) for x in rle.strip().split(' ')])
        rle = rle.reshape(-1, 2)
        start = 0
        for index, length in rle:
            start = start + index
            mask[start:start+length] = 1
            start = start + length
        mask = mask.reshape(width, height).T
    return mask

def convert_mask_to_png(data_path, csv_file):
    csv_path = os.path.join(data_path, csv_file)
    df = pd.read_csv(csv_path)
    for i in tqdm(range(len(df))):
        image_id = df.iloc[i]['ImageId']
        mask_rle = df.loc[i,'EncodedPixels']
        mask = rle2mask(mask_rle, 1024, 1024)
        mask = np.uint8(mask) * 255
        mask = cv2.resize(mask, (1024, 1024))
        output_file = os.path.join(data_path,"mask_png", image_id + ".png")
        cv2.imwrite(output_file, mask)


if __name__ == "__main__":
    data_path = "Chapter-9/input/siim_png/"
    csv_file = "train.csv"
    convert_mask_to_png(data_path, csv_file)