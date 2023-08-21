# ONLY NEEDS TO BE RUN ONCE
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np


class MinMaxDataset(Dataset):
    def __init__(self, data_dir="../data/pretraining/cleaned"):

        self.data_dir = data_dir

    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(f"{self.data_dir}")])

    def __getitem__(self, idx):

        # get the file at idx (subtract len(all files before this in the list))
        f = f"{self.data_dir}/sample_{idx}.orc"
        df = pd.read_orc(f)

        # Seperate the data into the sequence, chromosome, and read counts
        read_counts = torch.tensor(df["total_reads"].values.astype(np.float64))

        # Return the tokenized sequence, chromosome, read counts and position
        return read_counts


# Get mean and std
def getMinMax(loader):
    """
    The function calculates the Max and Min of the reads
    """
    max_val = 0
    min_val = 0

    for data in tqdm(loader):
        max_val = max(max_val, torch.max(data[0]).item())
        min_val = min(min_val, torch.min(data[0]).item())

    return min_val, max_val


def get_min_max(num_workers, data_dir="../data/pretraining/cleaned"):

    batch_size = 1

    # Load dataset
    dataset = MinMaxDataset(data_dir)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    min_val, max_val = getMinMax(dataloader)

    return min_val, max_val


if __name__ == "__main__":

    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    # Get min and max
    min_val, max_val = get_min_max(args.num_workers)

    # save min and max to file
    print(min_val, max_val)
