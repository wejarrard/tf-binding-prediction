# TODO WE NEED TO MAKE THE VALUDATION AND TESTING INCLUDE THE PEAK

import bisect
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tokenizer import get_tokenizer
import os
import glob
from torchvision.transforms import Compose
from torchtext import transforms
from torch.utils.data import Dataset, Subset, random_split, DataLoader
import random
TOKENIZERS_PARALLELISM = False


class GenomeDataset(Dataset):
    def __init__(self, min_val, max_val, tokenizer=get_tokenizer("./tokenizer.json"), data_dir="/home/ec2-user/output"):

        self.min_val = min_val
        self.max_val = max_val
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(f"{self.data_dir}")])

    def __getitem__(self, idx): 
        try: 
            f = f"{self.data_dir}/{idx}.orc"
            df = pd.read_orc(f)
        
        except FileNotFoundError:
            # might repeat a file
            f = f"{self.data_dir}/{idx + 1}.orc"
            df = pd.read_orc(f)


        # Seperate the data into the sequence, chromosome, and read counts
        dna_sequence = df["reference_base"].str.cat(sep="")
        read_counts = torch.tensor(df["total_reads"].values.astype(np.float64))
        chromosome = df["chromosome"].values[0]

        # Normalize the read counts
        read_counts = self.min_max_norm(
            read_counts, self.min_val, self.max_val)

        # Tokenize the sequence
        tokenized_sequence = self.tokenizer.encode(
            dna_sequence, return_tensors="pt")

        cur_pos = 0
 
        final_read_counts = torch.zeros(len(tokenized_sequence[0]))

        for i in range(len(tokenized_sequence[0])):

            # Get the token
            token = tokenized_sequence[0][i]

            # Get length of token
            token_length = len(self.tokenizer.decode(token))

            # Get the read counts for the token
            token_read_counts = read_counts[cur_pos:cur_pos+token_length]

            # Get the average read count for the token
            token_read_count = token_read_counts.mean()

            # Replace the read counts for the token with the average read count
            final_read_counts[i] = token_read_count

            # Update the current position
            cur_pos += token_length

        # Add batch dimension
        final_read_counts = torch.unsqueeze(final_read_counts, 0)

        # Get the tokenized sequence, chromosome, read counts and position for the 512 tokens if the sequence is longer than 512
        if len(tokenized_sequence[0]) >= 512:
            # Get random number between 0 and len(tokenized_sequence[0]) - 512
            start = random.randint(0, len(tokenized_sequence[0]) - 512)
            tokenized_sequence = tokenized_sequence[:, start:start+512]
            final_read_counts = final_read_counts[:, start:start+512]

        elif len(tokenized_sequence[0]) < 512:

            # Pad the tokenized sequence, masked tokens, read counts and position with 0s
            tokenized_sequence = torch.nn.functional.pad(
                tokenized_sequence, (0, 512 - len(tokenized_sequence[0])))
            final_read_counts = torch.nn.functional.pad(
                final_read_counts, (0, 512 - len(final_read_counts[0])))

        # create position tensor
        position = torch.arange(0, 512, 1)

        # Create chromosome tensor
        chromosome = torch.full((1, 512), chromosome)

        # Add batch dimension
        position = torch.unsqueeze(position, 0)

        # Return the tokenized sequence, chromosome, read counts and position
        return position, chromosome, tokenized_sequence, final_read_counts

    def min_max_norm(self, x, min_val, max_val):
        for i in range(len(x)):
            x[i] = (x[i] - min_val) / (max_val - min_val)
            if x[i] < 0:
                x[i] = 0
            if x[i] > 1:
                x[i] = 1
        return x


if __name__ == "__main__":

    # Create the dataset
    dataset = GenomeDataset(0, 100)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate through the dataset
    for i, data in enumerate(dataloader):
        print(data)
        break
