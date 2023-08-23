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

# os.environ['SM_CHANNEL_TRAINING'] = '/Users/wejarrard/projects/atacToChip/tf-binding-prediction/pretraining/preprocessing/output/'

class GenomicsDataset(Dataset):
    def __init__(self, dir_path=os.path.join(os.environ['SM_CHANNEL_TRAINING'], "train"), min_val=0, max_val=366.0038259577389, tokenizer=get_tokenizer(os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'tokenizer.json'))):
        self.tokenizer = tokenizer
        self.min_val = min_val
        self.max_val = max_val
        self.file_paths = []
        self.labels = []
        files = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if not pd.read_feather(os.path.join(dir_path, file)).empty]
        self.file_paths += files

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        df = pd.read_feather(file_path)

        # Seperate the data into the sequence, chromosome, and read counts
        # Introduce a small random cut
        cut = random.randint(0, 4)

        # Apply the cut
        dna_sequence = df["base"].str.cat(sep="")[cut:]
        read_counts = torch.tensor(df["reads"].values.astype(np.float64))[cut:]
        chromosome = df["chrom"].values[0]
        # Normalize the read counts
        read_counts = self.min_max_norm_global(read_counts, self.min_val, self.max_val)

        tokenized_sequence = self.tokenizer.encode(dna_sequence, return_tensors="pt")

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
        if len(tokenized_sequence[0]) > 512:  
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



        tokenized_sequence = tokenized_sequence.squeeze(1)
        position = position.squeeze(1)
        chromosome = chromosome.squeeze(1)
        final_read_counts = final_read_counts.squeeze(1)

        tokenized_sequence = tokenized_sequence.squeeze(0)
        position = position.squeeze(0)
        chromosome = chromosome.squeeze(0)
        final_read_counts = final_read_counts.squeeze(0)

        # Return the tokenized sequence, chromosome, read counts and position
        return {
            'input_ids': tokenized_sequence, 
            'position_ids': position, 
            'chromosome': chromosome, 
            'reads': final_read_counts
        }

    def min_max_norm_global(self, x, min_val, max_val, scale_max=1000):
        for i in range(len(x)):
            x[i] = scale_max * (x[i] - min_val) / (max_val - min_val)
            if x[i] < 0:
                x[i] = 0
            if x[i] > scale_max:
                x[i] = scale_max
        return x

    # def min_max_norm(self, x, scale_max=1000):
    #     min_val = torch.min(x)
    #     max_val = torch.max(x)
    #     normalized_x = scale_max * (x - min_val) / (max_val - min_val)
    #     return normalized_x.clamp(0, scale_max)


if __name__ == "__main__":

    # Create the dataset
    dataset = GenomicsDataset()

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate through the dataset
    for i, data in enumerate(dataloader):
        print(data)
        break
