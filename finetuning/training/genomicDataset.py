import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
import pandas as pd
import numpy as np
import random

os.environ['SM_CHANNEL_TRAINING'] = '/Users/wejarrard/projects/atacToChip/finetuning/preprocessing/output'
os.environ['SM_OUTPUT_DATA_DIR'] = './output'


def get_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, max_len=512)
    return tokenizer
class GenomicsDataset(Dataset):
    def __init__(self, dir_label_dict, min_val, max_val, dataset="train", tokenizer=get_tokenizer(os.path.join(os.environ['SM_CHANNEL_TRAINING'], 'trimmed_tokenizer.json'))):
        self.tokenizer = tokenizer
        self.min_val = min_val
        self.max_val = max_val
        self.dataset = dataset
        self.file_paths = []
        self.labels = []
        for dir_path, label in dir_label_dict.items():
            files = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
            self.file_paths += files
            self.labels += [label] * len(files)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        df = pd.read_feather(file_path)

        # Seperate the data into the sequence, chromosome, and read counts
        if self.dataset == "train":
            # Introduce a small random cut
            cut = random.randint(0,4)

            # Apply the cut
            dna_sequence = df["base"].str.cat(sep="")[cut:]
            read_counts = torch.tensor(df["reads"].values.astype(np.float64))[cut:]
        else:
            dna_sequence = df["base"].str.cat(sep="")
            read_counts = torch.tensor(df["reads"].values.astype(np.float64))
        chromosome = df["chrom"].values[0]

        # Normalize the read counts
        # read_counts = self.min_max_norm(read_counts, self.min_val, self.max_val)

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
            if self.dataset == "train":
                start = random.randint(0, len(tokenized_sequence[0]) - 512)
                tokenized_sequence = tokenized_sequence[:, start:start+512]
                final_read_counts = final_read_counts[:, start:start+512]
            else:
                # Get the middle 512 tokens
                start = (len(tokenized_sequence[0]) - 512) // 2
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
            'reads': final_read_counts, 
            'labels': torch.tensor(label)
        }

    def min_max_norm(self, x, min_val, max_val):
        for i in range(len(x)):
            x[i] = (x[i] - min_val) / (max_val - min_val)
            if x[i] < 0:
                x[i] = 0
            if x[i] > 1:
                x[i] = 1
        return x

    # Iterate through the dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create the directory-label dictionary
    dir_label_dict = {
        "../preprocessing/output/train/atacseq_only": 0,
        # "../preprocessing/output/train/chipseq_only": 1,
        "../preprocessing/output/train/intersecting": 1,
    }

    min_val, max_val = 0, 366.0038259577389

    # Create the Dataset
    dataset = GenomicsDataset(dir_label_dict, min_val, max_val, dataset="valid")

    # Create the DataLoader
    torch.manual_seed(42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Define color map
    colors = {0: 'g', 1: 'b'}  # 0: Red, 1: Green

    # Iterate through the dataset
    for i, data in enumerate(dataloader):
        input_ids = data['input_ids']
        position_ids = data['position_ids']
        labels = data['labels']
        reads = data['reads']

        plt.figure(figsize=(12, 8))  # Adjust size of the plot as needed
        for j, read in enumerate(reads):
            # Determine color for each read based on label
            color = colors[labels[j].item()]  # Extract label as Python number and determine color
            if labels[j].item() == 0:
                plt.plot(read, label=f'No Peak', color=color)
            else:
                plt.plot(read, label=f'Peak', color=color)

        plt.xlabel('Index')  # Label x-axis
        plt.ylabel('Value')  # Label y-axis
        plt.title(f'Plot of Reads')  # Title of the plot
        plt.legend()  # Display legend
        plt.show()

        break

