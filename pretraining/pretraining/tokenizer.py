import pandas as pd
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import os
import random


def get_training_corpus(data_dir, sample_size):
    total_files = sum([len(files) for r, d, files in os.walk(f"{data_dir}")])
    sampled_indices = random.sample(range(1, total_files + 1), sample_size)

    for i in sampled_indices:
        f = f"{data_dir}/{i}.orc"
        # Check if the file size is greater than 0
        if os.path.getsize(f) > 0:
            df = pd.read_orc(f)
            dna_sequence = df["reference_base"].str.cat(sep="")
            yield dna_sequence
        else:
            print(f"Skipping file {f} due to small file size.")


def train_tokenizer(data_dir="/home/ec2-user/atacToChip/output", vocab_size=5000, output_dir="./", tokenizer_name="tokenizer"):
    tokenizer = Tokenizer(models.BPE())
    # Pre-tokenizer
    print("Pre-tokenizing")
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Trainers
    print("Training")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[
                                  "[PAD]", "[MASK]"], show_progress=True)

    sample_size = 100000
    tokenizer.train_from_iterator(
        get_training_corpus(data_dir, sample_size=sample_size), trainer=trainer, length=sample_size)

    # Post-processing
    print("Post-processing")
    tokenizer.decoder = decoders.ByteLevel()

    # Save
    print("Saving")
    tokenizer.save(f"{output_dir}/{tokenizer_name}.json")


def get_tokenizer(tokenizer_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, max_len=512)
    return tokenizer


def main():
    train_tokenizer()


if __name__ == "__main__":
    main()
