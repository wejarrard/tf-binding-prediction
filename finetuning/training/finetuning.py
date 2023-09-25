from calendar import c
import torch
from transformers import (
    ElectraConfig,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import BCEWithLogitsLoss
import argparse
import os
import numpy as np
import logging
import copy
from genomicDataset import GenomicsDataset


def load_model(config_path, weights_path):
    # Load the configuration from json file
    config = ElectraConfig.from_json_file(config_path)
    config.num_labels = 1  # Adjust the number of output labels
    model = ElectraForSequenceClassification(config)

    # Load the model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_weights = torch.load(weights_path, map_location=device)

    # Create a new state dict with the weights for the sequence classification model
    sequence_classification_weights = {
        k: v for k, v in pretrained_weights.items() if k.startswith("electra.")
    }

    # Add the classifier weights manually
    sequence_classification_weights["classifier.dense.weight"] = pretrained_weights[
        "discriminator_predictions.dense.weight"
    ]
    sequence_classification_weights["classifier.dense.bias"] = pretrained_weights[
        "discriminator_predictions.dense.bias"
    ]
    sequence_classification_weights["classifier.out_proj.weight"] = pretrained_weights[
        "discriminator_predictions.dense_prediction.weight"
    ]
    sequence_classification_weights["classifier.out_proj.bias"] = pretrained_weights[
        "discriminator_predictions.dense_prediction.bias"
    ]

    # Load the sequence classification weights into the model
    model.load_state_dict(sequence_classification_weights)

    print(f"Using device: {device}")

    # Move the model to the correct device if
    model = model.to(device)

    # compile model if cuda is being used
    if torch.cuda.is_available():
        model = torch.compile(model)

    return model, device


def prepare_data(data_dir, train_frac=0.9, batch_size=16):
    # Create the directory-label dictionary
    dir_label_dict = {
        os.path.join(data_dir, "no_reads"): 0,
        os.path.join(data_dir, "with_reads"): 1,
    }

    # Create the Dataset
    # then create a full dataset without train or validation separation
    full_dataset = GenomicsDataset(dir_label_dict, dataset=None)

    # Determine the lengths of splits
    total_samples = len(full_dataset)
    train_len = int(train_frac * total_samples)
    valid_len = total_samples - train_len

    # Create the random splits
    train_data, valid_data = random_split(full_dataset, lengths=[train_len, valid_len])

    # Now, we create train and validation GenomicsDataset instances with only the file_paths and labels relevant to each split.
    train_dataset = GenomicsDataset(dir_label_dict, dataset="train")
    train_dataset.file_paths = [full_dataset.file_paths[i] for i in train_data.indices]
    train_dataset.labels = [full_dataset.labels[i] for i in train_data.indices]

    valid_dataset = GenomicsDataset(dir_label_dict, dataset="valid")
    valid_dataset.file_paths = [full_dataset.file_paths[i] for i in valid_data.indices]
    valid_dataset.labels = [full_dataset.labels[i] for i in valid_data.indices]

    # Calculate class frequencies and create oversample indices
    class_counts = np.bincount(train_dataset.labels)
    num_train_samples = len(train_dataset)
    train_labels = np.array(train_dataset.labels)

    # Using floor division to ensure integer results
    oversample_indices = np.repeat(
        np.arange(num_train_samples), num_train_samples // class_counts[train_labels]
    )

    # Shuffle the indices
    np.random.shuffle(oversample_indices)

    # Create a sampler and pass it to the DataLoader
    sampler = SubsetRandomSampler(oversample_indices)

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def train_model(
    model, device, train_loader, valid_loader, lr, epochs, output_data_dir, patience=5
):
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Specify loss function
    criterion = BCEWithLogitsLoss()

    # Best validation loss and early stopping setup
    best_valid_loss = float("inf")
    no_improvement_epochs = 0

    # Setup learning rate scheduler
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_data_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )

    print("Starting training loop...")

    # Training loop
    for epoch in range(epochs):
        # training phase
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, data in enumerate(train_loader):
            try:
                input_ids = data["input_ids"].to(device)
                position_ids = data["position_ids"].to(device)
                labels = data["labels"].to(device)
                reads = data["reads"].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids, position_ids=position_ids, reads=reads)

                logits = outputs.logits.squeeze()
                loss = criterion(logits, labels.float())

                preds = (torch.sigmoid(logits) > 0.5).long()
                correct = (preds == labels).sum().item()

                total_correct += correct
                total_samples += labels.size(0)

                loss.backward()
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                running_loss += loss.item()

                # Print the loss every 1000 loops
                if (i + 1) % 50 == 0:
                    print(f"Epoch: {epoch+1}, Step: {i+1}, Loss: {loss.item():.6f}, accuracy: {correct/labels.size(0):.2f}")

            except Exception as e:
                print(e)
                print("reads", data["reads"][0])
                print("input_ids", data["input_ids"][0])
                print("position_ids", data["position_ids"][0])
                print("labels", data["labels"][0])
                continue


        current_lr = optimizer.param_groups[0]["lr"]
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        logging.info(
            f"Training Epoch: {epoch+1}/{epochs}, Loss: {avg_train_loss:.6f}, Accuracy: {train_accuracy:.2f}, LR: {current_lr:.6f}"
        )

        # validation phase
        model.eval()
        running_valid_loss = 0.0
        total_valid_correct = 0
        total_valid_samples = 0

        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                input_ids = data["input_ids"].to(device)
                position_ids = data["position_ids"].to(device)
                labels = data["labels"].to(device)
                # chromosome = data['chromosome'].to(device)
                reads = data["reads"].to(device)

                outputs = model(input_ids, position_ids=position_ids, reads=reads)

                logits = outputs.logits.squeeze()
                loss = criterion(logits, labels.float())

                preds = (torch.sigmoid(logits) > 0.5).long()
                correct = (preds == labels).sum().item()

                total_valid_correct += correct
                total_valid_samples += labels.size(0)

                running_valid_loss += loss.item()

            avg_valid_loss = running_valid_loss / len(valid_loader)
            valid_accuracy = total_valid_correct / total_valid_samples

            logging.info(
                f"Validation Epoch: {epoch+1}/{epochs}, Loss: {avg_valid_loss:.6f}, Accuracy: {valid_accuracy:.2f}"
            )

        # Check for early stopping condition
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            no_improvement_epochs = 0
            best_model_wts = copy.deepcopy(
                model.state_dict()
            )  # Copy the best model weights

            # Save the best model weights immediately
            save_path = os.path.join(output_data_dir, "best_model")
            torch.save(best_model_wts, save_path)
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.environ["SM_CHANNEL_TRAINING"], "train"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument( "--config-path", type=str, default=os.path.join(os.environ["SM_CHANNEL_RESOURCES"], "discriminator.json"))
    parser.add_argument( "--weights-path", type=str, default=os.path.join(os.environ["SM_CHANNEL_RESOURCES"], "discriminator.pth"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    args = parser.parse_args()

    model, device = load_model(args.config_path, args.weights_path)
    train_loader, valid_loader = prepare_data(args.data_dir, batch_size=args.batch_size)
    train_model(
        model,
        device,
        train_loader,
        valid_loader,
        args.lr,
        args.epochs,
        args.output_data_dir,
        patience=args.patience,
    )
# export SM_CHANNEL_RESOURCES=/Users/wejarrard/projects/atacToChip/tf-binding-prediction/finetuning/training/resources
# export SM_CHANNEL_TRAINING=/Users/wejarrard/projects/atacToChip/tf-binding-prediction/finetuning/testing/synthetic_data/output/
# export SM_OUTPUT_DATA_DIR=./output/
# python finetuning.py \
# --epochs 20 \
# --lr 0.00005
