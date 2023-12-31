import os
import sys
import json

# To disable Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# set the environment variables
os.environ['SM_CHANNEL_TRAINING'] = '/Users/wejarrard/projects/atacToChip/tf-binding-prediction/pretraining/preprocessing/output'
os.environ['SM_OUTPUT_DATA_DIR'] = './output/'
# make output directory
os.makedirs(os.environ['SM_OUTPUT_DATA_DIR'], exist_ok=True)

import random
import logging
from time import time
from dataclasses import dataclass

import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from transformers import AutoConfig, ElectraForMaskedLM, ElectraForPreTraining

from electra import Electra
from dataset import GenomicsDataset
from tokenizer import get_tokenizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

########################################################################################################
# args


@dataclass
class Args:
    data_n_tensors_per_file: int = 2048
    data_max_seq_length: int = 128

    distributed_world_size: int = torch.cuda.device_count()
    gpu: int = 0
    gpu_enabled: bool = distributed_world_size > 0
    gpu_deterministic: bool = gpu_enabled
    gpu_mixed_precision: bool = False
    distributed_port: int = 8888
    distributed_enabled: bool = distributed_world_size > 1

    model_generator: str = 'generator.json'
    model_discriminator: str = 'discriminator.json'
    model_mask_prob: float = 0.15

    opt_lr: float = 3e-4  # Tried 5e-4, 4e-4
    opt_batch_size: int = 128 // distributed_world_size if distributed_enabled else 64
    opt_num_workers = torch.multiprocessing.cpu_count()

    opt_warmup_steps: int = 5000  # Tried 10_000, 5000, 2500
    opt_num_training_steps: int = 200_000

    step_log: int = 1
    step_ckpt: int = 100
    
    max_patience: int = 500


########################################################################################################
# train

def train(rank, args):

    #######################
    # distributed
    print(f'Args: {args}')

    if args.distributed_enabled:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.distributed_world_size,
            rank=rank)
    if args.gpu_enabled:
        device = torch.device('cuda:{}'.format(rank))
    else:
        device = torch.device('cpu')

    is_master = True if not args.distributed_enabled else args.distributed_enabled and rank == 0

    #######################
    # preamble

    set_gpus(rank)
    set_seed(rank)
    set_cuda(deterministic=args.gpu_deterministic)

    output_dir = f'{args.output_dir}/{rank}'
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(filename=f'{output_dir}/output.log', console=is_master)

    #######################
    # dataset
    

    # Define tokenizer and dataset
    tokenizer = get_tokenizer(f"{os.environ['SM_CHANNEL_TRAINING']}/tokenizer.json")
    vocab_size = len(tokenizer)


    print("Loading dataset...")
    dataset = GenomicsDataset()

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.opt_batch_size,
                            num_workers=args.opt_num_workers, drop_last=True)

    #######################
    # model

    def to_distributed_model(model):
        return model if not args.distributed_enabled else torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    def tie_weights(generator, discriminator):
        generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
        generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
        generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings
        generator.electra.embeddings.reads_weights = discriminator.electra.embeddings.reads_weights
        generator.electra.embeddings.chromosome_embeddings = discriminator.electra.embeddings.chromosome_embeddings

    class LogitsAdapter(torch.nn.Module):
        def __init__(self, adaptee):
            super().__init__()
            self.adaptee = adaptee

        def forward(self, *args, **kwargs):
            return self.adaptee(*args, **kwargs)[0]
        
    print("Loading model...")

    generator = ElectraForMaskedLM(
        AutoConfig.from_pretrained(args.model_generator))
    discriminator = ElectraForPreTraining(
        AutoConfig.from_pretrained(args.model_discriminator))

    tie_weights(generator, discriminator)

    model = to_distributed_model(Electra(
        LogitsAdapter(generator),
        LogitsAdapter(discriminator),
        num_tokens=vocab_size,
        mask_token_id=1,
        pad_token_id=0,
        mask_prob=args.model_mask_prob,
        random_token_prob=0.0).to(device))

    # model = torch.compile(model)

    #######################
    # optimizer

    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            learning_rate = max(
                0.0, 1. - (float(current_step) / float(num_training_steps)))
            learning_rate *= min(1.0, float(current_step) /
                                 float(num_warmup_steps))
            return learning_rate
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(
        model.named_parameters(), weight_decay=0.1), lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.opt_warmup_steps, num_training_steps=args.opt_num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu_mixed_precision)

    #######################
    # load checkpoint if exists
    checkpoint_dir = f'{args.output_dir}/ckpt'
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    start_step = 0

    print("Starting training...")

    if latest_checkpoint is not None:
        logger.info(f'Loading checkpoint: {latest_checkpoint}')
        start_step = int(latest_checkpoint.split('/')[-1])
        generator.load_state_dict(torch.load(
            f'{latest_checkpoint}/generator.pth'))
        discriminator.load_state_dict(torch.load(
            f'{latest_checkpoint}/discriminator.pth'))
        
        # Retie the weights after loading the state dicts
        tie_weights(generator, discriminator)
        
        # Recreate the model with the updated generator and discriminator
        model = to_distributed_model(Electra(
            LogitsAdapter(generator),
            LogitsAdapter(discriminator),
            num_tokens=vocab_size,
            mask_token_id=1,
            pad_token_id=0,
            mask_prob=args.model_mask_prob,
            random_token_prob=0.0).to(device))
        # model = torch.compile(model)
        
        optimizer.load_state_dict(torch.load(
            f'{latest_checkpoint}/optimizer.pth'))
        scheduler.load_state_dict(torch.load(
            f'{latest_checkpoint}/scheduler.pth'))


    #######################
    # train

    t, steps_s, eta_m = time(), 0., 0

    args.opt_num_training_steps = len(dataloader)

    best_loss = float('inf')
    patience_counter = 0

    # log number of training steps
    logging.info(
        f'Number of training steps left: {args.opt_num_training_steps - start_step}')

    for step in range(start_step, args.opt_num_training_steps):

        data = next(iter(dataloader))

        input_ids = data['input_ids'].to(device)
        position_ids = data['position_ids'].to(device)
        chromosome = data['chromosome'].to(device)
        reads = data['reads'].to(device)


        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.gpu_mixed_precision):
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(
                input_ids, position_ids=position_ids, chromosome=chromosome, reads=reads)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        metrics = {
            'step': (step, '{:8d}'),
            'loss': (loss.item(), '{:8.5f}'),
            'loss_mlm': (loss_mlm.item(), '{:8.5f}'),
            'loss_disc': (loss_disc.item(), '{:8.5f}'),
            'acc_gen': (acc_gen.item(), '{:5.3f}'),
            'acc_disc': (acc_disc.item(), '{:5.3f}'),
            'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
            'steps': (steps_s, '{:4.1f}/s'),
            'eta': (eta_m, '{:4d}m'),
        }

        if step % args.step_log == 0:
            sep = ' ' * 2
            logger.info(
                sep.join([f'{k}: {v[1].format(v[0])}' for (k, v) in metrics.items()]))

        if step > 0 and step % 100 == 0:
            t2 = time()
            steps_s = 100. / (t2 - t)
            eta_m = int(((args.opt_num_training_steps - step) / steps_s) // 60)
            t = t2

        if step % 200 == 0:
            logger.info(np.array2string(disc_labels[0].cpu().numpy(
            ), threshold=sys.maxsize, max_line_width=sys.maxsize))
            logger.info(np.array2string(disc_pred[0].cpu().numpy(
            ), threshold=sys.maxsize, max_line_width=sys.maxsize))

        if step > 0 and step % args.step_ckpt == 0 and is_master:
            checkpoint_path = f'{args.output_dir}/ckpt/{step}'
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(generator.state_dict(),
                       f'{checkpoint_path}/generator.pth')
            torch.save(discriminator.state_dict(),
                       f'{checkpoint_path}/discriminator.pth')
            torch.save(optimizer.state_dict(),
                       f'{checkpoint_path}/optimizer.pth')
            torch.save(scheduler.state_dict(),
                       f'{checkpoint_path}/scheduler.pth')
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
            # write text file with current step
            with open(f'{args.output_dir}/best_step.txt', 'w') as f:
                f.write(str(step))
        else:
            patience_counter += 1

        if patience_counter >= args.max_patience:
            print(f"Early stopping due to no improvement after {args.max_patience} steps.")
            break



########################################################################################################
# preamble


def set_gpus(gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id, output_path):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(output_path + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(filename, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [int(folder) for folder in os.listdir(
        checkpoint_dir) if folder.isnumeric()]
    if not checkpoints:
        return None
    latest_step = max(checkpoints)
    return os.path.join(checkpoint_dir, str(latest_step))

########################################################################################################
# main


import argparse
import os


def main(args, output_path):

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id, output_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/ckpt', exist_ok=True)
    copy_source(__file__, output_dir)

    args.output_dir = output_dir
    args.exp_id = exp_id

    # distributed
    if args.distributed_enabled:
        print("Training in distributed mode with multiple processes, 1 GPU per process.")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.distributed_port)
        torch.multiprocessing.spawn(
            train, nprocs=args.distributed_world_size, args=(args,))
    else:
        print("Training with a single process on 1 GPU.")
        train(rank=args.gpu, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.environ['SM_CHANNEL_TRAINING'], "train"))
    parser.add_argument('--lr', type=float)
    parser.add_argument('--config-path', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--patience', type=int, default=500)

    parser.add_argument('--discriminator-attention-probs-dropout-prob', type=float, default=0.1)
    parser.add_argument('--discriminator-hidden-dropout-prob', type=float, default=0.4)
    parser.add_argument('--discriminator-hidden-size', type=int, default=256)
    parser.add_argument('--discriminator-intermediate-size', type=int, default=1024)
    parser.add_argument('--discriminator-num-attention-heads', type=int, default=4)


    parser.add_argument('--generator-attention-probs-dropout-prob', type=float, default=0.05)
    parser.add_argument('--generator-hidden-dropout-prob', type=float, default=0.05)
    parser.add_argument('--generator-hidden-size', type=int, default=128)
    parser.add_argument('--generator-intermediate-size', type=int, default=512)
    parser.add_argument('--generator-num-attention-heads', type=int, default=2)

    command_args = parser.parse_args()

    # Populate the Args class using the parsed command line arguments
    args = Args()
    args.opt_lr = command_args.lr
    args.opt_batch_size = command_args.batch_size
    args.model_discriminator = os.path.join(command_args.config_path, "discriminator.json")
    args.model_generator = os.path.join(command_args.config_path, "generator.json")
    args.max_patience = command_args.patience

    # Load and update discriminator config
    with open(args.model_discriminator, 'r') as file:
        discriminator_config = json.load(file)
        discriminator_config['attention_probs_dropout_prob'] = command_args.discriminator_attention_probs_dropout_prob
        discriminator_config['hidden_dropout_prob'] = command_args.discriminator_hidden_dropout_prob
        discriminator_config['hidden_size'] = command_args.discriminator_hidden_size
        discriminator_config['intermediate_size'] = command_args.discriminator_intermediate_size
        discriminator_config['num_attention_heads'] = command_args.discriminator_num_attention_heads

    # Save updated discriminator config
    with open(os.path.join(command_args.output_data_dir, 'discriminator.json'), 'w') as file:
        json.dump(discriminator_config, file)

    # Load and update generator config
    with open(args.model_generator, 'r') as file:
        generator_config = json.load(file)
        generator_config['attention_probs_dropout_prob'] = command_args.generator_attention_probs_dropout_prob
        generator_config['hidden_dropout_prob'] = command_args.generator_hidden_dropout_prob
        generator_config['hidden_size'] = command_args.generator_hidden_size
        generator_config['intermediate_size'] = command_args.generator_intermediate_size
        generator_config['num_attention_heads'] = command_args.generator_num_attention_heads

    # Save updated generator config
    with open(os.path.join(command_args.output_data_dir, 'generator.json'), 'w') as file:
        json.dump(generator_config, file)

    args.model_discriminator = os.path.join(command_args.output_data_dir, "discriminator.json")
    args.model_generator = os.path.join(command_args.output_data_dir, "generator.json")

    main(args, command_args.output_data_dir)

# python pretraining.py \
#     --lr 3e-4
    

