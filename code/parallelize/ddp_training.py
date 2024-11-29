import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from transformers import T5Tokenizer, T5ForConditionalGeneration

from dataset import *
# This file contains utility_functions for distributed training.
from distributed_utils import *

def setup_model(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name, use_cache=False)
    tokenizer =  T5Tokenizer.from_pretrained(model_name, legacy=True)
    return model, tokenizer

def send_batch_to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    return batch


def train_model(model, train_loader, optimizer, device):
    """
        Train the model on the entire training dataset and return the global loss.
    """

    model.train()
    
    total_loss = 0

    for batch in train_loader:

        batch = send_batch_to_device(batch, device)
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        
        loss = output["loss"]
        loss.backward()
        total_loss += loss
        
        optimizer.step()
        optimizer.zero_grad()
    
    result = total_loss / len(train_loader)
    # Return the global average loss.
    torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)
    return result


def test_model(model, val_loader, device):
    """
        Evaluate the model on an evaluation set and return the global
        loss over the entire evaluation set.
    """
    model.eval()
    
    loss = 0

    with torch.no_grad():
        for batch in val_loader:

            batch = send_batch_to_device(batch, device)
            output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"])
            
            loss += output["loss"]

    result = loss / len(val_loader)
    # Return the global average loss.
    torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)
    return result


def main(args):
    
    # Initialize a communication group and return the right identifiers.
    local_rank, rank, device = setup()

    # Set up the model and tokenizer and move the model to the device
    model, tokenizer = setup_model("t5-base")
    model.to(device)
    # Wrap the model in DistributedDataParallel module 
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
    )

    # Set up the datasets and dataloaders
    train_dataset = Xsum(tokenizer, 'train', 1500, 512, 150)
    val_dataset = Xsum(tokenizer, 'validation', 300, 512, 150)
    test_dataset = Xsum(tokenizer, 'test', 300, 512, 150)

    # DistributedSampler object for each set to ensure that each process gets a different subset of the data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
                                                                    shuffle=True, 
                                                                    seed=args.seed)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            sampler=train_sampler, # pass the sampler argument to the DataLoader
                            num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')),
                            pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=val_sampler, # pass the sampler argument to the DataLoader
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            sampler=test_sampler, # pass the sampler argument to the DataLoader
                            pin_memory=True)             

    # Set up the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.3)
    
    best_val_loss = float("inf")

    start_time = time.perf_counter()

    # Train the model
    for epoch in range(args.epochs):
        # Pass the current epoch to the sampler to ensure proper data shuffling in each epoch
        train_sampler.set_epoch(epoch)

        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = test_model(model, val_loader, device)

        # We use the utility function print0 to print messages only from rank 0.
        print0(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, Validation loss: {val_loss:.5f}') ## TODO 11: Replace print by print0 to print messages once.

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # We allow only rank=0 to save the model
            save0(model, 'model-best')

        scheduler.step()

    end_time = time.perf_counter()
    # We use the utility function print0 to print messages only from rank 0.
    print0('Finished training after', end_time - start_time, 'seconds.') ## TODO 11: Replace print by print0 to print messages once.
    
    test_loss = test_model(model, test_loader, device)
    # We use the utility function print0 to print messages only from rank 0.
    print0('Final test loss:', test_loss.item()) ## TODO 11: Replace print by print0 to print messages once.

    # We allow only rank=0 to save the model
    save0(model, 'model-final')
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Single GPU Training')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size ')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.002,
                        help='learning rate (default: .002)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    main(args)
