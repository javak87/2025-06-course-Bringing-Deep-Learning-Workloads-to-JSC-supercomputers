import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from transformers import T5Tokenizer, T5ForConditionalGeneration

from xsum import *
## TODO 1: Import distributed_utils to use the utility methods available in it.


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
    ## TODO 10: Obtain the global average loss.
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
    ## TODO 10: Obtain the global average loss.
    return result


def main(args):
    
    ## TODO 2-3: Remove this line and replace it with a call to the utility function setup().
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the model and tokenizer and move the model to the device
    model, tokenizer = setup_model("t5-base")
    model.to(device)
    ## TODO 4: # Wraps the model in a DistributedDataParallel (DDP) module to parallelize the training across multiple GPUs.


    # Set up the datasets and dataloaders
    train_dataset = Xsum(tokenizer, 'train', 1500, 512, 150)
    val_dataset = Xsum(tokenizer, 'validation', 300, 512, 150)
    test_dataset = Xsum(tokenizer, 'test', 300, 512, 150)

    ## TODO 5: Create a DistributedSampler object for each set. ** shuffle=True only for training set


    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, ## TODO 6: Remove this line and replace it the sampler argument 
                            num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')),
                            pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            ## TODO 7: Don't forget to pass val_sampler to the sampler argument of the DataLoader.
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            ## TODO 8: Don't forget to pass test_sampler to the sampler argument of the DataLoader.
                            pin_memory=True)             

    # Set up the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.3)
    
    best_val_loss = float("inf")

    start_time = time.perf_counter()

    # Train the model
    for epoch in range(args.epochs):
        ## TODO 9: Sets the current epoch for the dataset sampler to ensure proper data shuffling in each epoch


        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = test_model(model, val_loader, device)


        print(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, Validation loss: {val_loss:.5f}') ## TODO 11: Replace print by print0 to print messages once.

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, 'model-best') ## TODO 12: Replace torch.save method with the utility function save0 to save the model.


        scheduler.step()

    end_time = time.perf_counter()
    print('Finished training after', end_time - start_time, 'seconds.') ## TODO 11: Replace print by print0 to print messages once.
    
    test_loss = test_model(model, test_loader, device)
    print('Final test loss:', test_loss.item()) ## TODO 11: Replace print by print0 to print messages once.

    torch.save(model, 'model-final') ## TODO 12: Replace torch.save method with the utility function save0 to save the model.
    
    
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Single GPU Training')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size ')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.002,
                        help='learning rate (default: .002)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    main(args)
