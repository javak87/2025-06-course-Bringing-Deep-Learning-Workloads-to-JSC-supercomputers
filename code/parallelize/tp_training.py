from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    SequenceParallel,
)
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.fsdp import fully_shard

from torch.utils.data import DataLoader

from dataset import LanguageModelingDataset, build_vocab
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LanguageModelingDataset, build_vocab
from transformerLM import TransformerLM, ModelArgs

# This file contains utility_functions for distributed training.
from distributed_utils import *

# from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed._tensor import Shard, Replicate, Partial
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel,
    loss_parallel
    # PrepareModuleInputOutput
)

from torch.distributed.tensor import DTensor
from torch.distributed._tensor.experimental import implicit_replication

def train_model(model, train_loader, vocab, optimizer, loss_func, device):
    """
        Train the model on the entire training dataset and return the global loss.
    """

    model.train()
    
    total_loss = 0

    for _, (src, tgt) in enumerate(train_loader):
        # with implicit_replication():
            # src = DTensor.from_local(src)
            # tgt = DTensor.from_local(tgt)
        src, tgt = src.to(device), tgt.to(device)
        output = model(src)  # (seq_len, batch, vocab)

   
        loss = loss_func(output.view(-1, len(vocab)), tgt.t().reshape(-1))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss

    result = total_loss / len(train_loader)
    ## TODO 10: Obtain the global average loss.


    return result

def test_model(model, dataloader, vocab, loss_func, device):
    """
        Evaluate the model on an evaluation set and return the global
        loss over the entire evaluation set.
    """
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = loss_func(output.view(-1, len(vocab)), tgt.t().reshape(-1))
            total_loss += loss

    result = total_loss / len(dataloader)
    ## TODO 10: Obtain the global average loss.


    return result



def main(args):

    # Initialize a communication group and return the right identifiers.
    local_rank, rank, device, world_size = setup()
    print0(f"Local rank: {local_rank}, Rank: {rank}, Device: {device}, World size: {world_size}")
    
    device_type = torch.accelerator.current_accelerator().type
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    # Build vocab from training data
    vocab, stoi, itos = build_vocab('train')

    # Set up the datasets and dataloaders Shared across all splits
    train_dataset = LanguageModelingDataset('train', seq_len=32, stoi=stoi, vocab=vocab)
    val_dataset = LanguageModelingDataset('validation', seq_len=32, stoi=stoi, vocab=vocab)
    test_dataset = LanguageModelingDataset('test', seq_len=32, stoi=stoi, vocab=vocab)

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
                            pin_memory=True,
                            drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            sampler=val_sampler, # pass the sampler argument to the DataLoader
                            pin_memory=True,
                            drop_last=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            sampler=test_sampler, # pass the sampler argument to the DataLoader
                            pin_memory=True,
                            drop_last=True)


    # Set up the model and move it to the device
    # model = TransformerLM(vocab_size=len(vocab), d_model=128, nhead=4, num_layers=2)

    model_args = ModelArgs(dim=128, n_heads=4, max_seq_length=2048, vocab_size=len(vocab), num_encoder_layers=2)
    model = TransformerLM(model_args)

    # simple_llama2_config = ModelArgs(dim=128, n_layers=2, n_heads=16, vocab_size=len(vocab), max_seq_len=1024)

    # model = Transformer.from_model_args(simple_llama2_config).to(device_type)

    model = model.to(device_type)
    print(f"Model is on device: {model}")
    # model = model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[local_rank],
    # )    

    # parallelize the first embedding and the last linear out projection
    model = parallelize_module(
        model,
        device_mesh,
        {
            "embed": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False
            ),
            "norm": SequenceParallel(),
            "fc": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
                # use_local_output=False
            ),
        }
    )

    for layer_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = {
            # "attention_norm": SequenceParallel(),
            "self_attn": PrepareModuleInput(
                input_layouts=(Shard(1), Shard(1), Shard(1)),
                desired_input_layouts=(Replicate(), Replicate(), Replicate()),
            ),
            "self_attn.linear_Q": ColwiseParallel(
                # input_layouts=Replicate(),   # ⬅️ each rank gets the *full* query vector
                # output_layouts=Replicate(),     # ⬅️ then we slice the output features across ranks
                use_local_output=False
            ),
            "self_attn.linear_K": ColwiseParallel(
                # input_layouts=Replicate(),
                # output_layouts=Replicate(),
                use_local_output=False
            ),
            "self_attn.linear_V": ColwiseParallel(
                # input_layouts=Replicate(),
                # output_layouts=Replicate(),
                use_local_output=False
            ),
            "self_attn.out_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),

            "norm1": SequenceParallel(),
            "linear1": ColwiseParallel(use_local_output=False),
            "linear2": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "norm2": SequenceParallel(use_local_output=False),

        }

        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block,
            device_mesh=device_mesh,
            parallelize_plan=layer_tp_plan
        )


    # 1) make the pe-buffer contiguous
    local_pe = model.pos_encoder.pe.contiguous()

    # 2) reconstruct it as a replicated DTensor
    model.pos_encoder.pe = DTensor.from_local(
        local_pe,      # now contiguous
        device_mesh,   # your DeviceMesh
        # Replicate()    # <— don’t forget this third arg!
    )


    # Set up the loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, foreach=True)
    
    best_val_loss = float("inf")

    # Train the model
    for epoch in range(args.epochs):
        # Pass the current epoch to the sampler to ensure proper data shuffling in each epoch
        train_sampler.set_epoch(epoch)

        train_loss = train_model(model, train_loader, vocab, optimizer, loss_func, device_type)
        val_loss = test_model(model, val_loader, vocab, loss_func, device_type)

        # We use the utility function print0 to print messages only from rank 0.
        print0(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            ## TODO 18: Replace save0 method by either save_full_model or save_sharded_model to save the full model state or the sharded model state respectively.
            # We allow only rank=0 to save the model
            # save0(model, 'model-best.pt')

    
    test_loss = test_model(model, test_loader, vocab, loss_func, device)
    # We use the utility function print0 to print messages only from rank 0.
    print0('Final test loss:', test_loss.item())

    ## TODO 18: Replace save0 method by either save_full_model or save_sharded_model to save the full model state or the sharded model state respectively.
    # We allow only rank=0 to save the model
    # save0(model, 'model-final.pt')

    # Destroy the process group to clean up resources
    destroy_process_group()




if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Single GPU Training')
    parser.add_argument('--batch-size', type=int, default=128,
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

