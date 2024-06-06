import torch
import torch.nn as nn

from tqdm import tqdm

def train_one_epoch(trn_dataloader, model, optimizer, criterion):
    """
    Performs one training epoch.
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    if cuda_available: model.cuda()
    
    model.train()
    trn_loss = 0.0

    for batch_idx, (images, labels) in tqdm(enumerate(trn_dataloader, start= 1),
                                            desc= 'Training',
                                            total= len(trn_dataloader),
                                            leave= True,
                                            ncols= 80):
        if cuda_available:
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Update the running total of the training loss by calculating a weighted average
        # of the current batch's loss relative to the existing total loss.
        # This helps stabilize the loss estimate and gives higher weight to earlier batches.
        trn_loss += (1 / batch_idx) * (loss.data.item() - trn_loss)

    return trn_loss

