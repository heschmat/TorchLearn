import torch
import torch.nn as nn

from tqdm import tqdm

from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot


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


def valid_one_epoch(val_dataloader, model, criterion):
    """
    Validate at the end of one eopoch training.
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    if cuda_available:
        model.to(device)

    with torch.inference_mode():
        model.eval()
        val_loss = 0.0

        for batch_idx, (images, labels) in tqdm(enumerate(val_dataloader, start= 1),
                                                desc= 'Validating',
                                                total= len(val_dataloader),
                                                leave= True,
                                                ncols= 80):
            if cuda_available:
                images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            # Check `train_one_epoch` for logic:
            val_loss += (1 / batch_idx) * (loss.data.item() - val_loss)
        return val_loss


def optimize(data_loaders, model, optimizer, criterion, n_epochs,
             model_savepath,
             interactive_tracking= False):
    liveloss = None
    if interactive_tracking:
        #@TODO Add `after_subplots()` in the `helper` module
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot= after_subplot)])

    val_loss_min = None
    logs = {}

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience= 3)
    for epoch in range(1, n_epochs + 1):
        trn_loss = train_one_epoch(data_loaders['train'], model, optimizer, criterion)
        val_loss = valid_one_epoch(data_loaders['valid'], model, criterion)

        print(f'Epoch [{epoch}/{n_epochs}] => Train Loss: {trn_loss:.4f} -- Valid Loss: {val_loss:.4f}')

    if val_loss_min is None or (val_loss_min - val_loss) > (.01 * val_loss_min):
        val_loss_min = val_loss
        print(f'New min validation loss: {val_loss_min:.4f}. Saving model ...')
        torch.save(model.state_dict(), model_savepath)

    scheduler.step()

    if interactive_tracking:
        logs['loss'] = trn_loss
        logs['val_loss'] = val_loss
        logs['lr'] = optimizer.param_groups[0]['lr']

        liveloss.update(logs)
        liveloss.send()
