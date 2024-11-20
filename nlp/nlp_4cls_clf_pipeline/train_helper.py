# train_helper

import torch
from logger import logger


# Train & Eval code ---------------------------------------------------- #
def train_1epoch(model, data_loader, optimizer, criterion, epoch, device):
    tr_loss = 0
    n_correct = 0
    n_steps = 0
    n_examples = 0

    model.train()
    for idx, data in enumerate(data_loader, start=1):
        ids = data['ids'].to(device, dtype=torch.long)
        msk = data['mask'].to(device, dtype=torch.long)
        labels = data['labels'].to(device, dtype=torch.long)

        outputs = model(ids, msk)
        loss = criterion(outputs, labels)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += (big_idx == labels).sum().item()

        n_steps += 1
        n_examples += labels.size(0)

        if idx % int(len(data_loader) / 3) == 0:
            step_loss = tr_loss / n_steps
            step_acc = (n_correct * 100) / n_examples
            logger.info(
                f'\tAfter {n_examples} examples >>> Train Loss: {step_loss:.4f} --- Train Acc : {step_acc:.2f}'
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_epoch = tr_loss / n_steps
    acc_epoch = (n_correct * 100) / n_examples
    logger.info(f'EPOCH {epoch} >>> Train Loss: {loss_epoch:.4f} | Train Acc: {acc_epoch:.2f}')


def eval_1epoch(model, data_loader, criterion, epoch, device):
    n_correct = 0
    running_loss = 0
    n_steps = 0
    n_examples = 0

    # AttributeError: module 'torch' has no attribute 'inference_mode'
    with torch.no_grad():
        for idx, data in enumerate(data_loader, start=1):
            ids = data['ids'].to(device, dtype=torch.long)
            msk = data['mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)

            outputs = model(ids, msk).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += (big_idx == labels).sum().item()

            n_steps += 1
            n_examples += labels.size(0)

    loss_epoch = running_loss / n_steps
    acc_epoch  = (n_correct * 100) / n_examples
    logger.info(f'EPOCH {epoch} >>> Eval Loss: {loss_epoch:.4f} | Eval Acc: {acc_epoch:.2f}\n')
