# dataset

import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.classes = set(df['label'].values)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        label = self.data['label'].iloc[idx]
        txt = self.data['title'].iloc[idx]
        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',  # pad to max_length
            truncation=True,
            return_tensors='pt',

        )
        ids = inputs['input_ids'].squeeze()  # (1, max_length) => (max_length,)
        msk = inputs['attention_mask'].squeeze()

        return {
            # UserWarning: To copy construct from a tensor, it is recommended
            # to use sourceTensor.clone().detach() rather than torch.tensor(sourceTensor).
            # 'ids': torch.tensor(ids, dtype=torch.long),
            'ids': ids.clone().detach(),
            'mask': msk.clone().detach(),
            'labels': torch.tensor(label, dtype=torch.long)
        }