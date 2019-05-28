from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch


a = [
    torch.tensor([[1],[2],[3]],dtype=torch.float),
    torch.tensor([[1,2],[2,3],[3,4]],dtype=torch.float)
    ]

train_data = TensorDataset(a[0],a[1])
print(train_data[0])


# train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)