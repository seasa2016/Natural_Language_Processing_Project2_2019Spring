import torch
import numpy as np

logits = torch.rand(1,3)

preds=[]
if len(preds) == 0:
    preds.append(logits.detach().cpu().numpy())
else:
    preds[0] = np.append(
        preds[0], logits.detach().cpu().numpy(), axis=0)
print(preds)
if len(preds) == 0:
    preds.append(logits.detach().cpu().numpy())
else:
    preds[0] = np.append(
        preds[0], logits.detach().cpu().numpy(), axis=0)
print(preds)
if len(preds) == 0:
    preds.append(logits.detach().cpu().numpy())
else:
    preds[0] = np.append(
        preds[0], logits.detach().cpu().numpy(), axis=0)
print(preds)