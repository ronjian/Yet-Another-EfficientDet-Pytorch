import yaml
from backbone import EfficientDetBackbone
import torch
import torch.nn as nn
from torch.nn import functional as F

class Pad(nn.Module):
    def __init__(self):
        super(Pad, self).__init__()
        pass

    def forward(self, x):
        # x = torch.constant_pad_nd(x,(0,0,0,2))
        x = F.pad(x, [0,0,0,2])
        return x

model = Pad()
model.eval()
_ = model(torch.Tensor(1,3,512,512))

trace_model = torch.jit.trace(model, (torch.Tensor(1,3,512,512), ))
trace_model.save('./weights/pad_jit.pt')
