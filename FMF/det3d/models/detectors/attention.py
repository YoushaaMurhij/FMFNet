import torch
from torch import nn
import torch.nn.functional as F

class _SelfAttentionBlock(nn.Module):

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)

    def forward(self, x, x_old):
        softmax_mode = 1

        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        query = self.f_query(x)
        key = self.f_key(x_old)
        value = self.f_value(x_old)

        key_transpose=torch.transpose(key, 2, 3)

        context = torch.matmul(query,key_transpose)
        context = (128**-.5) * context

        if softmax_mode == 1:
            context = context.view(batch_size, self.value_channels, -1)
            
            context = F.softmax(context, dim=-1)
            context = context.view(batch_size, self.value_channels, *x.size()[2:])
        else:
            context = F.softmax(context, dim=-1)

        context = torch.matmul(context,value)
   
        return context

def main():
    device = torch.device('cuda:0')
    print(f'cuda device is: {device}')
    torch_model = _SelfAttentionBlock(384,384,384).to(device)
    x = torch.randn(4, 384, 128, 128, requires_grad=True).to(device)
    x_old = torch.randn(4, 384, 128, 128, requires_grad=True).to(device)
    for i in range(100000):
        y = torch_model(x,x_old)
    print(y.shape)

    
if __name__ == "__main__":
    main()