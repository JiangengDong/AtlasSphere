from __future__ import division
import torch
from torch import nn as nn


# class VoxelEncoder(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(VoxelEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=[5, 5], stride=[2, 2]),
#             nn.PReLU(),
#             nn.Conv2d(in_channels=16, out_channels=8, kernel_size=[3, 3], stride=[1, 1]),
#             nn.PReLU(),
#             nn.MaxPool2d(kernel_size=[2, 2])
#         )
#         x = self.encoder(torch.autograd.Variable(torch.rand([1, input_size, input_size, input_size])))
#         first_fc_in_features = 1
#         for n in x.size()[1:]:
#             first_fc_in_features *= n
#         self.head = nn.Sequential(
#             nn.Linear(first_fc_in_features, 128),
#             nn.PReLU(),
#             nn.Linear(output_size, output_size)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         x = self.head(x)
#         return x


# class PNet(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(PNet, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_size, 896), nn.PReLU(), nn.Dropout(),
#             nn.Linear(896, 512), nn.PReLU(), nn.Dropout(),
#             nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
#             nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
#             nn.Linear(128, 64), nn.PReLU(),
#             nn.Linear(64, output_size))

#     def forward(self, x):
#         out = self.fc(x)
#         return out


# class PNet_Annotated(torch.jit.ScriptModule):
#     __constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','device']
#     def __init__(self):
#         super(PNet_Annotated, self).__init__()
#         self.fc1 = nn.Sequential(nn.Linear(134, 896), nn.PReLU())
#         self.fc2 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
#         self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
#         self.fc4 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
#         self.fc5 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
#         self.fc6 = nn.Sequential(nn.Linear(64, 3))

#         self.device = torch.device('cuda')

#     @torch.jit.script_method
#     def forward(self, x):
#         p = 0.5
#         scale = 1.0/p

#         drop1 = (scale)*torch.bernoulli(torch.full((1, 896), p)).to(device=self.device)
#         drop2 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
#         drop3 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
#         drop4 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)
#         drop5 = (scale)*torch.bernoulli(torch.full((1, 64), p)).to(device=self.device)

#         out1 = self.fc1(x)
#         out1 = torch.mul(out1, drop1)

#         out2 = self.fc2(out1)
#         out2 = torch.mul(out2, drop2)

#         out3 = self.fc3(out2)
#         out3 = torch.mul(out3, drop3)

#         out4 = self.fc4(out3)
#         out4 = torch.mul(out4, drop4)

#         out5 = self.fc5(out4)
#         out5 = torch.mul(out5, drop5)

#         out6 = self.fc6(out5)

#         return out6



class VoxelEncoder(nn.Module):
    def __init__(self, input_size, output_size): #input_size=64000; output_size=128
        super(VoxelEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256), nn.PReLU(),
            nn.Linear(256, 256), nn.PReLU(),
            nn.Linear(256, output_size))
    def forward(self, x):
        out = self.fc(x)
        return out


class PNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, output_size))
    def forward(self, x):
        out = self.fc(x)
        return out

class PNet_Annotated(torch.jit.ScriptModule):
    __constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6', "device"]
    def __init__(self, input_size, output_size):
        super(PNet_Annotated, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, 512), nn.PReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
        self.fc3 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
        self.fc4 = nn.Sequential(nn.Linear(128, 64), nn.PReLU())
        self.fc5 = nn.Sequential(nn.Linear(64, 32), nn.PReLU())
        self.fc6 = nn.Sequential(nn.Linear(32, output_size))

        self.device = torch.device("cuda")

    @torch.jit.script_method
    def forward(self, x):
        p = 0.5
        scale = 1.0/p

        drop1 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
        drop2 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
        drop3 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)
        drop4 = (scale)*torch.bernoulli(torch.full((1, 64), p)).to(device=self.device)

        out1 = self.fc1(x)
        out1 = torch.mul(out1, drop1)

        out2 = self.fc2(out1)
        out2 = torch.mul(out2, drop2)

        out3 = self.fc3(out2)
        out3 = torch.mul(out3, drop3)

        out4 = self.fc4(out3)
        out4 = torch.mul(out4, drop4)

        out5 = self.fc5(out4)

        out6 = self.fc6(out5)

        return out6
