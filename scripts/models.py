import torch
from torch import nn


class VoxelEncoder(nn.Module):
    def __init__(self, voxel_shape, output_size):
        super(VoxelEncoder, self).__init__()
        in_channels = voxel_shape[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3)),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # TODO: I don't like this way of figuring out the tensor size. However, it is easy and it works.
        with torch.no_grad():
            x = self.encoder(torch.autograd.Variable(torch.rand([1] + list(voxel_shape))))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 64),
            nn.PReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class PNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024), nn.PReLU(), nn.Dropout(),
            # nn.Linear(4096, 2048), nn.PReLU(), #nn.Dropout(),
            # nn.Linear(2048, 1024), nn.PReLU(), nn.Dropout(),
            nn.Linear(1024, 512), nn.PReLU(),  # nn.Dropout(),
            nn.Linear(512, 256), nn.PReLU(),  # nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(),  # nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out


class MPNet(nn.Module):
    def __init__(self, voxel_shape=(40, 40, 40), encoder_output_size=64, state_size=3):
        super(MPNet, self).__init__()
        self.state_size = state_size
        self.encoder = VoxelEncoder(voxel_shape=voxel_shape,
                                    output_size=encoder_output_size)
        self.pnet = PNet(input_size=encoder_output_size + state_size * 2,
                         output_size=state_size)

    def forward(self, x, obs):
        z = self.encoder(obs)
        hidden = torch.cat((z, x), 1)
        pred = self.pnet(hidden)
        output = pred.clone()
        return output
