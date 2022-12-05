import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_size=64 * 64, output_size=3, n_layers=3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.networks = nn.Sequential(
            *[
                self.get_fc_block(
                    self.input_size // 2**i,
                    self.input_size // 2 ** (i + 1)
                    if i != n_layers - 1
                    else output_size,
                    activation="relu" if i != n_layers - 1 else None,
                )
                for i in range(n_layers)
            ],
        )

    def get_fc_block(self, input_shape, output_shape, activation="relu"):

        if activation is None:
            return nn.Sequential(
                nn.Linear(input_shape, output_shape),
            )

        return nn.Sequential(
            nn.Linear(input_shape, output_shape),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
            nn.BatchNorm1d(output_shape),
        )

    def forward(self, x):
        """
        input: (batch_size, input_size)
        output: (batch_size, 1)
        """
        y = x.view(x.size(0), -1)
        y = self.networks(y)
        return y


class CNN(nn.Module):
    def __init__(self, input_size=(64, 64), output_size=3):
        super().__init__()
        self.input_size = input_size
        self.cnn = nn.Sequential(
            self.get_cnn_block(1, 16),
            self.get_cnn_block(16, 32),
            self.get_cnn_block(32, 64),
        )
        self.n_channel, self.final_image_size = self.get_final_image_size()
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channel, output_size)

    def get_cnn_block(self, input_shape, output_shape):

        return nn.Sequential(
            nn.Conv2d(input_shape, output_shape, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_shape),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def get_fc_block(self, input_shape, output_shape, activation="relu"):
        return nn.Sequential(
            nn.Linear(input_shape, output_shape),
            nn.ReLU() if activation == "relu" else nn.Tanh(),
        )

    def get_final_image_size(self):
        inputs = torch.randn(1, 1, *self.input_size)
        outputs = self.cnn(inputs)

        size = list(outputs.shape[1:])
        return size[0], size[1:]

    def forward(self, x):
        """
        input: (batch_size, 1, input_size, input_size)
        output: (batch_size, output_size)
        """

        y = self.cnn(x)
        y = self.global_average_pool(y)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        return y


if __name__ == "__main__":
    model = FCNN()
    print(model)
