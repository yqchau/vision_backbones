import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_size=32 * 32, output_size=3, n_layers=3, weights=None):
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
        if weights is not None:
            self.load_state_dict(torch.load(weights))

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
        input: (batch_size, 2, 1080)
        output: (batch_size, 1)
        """
        y = x.view(x.size(0), -1)
        y = self.networks(y)
        return y


if __name__ == "__main__":
    model = FCNN(input_size=32 * 32)
    model.eval()
    inputs = torch.randn(1, 1, 32, 32)
    outputs = model(inputs)

    # print(model)
    print(outputs.shape)
