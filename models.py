"""Neural networks for retrieving etched dimensions from processed THz spectra."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, inputs, hidden=(256, 128, 64, 32, 16)):
        super().__init__()
        widths = (inputs, *hidden, 4)
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.bn_linears = nn.ModuleList(nn.BatchNorm1d(b) for b in widths[1:])

    def forward(self, x):
        for linear, norm in zip(list(self.linears)[:-1], self.bn_linears):
            x = F.relu(norm(linear(x)))
        return self.linears[-1](x)


class CNN(nn.Module):
    def __init__(self, inputs, pool_bins=4, dropout=0.1):
        super().__init__()
        self.input_size = inputs
        layers = []
        for cin, cout, kernel, dilation in (
            (1, 32, 7, 1), (32, 32, 5, 1), (32, 64, 5, 2),
            (64, 64, 3, 4), (64, 128, 3, 8), (128, 128, 3, 1),
        ):
            layers.append(nn.Sequential(
                nn.Conv1d(cin, cout, kernel, padding=(kernel // 2) * dilation,
                          dilation=dilation, bias=False),
                nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            ))
        self.feat = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(pool_bins))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * pool_bins, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, 4),
        )

    def forward(self, x):
        return self.head(self.feat(x.unsqueeze(1)))


class Transformer(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.input_proj = nn.Linear(inputs, 128)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=2, enable_nested_tensor=False,
        )
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)
        return self.fc(self.transformer_encoder(x).squeeze(0))


MODELS = {"MLP": MLP, "CNN": CNN, "Transformer": Transformer}


def save_weights(path, state):
    payload = {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **payload)


def load_weights(kind, inputs, path, device="cpu"):
    with np.load(path, allow_pickle=False) as data:
        state = {name: torch.from_numpy(data[name]) for name in data.files}
    if kind == "MLP":
        hidden = [state[f"linears.{i}.weight"].shape[0]
                  for i in range(sum(k.startswith("linears.") and k.endswith(".weight") for k in state) - 1)]
        model = MLP(inputs, hidden)
    else:
        model = MODELS[kind](inputs)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, spectra, device="cpu", batch_size=64):
    model.eval()
    x = torch.as_tensor(spectra, dtype=torch.float32)
    return np.concatenate([model(batch.to(device)).cpu().numpy() for batch in x.split(batch_size)])
