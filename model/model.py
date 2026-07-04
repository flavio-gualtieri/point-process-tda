# model/model.py

from __future__ import annotations

import pickle

import numpy as np
import torch
import torch.nn as nn

from cloudforger.nn.encoders.stats import StatsEncoder
from cloudforger.nn.models.single_modal import SingleModalModel

DEFAULT_MODEL_PATH = "/Users/qp252676/Desktop/point-process-tda/results/params/2d/thomas/betti_0/model.pt"
DEFAULT_RESULTS_PATH = "/Users/qp252676/Desktop/point-process-tda/results/params/2d/thomas/betti_0/results.pt"
DEFAULT_DATA_PATH = "/Users/qp252676/Desktop/point-process-tda/data/barro/bci.tree1_betti0.pkl"


class Model(nn.Module):

    def __init__(
            self,
            model_path: str = DEFAULT_MODEL_PATH,
            results_path: str = DEFAULT_RESULTS_PATH,
    ):
        super().__init__()

        self.model: SingleModalModel = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
        self.model.eval()

        results = torch.load(results_path, map_location="cpu", weights_only=False)
        self.label_names: list[str] = results["label_names"]
        self.register_buffer(
            "log_mean", torch.as_tensor(np.asarray(results["label_log_mean"]), dtype=torch.float32)
        )
        self.register_buffer(
            "log_std", torch.as_tensor(np.asarray(results["label_log_std"]), dtype=torch.float32)
        )

    @torch.no_grad()
    def forward(self, betti_0: torch.Tensor) -> dict[str, float] | torch.Tensor:
        x = torch.as_tensor(betti_0, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        normalized_log_params = self.model(x)
        params = torch.exp(normalized_log_params * self.log_std + self.log_mean)

        if params.shape[0] == 1:
            return dict(zip(self.label_names, params[0].tolist()))
        return params


def load_betti_0(data_path: str) -> torch.Tensor:
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list):
        data = data[0]
    curve = np.asarray(data["betti0_curve"], dtype=np.float32)
    return torch.as_tensor(curve, dtype=torch.float32)


def main() -> None:
    data_path = DEFAULT_DATA_PATH
    betti_0 = load_betti_0(data_path)
    model = Model()
    params_est = model.forward(betti_0=betti_0)
    print(params_est)


if __name__ == "__main__":
    main()