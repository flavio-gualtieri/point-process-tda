# src/cloudforger/processes/thomas.py

import math

import numpy as np

from .neyman_scott import NeymanScottProcess, poisson_counts
from .kernels import GaussianKernel


class ThomasProcess(NeymanScottProcess):

    def __init__(
        self,
        parent_intensity: float,
        mean_offspring: float,
        cluster_scale: float,
        edge_buffer: float | None = None,
    ):

        super().__init__(
            parent_intensity=parent_intensity,
            kernel=GaussianKernel(cluster_scale),
            offspring_count_sampler=poisson_counts(mean_offspring),
            edge_buffer=edge_buffer,
            process_name="thomas",
            param_dict={
                "mean_offspring": mean_offspring,
                "c1": 2.0 * cluster_scale * math.sqrt(parent_intensity),
            },
        )