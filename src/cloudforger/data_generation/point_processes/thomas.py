# src/cloudforger/processes/thomas.py

import math

import numpy as np

from .neyman_scott import NeymanScottProcess, gaussian_displacements, poisson_counts


class ThomasProcess(NeymanScottProcess):

    def __init__(
        self,
        parent_intensity: float,
        mean_offspring: float,
        cluster_scale: float,
        edge_buffer: float | None = None,
    ):
        if edge_buffer is None:
            edge_buffer = 4.0 * cluster_scale

        super().__init__(
            parent_intensity=parent_intensity,
            offspring_count_sampler=poisson_counts(mean_offspring),
            displacement_sampler=gaussian_displacements(cluster_scale),
            edge_buffer=edge_buffer,
            process_name="thomas",
            param_dict={
                "mean_offspring": mean_offspring,
                "cluster_scale": cluster_scale,
                "c1": 2.0 * cluster_scale * math.sqrt(parent_intensity),
            },
        )