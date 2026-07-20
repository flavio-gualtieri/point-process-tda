# src/cloudforger/features/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.diagram import PersistenceDiagram

import numpy as np


class DiagramFeature(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def compute(self, diagram: PersistenceDiagram) -> Any:
        ...
