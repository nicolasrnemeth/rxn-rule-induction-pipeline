from dataclasses import dataclass

@dataclass
class MetricSpec:
    name: str
    weight: float
    higher_is_better: bool = True