import numpy as np


class SliceScheduler:
    def __init__(self, strategy="log", max_slices=400, min_slices=2):
        self.strategy = strategy
        self.max_slices = max_slices
        self.min_slices = min_slices

    def get_num_slices(self, episode: int) -> int:
        if self.strategy == "fixed":
            return self.max_slices

        elif self.strategy == "linear":
            return min(self.max_slices, self.min_slices + episode // 100)

        elif self.strategy == "log":
            return int(min(self.max_slices, self.min_slices + np.log1p(episode) * 10))

        elif self.strategy == "curriculum":
            if episode < 3000:
                return int(self.min_slices + episode // 250)  # smoother growth
            else:
                return self.max_slices

        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")
