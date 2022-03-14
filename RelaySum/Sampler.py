import random

class AvgSampler:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def sample(self):
        new_sample = random.random()
        self.sum += new_sample
        self.count += 1
        return new_sample

    def get_mean(self):
        return self.sum / self.count

