import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


def image_loader(path):
    return Image.open(path).convert('RGB')



