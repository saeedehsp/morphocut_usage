from contextlib import nullcontext
from random import sample
import numpy as np
from morphocut.batch import BatchPipeline
from morphocut.core import Pipeline
from morphocut.image import ImageReader, ImageWriter, RescaleIntensity
from morphocut.stream import Unpack
from morphocut.torch import PyTorch
import torch.nn
import os

train_dir = '../Crustacea/train'
test_dir = '../Crustacea/val'


class_names0 = os.listdir(train_dir)
if ".DS_Store" in class_names0:
    class_names0.remove(".DS_Store")
class_names = sorted(class_names0)
if ".DS_Store" in class_names:
    class_names.remove(".DS_Store")

num_class = len(class_names)
image_files = [[os.path.join(train_dir, class_name, x)
               for x in os.listdir(os.path.join(train_dir, class_name))]
               for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
num_total = len(image_label_list)


class MyModule(torch.nn.Module):
    def __init__(self, output_key=None) -> None:
        super().__init__()
        self.output_key = output_key

    def forward(self, input):
        if self.output_key is None:
            return input

        return {self.output_key: input}


def t_PyTorch(device, n_parallel, batch, output_key):
    module = MyModule(output_key)
    image_path = sample(image_file_list,1)[0]
    with Pipeline() as p:
        input_im = ImageReader(image_path)
        block = BatchPipeline(2) if batch else nullcontext(p)
        with block:
            result = PyTorch(
                module,
                input_im,
                is_batch=batch,
                device=device,
                n_parallel=n_parallel,
                output_key=output_key,
            )
            result = RescaleIntensity(result, dtype=np.uint8)
    p.run()

    new_path = image_path.replace("train", "DL/Pytorch")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    results = [o[result] for o in p.transform_stream()]
    with Pipeline() as pipeline:
        ImageWriter(new_path, results[0][0])
    pipeline.run()

t_PyTorch("cpu", 0, False, "foo")
