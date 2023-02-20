from contextlib import nullcontext
from random import sample
from morphocut.batch import BatchPipeline
from morphocut.core import Pipeline
from morphocut.image import ImageWriter, RescaleIntensity
from src.tensor_flow import TensorFlow
from PIL import Image
import PIL
import numpy as np
import tensorflow as tf
import os
from torchvision.transforms.functional import to_tensor as NumpyToTensor

train_dir = './Crustacea/train'
test_dir = './Crustacea/val'


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


class MyModule(tf.Module):
    def __init__(self, output_key=None):
        super().__init__()
        self.output_key = output_key

    def __call__(self, input):
        if self.output_key is None:
            return input

        return {self.output_key: input}


def t_tensor_flow(device, n_parallel, batch, output_key):
    module = MyModule(output_key)
    image_path = sample(image_file_list,1)
    input_pil = PIL.Image.open(image_path[0])
    input_np = np.array(input_pil)
    input_im = NumpyToTensor(input_np)
    with Pipeline() as p:
        block = BatchPipeline(2) if batch else nullcontext(p)
        with block:
            result = TensorFlow(
                module,
                input_im,
                is_batch=batch,
                device=device,
                n_parallel=n_parallel,
                output_key=output_key,
            )
            result = RescaleIntensity(result, dtype=np.uint8)
    p.run()

    new_path = image_path[0].replace("train", "DL/TensorFlow")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    results = [o[result] for o in p.transform_stream()]
    with Pipeline() as pipeline:
        ImageWriter(new_path, results[0][0])
    pipeline.run()


t_tensor_flow("cpu", 0, False, "foo")
