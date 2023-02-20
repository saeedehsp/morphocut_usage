from morphocut import Pipeline
from morphocut.image import (
    ExtractROI,
    FindRegions,
    Gray2RGB,
    ImageReader,
    ImageWriter,
    RescaleIntensity,
    RGB2Gray,
    ThresholdConst,
    ImageStats,
)
import numpy as np
import os
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


def roi_crustacea():
    for image_path in image_file_list:
        with Pipeline() as p:
            image = ImageReader(image_path)
            image = RGB2Gray(image)
            mask = ThresholdConst(image, 255)
            regions = FindRegions(mask, image)
            roi = ExtractROI(image, regions, bg_color=255)
            result = RescaleIntensity(roi, dtype=np.uint8)
            new_path = image_path.replace("train", "ROI")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            ImageWriter(new_path, result)
        p.run()


roi_crustacea()
