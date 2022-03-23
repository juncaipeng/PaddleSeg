# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
To visualize the annotation:
1. Add the origin image and annotated image to produce the weighted annotated image.
2. If provide the directory of predicted images by pred_dir,  use the same method to
    produce the weight predicted image.
3. Paste these images to generate the final image.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from paddleseg import utils
from paddleseg.utils import logger, progbar, visualize


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument(
        '--file_path',
        help='The file contains the path of origin and annotated images',
        type=str)
    parser.add_argument(
        '--pred_dir',
        help='The directory of predicted images. It is dispensable.',
        type=str)
    parser.add_argument('--save_dir',
                        help='The directory for saving the visualized images',
                        type=str,
                        default='./output/visualize_annotation')
    return parser.parse_args()


def get_images_path(file_path):
    """
    Get the path of origin images and annotated images.
    """
    assert os.path.isfile(file_path)

    images_path = []
    image_dir = os.path.dirname(file_path)

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            origin_path, annot_path = line.split(" ")
            origin_path = os.path.join(image_dir, origin_path)
            annot_path = os.path.join(image_dir, annot_path)
            images_path.append([origin_path, annot_path])

    return images_path


def main(args):
    file_path = args.file_path
    pred_dir = args.pred_dir
    save_dir = args.save_dir
    weight = 0.3

    images_path = get_images_path(file_path)
    bar = progbar.Progbar(target=len(images_path), verbose=1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (origin_path, annot_path) in enumerate(images_path):
        origin_img = Image.open(origin_path)
        annot_img = Image.open(annot_path).convert("L")
        annot_img = np.array(annot_img)
        img_list = [origin_img]

        # weighted annoted image
        color_map = visualize.get_color_map_list(256)
        wt_annoted_img = utils.visualize.visualize(origin_path,
                                                   annot_img,
                                                   color_map,
                                                   weight=weight)
        wt_annoted_img = Image.fromarray(
            cv2.cvtColor(wt_annoted_img, cv2.COLOR_BGR2RGB))
        img_list.append(wt_annoted_img)

        # weighted pred image
        if pred_dir is not None:
            filename = os.path.basename(origin_path)
            filename = filename.split(".")[0] + ".png"
            pred_path = os.path.join(pred_dir, filename)
            assert os.path.exists(
                pred_path), "The predicted image {} is not existed".format(
                    pred_path)
            pred_img = Image.open(pred_path)
            pred_img = np.array(pred_img)

            wt_pred_img = utils.visualize.visualize(origin_path,
                                                    pred_img,
                                                    color_map,
                                                    weight=weight)
            wt_pred_img = Image.fromarray(
                cv2.cvtColor(wt_pred_img, cv2.COLOR_BGR2RGB))
            img_list.append(wt_pred_img)

        # result image
        result_img = visualize.paste_images(img_list)

        image_name = os.path.split(origin_path)[-1]
        result_img.save(os.path.join(save_dir, image_name))

        bar.update(idx + 1)


if __name__ == '__main__':
    args = parse_args()
    main(args)
