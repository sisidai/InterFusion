import argparse
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random

from visualize import render_pose, add_index_to_image
from utils import get_topk_poses


def main(in_path, out_path, inter_text, topk):


    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda" # neural_render does not support inference on cpu
    
    codebook_data = torch.load(in_path)

    text = "a rendered 3d man is " + inter_text
    out_dir = os.path.join(out_path, f"{inter_text.replace(' ', '_')}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    top_poses = get_topk_poses(text, codebook_data, topk)

    indexes = list(range(topk))
    for i in indexes:
        npy_path = os.path.join(out_dir, 'top_%d.npy' % i)
        np.save(npy_path, top_poses[i].detach().cpu().numpy())
        image_path = os.path.join(out_dir, 'top_%d.jpg' % i)
        render_pose(top_poses[i], image_path)

    for i in indexes:
        img_path = os.path.join(out_dir, 'top_%d.jpg' % i)
        add_index_to_image(img_path, img_path.replace('/top', '/index'), i)
        # img_list.append(cv2.imread(img_path))
        try:
            os.remove(img_path)
        except FileNotFoundError:
            print(f"The image {img_path} does not exist.")


    random.shuffle(indexes)
    img_list = []
    for i in indexes:
        img_list.append(cv2.imread(os.path.join(out_dir, 'index_%d.jpg' % i)))

    cv2.imwrite(out_dir+'.png', np.hstack(img_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-in_path', type=str, default='../data/codebook.pth')
    parser.add_argument('-out_path', type=str, default='results/interfusion_poses')
    parser.add_argument('-inter_text', type=str, default='riding a bike')
    parser.add_argument('-topk', type=int, default=7)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    main(args.in_path, args.out_path, args.inter_text, args.topk)

    # inter_texts = ["carrying a backpack", "holding a box", "jumping a skateboard", "playing the guitar", "playing the saxophone", "playing the violin", 
    # "pushing a shopping cart", "reading a laptop", "riding a bike", "riding a horse", "riding a motorcycle", "riding an electric scooter", "sitting on a chair", "sitting on a sofa"]
