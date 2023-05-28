import time
import datetime
import argparse
import os
import copy

import numpy as np
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision
import warnings
warnings.filterwarnings("ignore")

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import Model

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


def format_bbox(boxes_filt, pred_phrases, classes):
    output_text = ''
    for box, label in zip(boxes_filt, pred_phrases):
        box.numpy()
        x0, y0 = box[0], box[1] # bot left most likely, not checked
        w, h = box[2] - box[0], box[3] - box[1]
        x, y = (box[2] + box[0])/2, (box[3] + box[1])/2
        label = label[:label.index('(')]
        if label == '':
            label_idx = -1
        else:
            # if label[0] == '#':
            #     label = 'excavator'  # todo: quick fix, not sure why # is generated in GDINO
            label_idx = Model.find_index(label, classes)
        output_text = output_text + f'{label_idx} {x} {y} {w} {h}\n'
    # print(output_text)
    return output_text


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


from automatic_label_demo import get_grounding_output


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    # parser.add_argument(
    #     "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    # )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    # image_path = args.input_image
    text_prompt = args.text_prompt
    # output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    iou_threshold = args.iou_threshold
    classes = text_prompt.lower().replace('.','').split()
    print(classes)

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    ######################################################
    image_folder = r'/media/leyang/New Volume/i3ce2023_datathon/youtube_images/Frames'
    text_folder = r'/media/leyang/New Volume/i3ce2023_datathon/youtube_images/auto_labels'
    visualize_folder = r'/media/leyang/New Volume/i3ce2023_datathon/youtube_images/visualize'
    start_time = time.time()
    for folder_idx, (root, dirs, files) in enumerate(os.walk(image_folder)):
        # search through the folder and subfolders for all images
        folder_length = folder_idx
    print(f'Found {folder_length} folders')
    for folder_idx, (root, dirs, files) in enumerate(os.walk(image_folder)):
        file_length = len(files)
        for file_idx, file in enumerate(files):
            if file_idx >5:
                visualize = False
            else:
                visualize = True
            timestamp = time.time()
            if file.endswith('.jpg'):
                image_name = os.path.join(root, file)
                txt_name = image_name.replace('.jpg', '.txt').replace(image_folder, text_folder)
                # check if txt file exists
                if os.path.exists(txt_name):
                    print(f'Folder: {folder_idx}/{folder_length} | file: {file_idx}/{file_length} | {txt_name} exists, skipping', end='\r')
                    continue

                '''
                GroundDino
                '''

                image_pil, image = load_image(image_name)
                # run grounding dino model
                boxes_filt, scores, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, device=device
                )

                size = image_pil.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()
                # use NMS to handle overlapped boxes
                # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
                nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
                boxes_filt = boxes_filt[nms_idx]
                pred_phrases = [pred_phrases[idx] for idx in nms_idx]
                # print(f"After NMS: {boxes_filt.shape[0]} boxes")

                if visualize:
                    # draw output image
                    image = cv2.imread(image_name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    for box, label in zip(boxes_filt, pred_phrases):
                        show_box(box.numpy(), plt.gca(), label)

                    plt.axis('off')

                    vis_name = image_name.replace(image_folder, visualize_folder)
                    # create folder if it doesn't exist
                    if not os.path.exists(os.path.dirname(vis_name)):
                        os.makedirs(os.path.dirname(vis_name))
                    plt.savefig(vis_name,
                        bbox_inches="tight", dpi=300, pad_inches=0.0
                    )


                ouput_text = format_bbox(boxes_filt, pred_phrases, classes)
                # create folder if it doesn't exist
                if not os.path.exists(os.path.dirname(txt_name)):
                    os.makedirs(os.path.dirname(txt_name))
                with open(txt_name, 'w') as f:
                    f.write(ouput_text)
                    f.close()
                this_time = time.time() - timestamp
                cum_time = time.time() - start_time
                print(f'Folder: {folder_idx}/{folder_length} | file: {file_idx}/{file_length}| time: {this_time:.2f}s | cum time: {datetime.timedelta(seconds=cum_time)} | {txt_name} created ', end='\r')


