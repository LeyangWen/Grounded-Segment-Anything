import cv2
import numpy as np
import supervision as sv
from typing import List
import os
import time

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam_predictor = SamPredictor(sam)


# Predict classes and hyper-param for GroundingDINO
CLASSES = ["Crane", "Excavator", "Bulldozer", "Scraper", "Truck", "Worker"]
CLASSES = ['Crane', 'Excavator', 'Bulldozer', 'Dump Truck', 'Worker'] #Crane. Excavator. Bulldozer. Dump Truck. Worker.
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

if True:
    SOURCE_DIR = "/home/leyang/Documents/construction_site_images"
    OUTPUT_DIR = "/home/leyang/Documents/construction_site_images_output/GroundingDINO"
    USE_SAM = False
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # find all images in the directory
    images = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                images.append(os.path.join(root, file))

    image_num = len(images)
    for i, SOURCE_IMAGE_PATH in enumerate(images):
        # load image
        start_time = time.time()
        print(f"Image {i}/{image_num} {SOURCE_IMAGE_PATH} processing...")
        image = cv2.imread(SOURCE_IMAGE_PATH)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

        # NMS post process
        before_nms = len(detections.xyxy)
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # save the annotated grounding dino image
        output_file_sam_name = os.path.join(OUTPUT_DIR, 'SAM'+SOURCE_IMAGE_PATH.split('/')[-1])
        output_file_dino_name = os.path.join(OUTPUT_DIR, 'DINO'+SOURCE_IMAGE_PATH.split('/')[-1])
        cv2.imwrite(output_file_dino_name, annotated_frame)

        print(f"After NMS: {before_nms} --> {len(detections.xyxy)} boxes")
        dino_time = time.time()
        print(f"GroundingDINO time: {dino_time - start_time}")
        if USE_SAM:
            # Prompting SAM with detected boxes
            def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
                sam_predictor.set_image(image)
                result_masks = []
                for box in xyxy:
                    masks, scores, logits = sam_predictor.predict(
                        box=box,
                        multimask_output=True
                    )
                    index = np.argmax(scores)
                    result_masks.append(masks[index])
                return np.array(result_masks)


            # convert detections to masks
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

            # annotate image with detections
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            labels = [
                f"{CLASSES[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _
                in detections]
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            # save the annotated grounded-sam image

            cv2.imwrite(output_file_sam_name, annotated_image)
            sam_time = time.time()
            print(f"SAM time: {sam_time - dino_time}")
