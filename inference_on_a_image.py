import argparse
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import logging
import warnings
from tqdm import tqdm

from segment_anything import SamPredictor, sam_model_registry

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignore warnings
warnings.filterwarnings("ignore")

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def get_box_from_image(tgt):
  """Returns the box coordinates from the target dictionary that was computed by GroundingDINO."""
  H, W = tgt["size"]
  boxes = tgt["boxes"]
  labels = tgt["labels"]
  assert len(boxes) == len(labels), "boxes and labels must have same length"

  # all boxes
  box_coords = []

  # Get boxes coords
  for box in boxes:
      # From [0..1] to [0..W, 0..H]
      box = box * torch.Tensor([W, H, W, H])
      # From xywh to xyxy
      box[:2] -= box[2:] / 2
      box[2:] += box[:2]
      
      x0, y0, x1, y1 = box.int().tolist()  # To integers
      box_coords.append([x0, y0, x1, y1])
  return box_coords


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


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    logging.info(load_res)
    _ = model.eval()
    return model


def create_sam_mask(box_coordinates, image_path, sam_predictor, multiple_boxes=False):
    """Computes SAM masks for a given image and box coordinates.
    If multiple boxes are provided, the function will return several masks and scores.
    SAM is creating 3 masks for each box.
    If multiple_boxes is True, will combine all boxes in one image to compute only 3 masks."""
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image)

    if not multiple_boxes:
        # Future results
        results = {}
        
        # Process each box
        for idx, box in enumerate(box_coordinates):
            input_box = np.array([box])  # Ensure the box is in a list
            masks, scores, logits = sam_predictor.predict(
                box=input_box,
                multimask_output=True,
            )
            
            # Dictionary for the results
            results[idx] = {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
        return results
    else:
        input_boxes = torch.tensor(box_coordinates, device=DEVICE)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, scores, logits = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )
        return masks, scores, logits


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


def fuse_masks(masks):
    """Fuses multiple masks into a single mask."""
    fused_mask = np.zeros_like(masks[0][0], dtype=np.uint8)

    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            mask = masks[i][j]
            fused_mask = np.maximum(fused_mask, mask)

    return fused_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", "-d", type=str, required=True, help="path to image directory")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help="The positions of start and end positions of phrases of interest.")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!", default=False)
    parser.add_argument("--config_file", "-c", type=str, default="GroundingDINO_SwinT_OGC.py", help="path to the model config file")
    parser.add_argument("--checkpoint_path", "-m", type=str, default="groundingdino_swint_ogc.pth", help="path to the model checkpoint")
    parser.add_argument("--split_masks", "-s", action="store_true", default=False, help="Create one mask per detected subject in the image")
    parser.add_argument("--sam_checkpoint_path", "-g", type=str, default="sam_vit_h.pth", help="path to the sam model checkpoint")
    parser.add_argument("--sam_checkpoint_type", "-y", type=str, default="vit_h", help="type of the sam model checkpoint : [vit_b, vit_h, vit_l]")

    args = parser.parse_args()
    
    # cfg
    config_file = Path(args.config_file)
    checkpoint_path = Path(args.checkpoint_path)
    image_dir = Path(args.image_dir)
    text_prompt = args.text_prompt
    output_dir = Path(args.output_dir)
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    sam_checkpoint_path = Path(args.sam_checkpoint_path)
    sam_checkpoint_type = args.sam_checkpoint_type

    # make dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the provided path is a directory
    if not image_dir.is_dir():
        raise ValueError(f"The provided path {image_dir} is not a directory.")

    # List all image files in the directory
    image_files = [f.name for f in image_dir.iterdir() if f.is_file()]

    sam = sam_model_registry[sam_checkpoint_type](checkpoint=sam_checkpoint_path.as_posix()).to(DEVICE)
    predictor = SamPredictor(sam)

    for i, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images"):
        try:
            image_path = image_dir / image_file

            # load image
            image_pil, image = load_image(image_path)
            # load model
            model = load_model(config_file.as_posix(), checkpoint_path.as_posix(), cpu_only=args.cpu_only)

            # set the text_threshold to None if token_spans is set.
            if token_spans is not None:
                text_threshold = None
                logging.info("Using token_spans. Set the text_threshold to None.")

            # run model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=eval(f"{token_spans}")
            )
            
            # visualize pred
            size = image_pil.size
            pred_dict = {
                "boxes": boxes_filt,
                "size": [size[1], size[0]],  # H,W
                "labels": pred_phrases,
            }

            box_coords = get_box_from_image(pred_dict)

            # Creating one mask for each box if args.split_masks is True
            if args.split_masks:
                results = create_sam_mask(box_coords, image_path, predictor, multiple_boxes=False)

                # For each box, there are three masks, we will save the best one for each one.
                for idx, result in results.items():
                    masks = result['masks']
                    scores = result['scores']
                    
                    # Skip if no masks or scores found
                    if masks.size == 0 or scores.size == 0:
                        logging.warning(f"No masks or scores found for box {idx}. Skipping.")
                        continue

                    # Find the best mask
                    best_mask_index = np.argmax(scores)
                    best_mask = masks[best_mask_index]
                    best_score = scores[best_mask_index]

                    # Save the mask
                    mask_image = (best_mask * 255).astype(np.uint8)
                    mask_filename = output_dir / f"{image_file}_box_{idx}.png"  # Unique name for each box
                    cv2.imwrite(mask_filename.as_posix(), mask_image)
                    logging.info(f"Saved {mask_filename} with score {best_score:.4f}")
            else:
                masks, scores, logits = create_sam_mask(box_coords, image_path, predictor, multiple_boxes=True)

                # In this case, results are on GPU, getting them back
                masks = masks.cpu().numpy()
                scores = scores.cpu().numpy()
                logits = logits.cpu().numpy()
                
                # Skip if no masks or scores found
                if masks.size == 0 or scores.size == 0:
                    logging.warning(f"No masks or scores found for the provided boxes. Skipping.")
                    continue

                # Fuse all masks into one
                fused_mask = fuse_masks(masks)

                # Save the fused mask
                mask_image = (fused_mask * 255).astype(np.uint8)
                mask_filename = output_dir / f"{image_file}.png"
                cv2.imwrite(mask_filename.as_posix(), mask_image)
                logging.info(f"Saved fused mask {mask_filename}")
        except Exception as e:
            logging.error(f"An error occurred while processing {image_file}: {e}")
            continue  # Skip to the next image in case of an error