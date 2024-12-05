import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import logging
import warnings
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

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

# Disable logging for cleaner print (uncomment)
logging.disable(logging.CRITICAL)

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


def load_groundingdino_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    logging.info(load_res)
    _ = model.eval()
    return model


def create_sam_mask(box_coordinates: list, image_path, sam_predictor, multiple_boxes=False):
    """Computes SAM masks for a given image and box coordinates. The box_coordinates list is considered not empty to compute a mask.
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
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_coordinates,
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


def load_models(dino_config_file, dino_checkpoint_path, sam2_checkpoint_path, sam2_config_file, cpu_only=False):
    """Load GroundingDINO and SAM models."""
    model = load_groundingdino_model(dino_config_file, dino_checkpoint_path, cpu_only)
    device = "cuda" if not cpu_only else "cpu"
    print(sam2_config_file)
    sam2_model = build_sam2(sam2_config_file, sam2_checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return model, predictor


def get_boxes(image_path, model, text_prompt, box_threshold, text_threshold=None, token_spans=None):
    """Run GroundingDINO to get the boxes around the subjects in the image."""
    image_pil, image = load_image(image_path)
    image_width, image_height = image_pil.size
    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
      text_threshold = None
      logging.info("Using token_spans. Set the text_threshold to None.")
      
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
    return box_coords



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", "-d", type=str, required=True, help="path to image directory")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!", default=False)
    parser.add_argument("--config_file", "-c", type=str, default="config_files/GroundingDINO_SwinT_OGC.py", help="path to the model config file")
    parser.add_argument("--checkpoint_path", "-m", type=str, default="groundingdino_swint_ogc.pth", help="path to the model checkpoint")
    parser.add_argument("--sam2_checkpoint_path", "-g", type=str, default="sam2.1_hiera_large.pt", help="path to the sam 2 model checkpoint")
    # Hydra module is susceptible so please put your config in the sam2_repo/configs/sam2.1/ directory (see sam2 documentation for more details)
    parser.add_argument("--sam2_config_file", "-y", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="path to the sam2 model config file")

    parser.add_argument("--keepObject", "-k", action="store_true", default=False, help="If specified, the prompted object will be kept and the rest will be blacked out.")
    parser.add_argument("--saveMask", "-s", action="store_true", default=False, help="If specified, will save the mask that was used to black out the object")


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
    sam2_checkpoint_path = Path(args.sam2_checkpoint_path)
    sam2_config_file = args.sam2_config_file

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"

    # make dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the provided path is a directory
    if not image_dir.is_dir():
        raise ValueError(f"The provided path {image_dir} is not a directory.")

    # List all image files in the directory
    image_files = [f.name for f in image_dir.iterdir() if f.is_file()]

    model, predictor = load_models(config_file.as_posix(), checkpoint_path.as_posix(), sam2_checkpoint_path.as_posix(), sam2_config_file, cpu_only=args.cpu_only)

    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            image_path = image_dir / image_file
            # Get the masks
            box_coords = get_boxes(image_path, model, text_prompt, box_threshold, text_threshold, token_spans)

            # Loading the original image for later
            input_image = cv2.imread(image_path.as_posix())

            # If boxes are found, creating masks using SAM
            if len(box_coords) != 0 :
                # ------------- SAM MASK GENERATION ----------------
                masks, scores, logits = create_sam_mask(box_coords, image_path, predictor, multiple_boxes=True)
                
                # Skip if no masks or scores found
                if masks.size == 0 or scores.size == 0:
                    logging.warning(f"No masks or scores found for the provided boxes. Skipping.")
                    continue

                # Fuse all masks into one
                fused_mask = fuse_masks(masks)

                # Save the fused mask as an image object
                mask_image = (fused_mask * 255).astype(np.uint8)

            # No boxes are found, generating a blank mask (nothing to mask)
            else:
                height, width = input_image.shape[:2]
                mask_image = np.zeros((height, width), dtype=np.uint8) * 255

            if not args.keepObject: # If the object is not to be kept, invert the mask so the object is actually white in the mask, not black, resulting in keeping the object only in the final image.
                mask_image = 255 - mask_image
                
            # Apply mask to the image
            mask = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            masked_image = cv2.bitwise_and(input_image, mask)
        
            # Save the masked image
            masked_image_filename = output_dir / f"{Path(image_file).stem}.png"
            cv2.imwrite(masked_image_filename.as_posix(), masked_image)
            logging.info(f"Saved fused image {masked_image_filename}")

            if args.saveMask:
                # Also save the mask
                mask_filename = output_dir / f"{Path(image_file).stem}_mask.png"
                cv2.imwrite(mask_filename.as_posix(), mask_image)
                logging.info(f"Saved associated mask {masked_image_filename}")

        except Exception as e:
            logging.error(f"An error occurred while processing {image_file}: {e}")
            continue  # Skip to the next image in case of an error
