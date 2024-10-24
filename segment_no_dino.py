from segment_anything import SamPredictor, sam_model_registry
import cv2, os, torch
import numpy as np
from pathlib import Path

# Charger le modèle
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_vit_b.pth"

IMAGE_PATH = "frame_0454.jpg"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
predictor = SamPredictor(sam)

FOLDER_PATH = "C:\\Users\\se55980\\Desktop\\Mondial_Auto\\Twingo\\cam1"
images_paths = [os.path.join(FOLDER_PATH, path) for path in os.listdir(FOLDER_PATH) if path.lower().endswith(('.png', '.jpg'))]

SAVE_PATH = "C:\\palette\\apl\\segment_anything\\output"

for path in images_paths:
    image_name = Path(path).stem

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])

    # Convert numpy arrays to tensors to use GPU
    input_point_tensor = torch.tensor(input_point).to(DEVICE)
    input_label_tensor = torch.tensor(input_label).to(DEVICE)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Save all three masks on disk and print their scores
    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     mask_image = (mask * 255).astype(np.uint8)
    #     cv2.imwrite(f"{SAVE_PATH}\\{image_name}_{i+1}.png", mask_image)
    #     print(f"Saved {SAVE_PATH}\\{image_name}_{i+1}.png with score {score:.4f}")

    best_mask_index = np.argmax(scores)
    best_mask = masks[best_mask_index]
    best_score = scores[best_mask_index]

    # Save the best mask on disk
    mask_image = (best_mask * 255).astype(np.uint8)
    cv2.imwrite(f"{SAVE_PATH}\\{image_name}.png", mask_image)
    print(f"Saved {SAVE_PATH}\\{image_name}.png with score {best_score:.4f}")