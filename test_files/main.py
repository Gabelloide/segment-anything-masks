from segment_anything import SamPredictor, sam_model_registry
import cv2, os, torch
import numpy as np
from pathlib import Path
from groundingdino.util.inference import load_model, predict
from groundingdino.util.misc import nested_tensor_from_tensor_list
from torchvision import transforms


# Charger le modèle SAM
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "sam_vit_b.pth"
GROUNDINGDINO_CHECKPOINT = "groundingdino_swint_ogc.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialiser SAM et GroundingDINO
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)
predictor = SamPredictor(sam)

groundingdino = load_model(
    model_checkpoint_path=GROUNDINGDINO_CHECKPOINT, 
    model_config_path="GroundingDINO_SwinT_OGC.py"
).to(DEVICE)

FOLDER_PATH = "C:\\Users\\olivi\\Documents\\Olivier\\Mondial_Auto\\R17\\inputs\\images\\cam1"
images_paths = [os.path.join(FOLDER_PATH, path) for path in os.listdir(FOLDER_PATH) if path.lower().endswith(('.png', '.jpg'))]

SAVE_PATH = "C:\\Users\\olivi\\Documents\\Olivier\\Mondial_Auto\\R17\\inputs\\masks\\cam1"

# Transformation de l'image pour GroundingDINO
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for path in images_paths:
    image_name = Path(path).stem

    # Charger et convertir l'image
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Vérifier que l'image a 3 canaux
    if image_rgb.shape[2] != 3:
        raise ValueError(f"L'image {path} n'a pas 3 canaux")

    # Appliquer la transformation et créer un NestedTensor pour GroundingDINO
    image_tensor = transform(image_rgb).to(DEVICE)

    # Utiliser GroundingDINO pour détecter les bounding boxes
    caption = "car"  # Légende pour détecter les voitures
    box_threshold = 0.3  # Ajuster selon les besoins
    text_threshold = 0.25  # Ajuster selon les besoins

    # Prédiction des boîtes avec GroundingDINO
    boxes, scores, labels = predict(
        model=groundingdino, 
        image=image_tensor, 
        caption=caption, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold, 
        device=DEVICE
    )

    # Si une boîte de délimitation est détectée
    if len(boxes) > 0:
        bbox = boxes[0]
        x1, y1, x2, y2 = bbox.int().tolist()  # Convertir les coordonnées en entiers

        # Vérifier que les coordonnées sont valides et dans les limites de l'image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_rgb.shape[1], x2), min(image_rgb.shape[0], y2)

        if x1 >= x2 or y1 >= y2:
            print(f"Invalid bounding box for {image_name}: {bbox}")
            continue

        # Extraire la région d'intérêt (ROI) de l'image
        roi = image_rgb[y1:y2, x1:x2]

        # Vérifier que la ROI est assez grande pour SAM
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            print(f"ROI trop petite pour {image_name}, bbox: {bbox}")
            continue

        # Utiliser SAM pour segmenter la voiture dans la ROI
        predictor.set_image(roi)

        height, width, _ = roi.shape
        center_x, center_y = width // 2, height // 2
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])

        # Prédiction du masque avec SAM
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Sélectionner le meilleur masque
        best_mask_index = np.argmax(scores)
        best_mask = masks[best_mask_index]
        best_score = scores[best_mask_index]

        # Sauvegarder le meilleur masque sur le disque
        mask_image = (best_mask * 255).astype(np.uint8)
        cv2.imwrite(f"{SAVE_PATH}\\{image_name}.png", mask_image)
        print(f"Saved {SAVE_PATH}\\{image_name}.png with score {best_score:.4f}")
    else:
        print(f"No bounding box found for {image_name}")
