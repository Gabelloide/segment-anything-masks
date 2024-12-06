from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth")
IMAGE_PATH = "C:\\Users\\se55980\\Desktop\\Mondial_Auto\\Twingo\\cam1\\frame_0003.jpg"
TEXT_PROMPT = "car . vehicle ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

print(boxes)
print(logits)
print(phrases)


annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotated_image.jpg", annotated_frame)