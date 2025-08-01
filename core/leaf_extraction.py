import torch
import torchvision
import numpy as np
import os
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T

# Load Mask R-CNN model and modify heads for 2-class detection
def get_model(num_classes=2):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# Run inference on PIL image and extract leaf masks
def run_inference_from_pil(pil_image, model_path="D:/projects/AgriSavant/ETL PEST DETECTION CODE MODULARITY/pest_etl_app/models/leaf_maskrcnn.pth", output_dir="data/outputs", conf_thresh=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    os.makedirs(output_dir, exist_ok=True)

    image = pil_image.convert("RGBA")
    img_tensor = T.ToTensor()(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)[0]

    original = np.array(image)
    output_paths = []
    mask_count = 0

    for i, (score, mask) in enumerate(zip(output["scores"], output["masks"])):
        if score < conf_thresh:
            continue

        m = mask.squeeze().cpu().numpy()
        binary_mask = (m > 0.5).astype(np.uint8)

        extracted = original.copy()
        extracted[..., 3] = binary_mask * 255
        extracted[binary_mask == 0] = [0, 0, 0, 0]

        out_path = os.path.join(output_dir, f"leaf_extracted_{i}.png")
        Image.fromarray(extracted).save(out_path)
        output_paths.append(out_path)
        mask_count += 1

    if mask_count == 0:
        return [], "⚠️ No leaves detected above confidence threshold."

    return output_paths, None
