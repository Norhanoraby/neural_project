"""
AWS Lambda handler for pneumonia classification.

Runs inside a container image (PyTorch + transformers are far too large for
a normal Lambda zip). The model is loaded once per container — outside the
handler — so warm invocations are fast. Exposed via a Lambda Function URL,
called by the web UI (index.html).
"""

import base64
import io
import json
import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
# If you bake your fine-tuned weights into the image (see Dockerfile), they
# are loaded automatically; otherwise the base pretrained model is used.
CHECKPOINT = f"{os.environ.get('LAMBDA_TASK_ROOT', '.')}/best_cxrpretrained_vit_3000_clean.pth"

device = torch.device("cpu")

# --- loaded once per container start ---
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device).eval()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Content-Type": "application/json",
}


def handler(event, context):
    # Answer CORS preflight requests from the browser.
    method = (
        event.get("requestContext", {}).get("http", {}).get("method", "")
    )
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    try:
        body = event.get("body", "{}")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        data = json.loads(body)

        image_b64 = data["image_base64"]
        # Browsers send "data:image/png;base64,...." — strip the prefix.
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=1)[0, 1].item()

        result = {
            "prediction": "Pneumonia" if prob >= 0.5 else "Non-Pneumonia",
            "pneumonia_probability": round(prob, 4),
        }
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(result)}

    except Exception as exc:  # noqa: BLE001
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(exc)}),
        }
