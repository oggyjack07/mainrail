from fastapi import FastAPI, File, UploadFile
import torch
import open_clip
from PIL import Image
import io
import gc

app = FastAPI()

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Class Labels
classes = ["person", "car", "dog", "cat", "phone", "laptop", "hand", "500 rupees"]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Model Prediction
        text_inputs = tokenizer(classes).to(device)
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        pred_class = classes[similarity.argmax().item()]

        return {"prediction": pred_class}

    except RuntimeError as e:
        return {"error": str(e)}

    finally:
        model.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
