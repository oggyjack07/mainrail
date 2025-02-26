from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
import open_clip
from PIL import Image

# ✅ 1️⃣ FastAPI app initialize karo
app = FastAPI()

# ✅ 2️⃣ Model load karo
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# ✅ 3️⃣ Classes define karo
classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush", "don't know", "alien",
    "smart phone", "smart watch", "hand", "500 rupees"
]

# ✅ 4️⃣ Prediction function
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")  # Open image
        image_input = preprocess(image).unsqueeze(0).to(device)  # Preprocess
        text_inputs = tokenizer(classes).to(device)  # Tokenize classes

        # Model se prediction lo
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).softmax(dim=-1)

        # Sabse highest similarity wali class nikalo
        pred_class = classes[similarity.argmax().item()]
        return {"prediction": pred_class}

    except Exception as e:
        return {"error": str(e)}

# ✅ 5️⃣ FastAPI server run karo (for Railway)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
