from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
from pyngrok import ngrok
import uvicorn, shutil, os

# setup
app = FastAPI(title="PPE Detection API", version="2.0")
model = YOLO("/content/drive/MyDrive/ppe_detection_classification/runs/ppe_kits_detection/weights/best.pt")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    results = model.predict(source=input_path, conf=0.5, save=True, project=UPLOAD_DIR, name="results", exist_ok=True)
    result_img_path = os.path.join(UPLOAD_DIR, "results", file.filename)
    return FileResponse(result_img_path, media_type="image/jpeg")

@app.get("/")
def home():
    return {"message": "PPE Detection API (with image output) is running!"}


if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print("Public URL:", public_url)
    uvicorn.run(app, host="0.0.0.0", port=8000)
