from ultralytics import YOLO
def model_training():
  model=YOLO("yolov8n.pt")
  model.train(
      data="/content/drive/MyDrive/ppe_detection_classification/data.yaml",
      epochs=300,
      imgsz=640,
      batch=8,
      device=0,
      name="ppe_kits_detection",
      project="/content/drive/MyDrive/ppe_detection_classification/runs"
  )
model_training() 
