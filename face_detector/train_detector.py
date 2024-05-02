from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='../data/wider_yolo_format/data.yaml',
                      epochs=100,
                      imgsz=640,
                      patience=100,
                      batch=32,
                      save=True,
                      device=0,
                      workers=8,
                      pretrained=True,
                      optimizer='SGD',
                      lr0=0.01,
                      weight_decay=5e-4,
                      cos_lr=True,
                      plots=True,)
