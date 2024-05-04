from ultralytics import YOLO
from PIL import Image


if __name__ == '__main__':
    img = 'data/wider_yolo_format/train/images/9_Press_Conference_Press_Conference_9_925.jpg'
    model = 'weight/face_detector/weights/best.pt'
    model = YOLO(model)

# Detection
    im1 = Image.open(img)
    # results = model.predict(source=[img], device='cuda:0', save_crop=True)


# Tracking for Video
    results = model.track(source="https://youtu.be/LNwODJXcvt4", device='cuda:0', show=True, save_crop=True)
    # results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
