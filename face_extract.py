from ultralytics import YOLO
from PIL import Image


if __name__ == '__main__':
# Detection
    img = 'data/wider_yolo_format/train/images/9_Press_Conference_Press_Conference_9_925.jpg'
    model = 'weight/face_detector/weights/best.pt'
    model = YOLO(model)
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # from PIL
    im1 = Image.open(img)
    results = model.predict(source=[img])
    print(f'len(results) : {len(results)}')
    for result in results:
        
    
    # results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

    # results = model.predict(source=im1, save=True)  # save plotted images

    # from ndarray
    # im2 = cv2.imread("bus.jpg")
    # results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

    # from list of PIL/ndarray
    # results = model.predict(source=[im1])

'''
# Tracking for Video
    # Load a model
    model = YOLO('yolov8n.pt')  # load an official detection model
    model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
    model = YOLO('path/to/best.pt')  # load a custom model

    # Track with the model
    results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
    results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
'''