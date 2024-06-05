from ultralytics import YOLO
from PIL import Image
import numpy as np
from deepfake_detector import Recce
import torch
from torchvision import transforms as T
import warnings
warnings.filterwarnings(action='default')

transform = T.Compose([T.ToTensor(), T.Resize(299)])

if __name__ == '__main__':
    # Loading face detector
    face_detector = 'face_detector.pt'
    face_detector = YOLO(face_detector)
    
    # Loading deepfake detector
    deepfake_detector = Recce(num_classes=1)
    # before_weight = torch.load('deepfake_detector.pt')['state_dict']
    # from collections import OrderedDict
    # new_weight = OrderedDict()
    # for key, val in before_weight.items():
    #     new_weight[key.replace('model.', '')] = val
    # deepfake_detector.load_state_dict(new_weight, strict=True)
    deepfake_detector.load_state_dict(torch.load('q_deepfake_detector.pt'), strict=True)
    deepfake_detector.cuda()
    deepfake_detector.eval()
    
    # Loading input
    img = Image.open('data/150000_10032.png')  # Fake Image
    # img = Image.open('data/1031.png')  # Real Image
    
    # Inference
    results = face_detector.predict(source=img, device='cuda:0', iou=0.01)
    img = np.array(img)
    boxes = results[0].boxes
    preds = list()
    
    for box in boxes:
        x_l, y_l, x_r, y_r = list(map(int, np.array(box.xyxy.cpu().squeeze())))
        cropped_img = img[y_l:y_r, x_l:x_r, :]
        with torch.no_grad():
            cropped_img = transform(cropped_img)
            cropped_img = cropped_img.unsqueeze(0).cuda()
            pred = deepfake_detector(cropped_img)
            pred = torch.sigmoid(pred.cpu()).item()
            preds.append(pred)
    
    preds = np.array(preds)
    pred = np.max(preds)
    flag = "Real" if pred > 0.5 else "Fake"
    print(flag)


# Tracking for Video
# results = model.track(source="https://youtu.be/LNwODJXcvt4", device='cuda:0', show=True, save_crop=True)
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
