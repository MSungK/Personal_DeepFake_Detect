from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
from deepfake_detector import Recce
import torch
from torchvision import transforms as T
import tkinter as tk
from tkinter import filedialog, messagebox
from os import path as osp
import tkinter.font as tkFont

# Setting Preprocessing
transform = T.Compose([T.ToTensor(), T.Resize(299)])

# Loading face detector
face_detector = 'face_detector.pt'
face_detector = YOLO(face_detector)

# Loading deepfake detector
deepfake_detector = Recce(num_classes=1)
deepfake_detector.load_state_dict(torch.load('q_deepfake_detector.pt'), strict=True)
deepfake_detector.cuda()
deepfake_detector.eval()

# 파일 선택 함수
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path and osp.splitext(file_path)[-1] in ['.jpg', '.JPG', '.png', '.PNG']:
        file_path_var.set(file_path)
        display_image(file_path)

# 이미지 표시 함수
def display_image(file_path):
    img = Image.open(file_path)
    img = img.resize((400, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# 실행 버튼 함수
def run_model():
    file_path = file_path_var.get()
    if not file_path:
        messagebox.showerror("Error", "Please select a file first.")
        return
    
    # Loading Input
    img = Image.open(file_path)
    
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
    
    result_var.set(f"Result: {flag}")

if __name__ == '__main__':
    # Tkinter GUI 설정
    root = tk.Tk()
    root.title("Deepfake Detector")
    
    # 윈도우 사이즈 설정
    root.geometry("800x600")
    root.configure(bg='white')

    # 파일 경로 변수
    file_path_var = tk.StringVar()
    result_var = tk.StringVar()

    button_font = ('Helvetica', 16)
    result_font = tkFont.Font(family='Helvetica', size=64)  # 결과 레이블의 폰트 크기 64로 설정

    # 파일 선택 버튼
    select_button = tk.Button(root, text="Select File", command=select_file, bg='white', font=button_font)
    select_button.pack(pady=20, padx=10)

    # 파일 경로 표시 레이블
    file_path_label = tk.Label(root, textvariable=file_path_var, bg='white', font=button_font)
    file_path_label.pack(pady=20, padx=10)

    # 이미지 표시 레이블
    image_label = tk.Label(root, bg='white')
    image_label.pack(pady=10)

    # 실행 버튼
    run_button = tk.Button(root, text="Run Model", command=run_model, bg='white', font=button_font)
    run_button.pack(pady=20, padx=10)

    # 결과 표시 레이블
    result_label = tk.Label(root, textvariable=result_var, bg='white', font=result_font)
    result_label.pack(pady=20, padx=10)

    # GUI 실행
    root.mainloop()
