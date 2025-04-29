from ultralytics import YOLO

model = YOLO("/Users/coldreign/robo3/best.pt")

model.export(format="ncnn")

