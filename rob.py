import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("/Users/coldreign/robotStasik/runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 820)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

square_size = 25

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка получения кадра")
        break
    
    results = model.predict(source=frame, conf=0.35, verbose=False)
    frame = results[0].plot()

    object_positions = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = model.names[class_id]
            x_c = (x1 + x2) / 2
            y_c = (y1 + y2) / 2
            if label != "bottle":
                object_positions[label] = (x_c, y_c)
            else:
                object_positions[label] = (x2 -(x2-x1), y2)

    if all(label in object_positions for label in ["0", "x", "y", "bottle"]):
        x_0, y_0 = object_positions["0"]
        x_x, y_x = object_positions["x"]
        x_y, y_y = object_positions["y"]
        x_b, y_b = object_positions["bottle"]

        x_xy = x_x + (x_y - x_0)
        y_xy = y_x + (y_y - y_0)

        pts_pixel = np.array([
            [x_0, y_0],
            [x_x, y_x],
            [x_y, y_y],
            [x_xy, y_xy]
        ], dtype=np.int)

        pts_real = np.array([
            [0, 0],
            [square_size, 0],
            [0, square_size],
            [square_size, square_size]
        ], dtype=np.float32)

        homography_matrix, _ = cv2.findHomography(pts_pixel, pts_real)

        bottle_pixel = np.array([[x_b, y_b]], dtype=np.float32)
        bottle_real = cv2.perspectiveTransform(bottle_pixel.reshape(-1, 1, 2), homography_matrix)


        x_real, y_real = bottle_real[0][0]

        cv2.putText(frame, f"Bottle: X={x_real:.1f}cm, Y={y_real:.1f}cm", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        x11 = x_real
        y11 = y_real

        x01 = - 35.5
        y01 = 89

        x21 = x01 + x11
        y21 = y01 - y11

        print(x21, y21, x_real, y_real)
        
    cv2.imshow("Webcam YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
