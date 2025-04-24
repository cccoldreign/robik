from forvova import coords

path = "/Users/coldreign/robo3/best.pt"
camera_vision = coords(path)

x, y = camera_vision.coordinates()
print(f"В притирочку: X={x:.2f}cm, Y={y:.2f}cm")