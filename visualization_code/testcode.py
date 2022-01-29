import cv2
from PIL import Image

cap = cv2.VideoCapture(0)   # /dev/video0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cv2.imshow('openCV',frame)
    c = cv2.waitKey(1) # ASCII 'Esc' value
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()