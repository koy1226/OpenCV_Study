import cv2
from tracker import *
import numpy as np

# Create tracker object
tracker = EuclideanDistTracker()

# Background subtraction parameter
learning_rate = 0.001

# Load image files
image_files = []
for i in range(588):
    file_number = str(i).zfill(3)
    image_file = f"xing/xing{file_number}.png"
    image_files.append(image_file)

# Initialize background model using the first frame
previous_frame = cv2.imread("xing/init.png")
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY).astype("float")

for image_file in image_files:
    # Read image file
    frame = cv2.imread(image_file)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between background and frame
    frame_diff = cv2.absdiff(gray, cv2.convertScaleAbs(previous_gray))
    #이미지 처리 과정에서 발생하는 부동 소수점 값을 8비트 부호 없는 정수 형식으로 변환하기 위해
    
    # Apply thresholding to obtain foreground mask
    block_size = 77
    C = -3
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

    # Apply morphological operation to remove noise
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 1. Find contours of objects in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Object detection from contours
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            #if w < 90 and h < 90:
            detections.append([x, y, w, h])
    
    # 3. Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 4. Show image
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(100) & 0xff
    if key == ord('q'):
        break

    # Update background model
    cv2.accumulateWeighted(gray, previous_gray, learning_rate)

cv2.destroyAllWindows()