import cv2
from tracker import *
import numpy as np

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    # 입력 영상의 컬러 영상 변환
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Draw lines
    cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)
    
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)
        
    return vis

# Create tracker object
tracker = EuclideanDistTracker()

# Load image files
image_files = []
for i in range(588):
    file_number = str(i).zfill(3)
    image_file = f"xing/xing{file_number}.png"
    image_files.append(image_file)

# Initialize previous frame
previous_frame = cv2.imread("xing/init.png")
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

for image_file in image_files:
    # Read image file
    frame = cv2.imread(image_file)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    optical_flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Extract flow magnitude and angle
    magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

    # Threshold the magnitude to obtain foreground mask
    threshold = 1
    mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)[1]

    # Convert mask to 8-bit single channel
    mask = mask.astype(np.uint8)

    # Apply morphological operation to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    #mask = cv2.erode(mask, kernel, iterations=1)

    # Find contours of objects in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Object detection from contours
    detections = []
    for cnt in contours:
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 300 and area < 10000:
            detections.append([x, y, w, h])
    
    # Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Show image
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow('Frame2', draw_flow(gray, optical_flow))

    key = cv2.waitKey(100) & 0xff
    if key == ord('q'):
        break

    # Update previous frame
    previous_gray = gray.copy()

cv2.destroyAllWindows()
