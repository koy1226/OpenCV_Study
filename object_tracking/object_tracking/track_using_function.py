import cv2
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

# Load image files
image_files = []
for i in range(588):
    file_number = str(i).zfill(3)
    image_file = f"xing/xing{file_number}.png"
    image_files.append(image_file)

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2()

# Initialize background model using the first frame
previous_frame = cv2.imread("xing/init.png")
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY).astype("float")

output_file = "output.avi"
frame_width = previous_frame.shape[1]
frame_height = previous_frame.shape[0]
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_file, fourcc, 10.0, (frame_width, frame_height))

for image_file in image_files:
    # Read image file
    frame = cv2.imread(image_file)
    height, width, _ = frame.shape

    # 1. Object Detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 300:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Write frame to output video
    out.write(frame)

    key =  cv2.waitKey(100) & 0xff
    if key == ord('q'):
        break

out.release()
cv2.destroyAllWindows()