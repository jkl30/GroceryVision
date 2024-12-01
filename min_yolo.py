import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv11n model
model = YOLO("yolo11x.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Run inference
        results = model(frame)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()  # Get the annotated frame

        # Display the annotated frame
        cv2.imshow("YOLOv11 Webcam Feed", annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
