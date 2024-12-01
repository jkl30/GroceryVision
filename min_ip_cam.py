import cv2

# Open a connection to the camera feed
cap = cv2.VideoCapture("http://192.168.0.29:8080/video")

if not cap.isOpened():
    print("Error: Unable to access the camera feed.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to read frame.")
        break
    
    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
