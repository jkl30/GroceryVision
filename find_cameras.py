import cv2

def test_cameras():
    # Maximum number of possible camera indices to test
    max_cameras = 10
    
    for camera_index in range(max_cameras):
        # Try to open the camera with index `camera_index`
        cap = cv2.VideoCapture(camera_index,cv2.CAP_DSHOW)
        
        if cap.isOpened():  # If the camera is available
            print(f"Camera {camera_index} is available.")
            
            # Display the camera feed
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to grab frame from camera {camera_index}")
                    break
                
                # Show the frame
                cv2.imshow(f"Camera {camera_index}", frame)
                
                # Exit the feed when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Release the camera after testing
            cap.release()
        else:
            print(f"Camera {camera_index} is not available.")
    
    cv2.destroyAllWindows()

# Run the camera test
test_cameras()
