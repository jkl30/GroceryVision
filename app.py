# app.py

from flask import Flask, render_template, Response
import logging
from logger_config import configure_logging
import json
import time
import cv2
import base64
from threading import Lock
from object_detection import ObjectDetection
from ocr_pp import OCR
from frame_annotator import FrameAnnotator, AnnotationMode

app = Flask(__name__)
logger = logging.getLogger(__name__,)

# Configure logging
configure_logging()  # Call the logging configuration function

# Global variables for camera management
camera_lock = Lock()
#camera_url = "http://192.168.0.29:8080/video"
camera_url=0
reconnect_timeout = 5  # seconds between reconnection attempts

def get_camera_connection():
    """
    Establish connection to the IP camera with error handling and reconnection logic.
    """
    while True:
        with camera_lock:
            cap = cv2.VideoCapture(camera_url)
            if not cap.isOpened():
                logger.error(f"Failed to connect to camera at {camera_url}")
                time.sleep(reconnect_timeout)
                continue
            
            # Set camera buffer size to 1
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Disable internal frame buffer
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            return cap

def flush_camera_buffer(cap):
    """
    Flush the camera buffer by reading frames until we get the latest one.
    """
    for _ in range(1):  # Read several frames to ensure we've cleared any buffered frames
        cap.grab()  # Use grab() instead of read() for faster buffer clearing
    
    ret, frame = cap.retrieve()  # Only retrieve the last grabbed frame
    return ret, frame

def convert_frame_to_base64(frame):
    """
    Convert an OpenCV frame to a base64 encoded string.
    """
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    except Exception as e:
        logger.error(f"Error converting frame to base64: {str(e)}")
        return None

def generate_frames():
    """
    Generator function to yield video frames with error handling and reconnection logic.
    """
    cap = None
    
    # Initialize processing components
    object_detector = ObjectDetection('yolo11n.pt')
    ocr_reader = OCR()
    frame_annotator = FrameAnnotator()  # Initialize without parameters as per the class definition
    
    while True:
        try:
            if cap is None or not cap.isOpened():
                cap = get_camera_connection()
                if cap is None:
                    time.sleep(reconnect_timeout)
                    continue

            # Flush buffer and get latest frame
            ret, frame = flush_camera_buffer(cap)

            if not ret:
                logger.warning("Failed to read frame, attempting to reconnect...")
                if cap is not None:
                    cap.release()
                cap = None
                time.sleep(reconnect_timeout)
                continue
            
            # Object Detection
            yolo_results = object_detector.detect_objects(frame)
            
            # OCR Detection
            ocr_results = ocr_reader.detect_text(frame)
            
            # Annotate Frame with the correct method signature
            # The method now returns both the annotated frame and the closest tracked object
            annotated_frame, closest_object = frame_annotator.annotate_frame(
                frame=frame,
                yolo_results=yolo_results,
                ocr_results=ocr_results,
                filter_tracked=True,  # Set to True to only show tracked objects
                annotation_mode=AnnotationMode.DOT_AND_TEXT
            )
            
            # Convert to base64
            frame_base64 = convert_frame_to_base64(annotated_frame)
            if frame_base64 is None:
                continue

            # Create response with additional data
            response_data = {
                'frame': frame_base64,
                'timestamp': time.time(),
                'closest_tracked_object': closest_object  # Include the closest tracked object in response
            }

            response = json.dumps(response_data)
            print('closest', closest_object)
            yield f"data: {response}\n\n"

            # Small delay to prevent overwhelming the system
            time.sleep(0.01)  # 10ms delay between frames

        except Exception as e:
            logger.error(f"Error in frame generation: {str(e)}")
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(reconnect_timeout)

@app.route('/')
def index():
    """
    Route for the main page.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Route for the video feed using Server-Sent Events.
    """
    return Response(generate_frames(),
                   mimetype='text/event-stream')

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Disable Flask's debugger reloader in production
    app.run(debug=False, host='192.168.0.30', threaded=True)
    #app.run(debug=False, host='10.122.152.129', threaded=True)
    #app.run(debug=False, threaded=True)