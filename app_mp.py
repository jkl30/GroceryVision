# app_mp.py

from flask import Flask, render_template, Response
import logging
from logger_config import configure_logging
import json
import time
import cv2
import base64
from multiprocessing import Process, Queue, Event, Lock
from object_detection import ObjectDetection
from ocr_pp import OCR
from frame_annotator import FrameAnnotator, AnnotationMode

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Configure logging
configure_logging()  # Call the logging configuration function

# Global variables for camera management
camera_lock = Lock()
#camera_url = "http://192.168.0.29:8080/video"
camera_url = 0
reconnect_timeout = 5  # seconds between reconnection attempts

def get_camera_connection():
    """
    Establish connection to the IP camera with error handling and reconnection logic.
    """
    while True:
        with camera_lock:
            cap = cv2.VideoCapture(camera_url,cv2.CAP_DSHOW)
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

def ocr_process(frame_queue, ocr_results_queue, stop_event):
    """
    Process frames for OCR in a separate process.
    """
    ocr_reader = OCR()
    while not stop_event.is_set():
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = frame_queue.get(timeout=0.1)
            ocr_results = ocr_reader.detect_text(frame)

            if ocr_results:
                if ocr_results_queue.full():
                    try:
                        ocr_results_queue.get_nowait()
                    except:
                        pass
                ocr_results_queue.put(ocr_results)
                logger.debug("OCR results added to queue.")
            else:
                logger.debug("No OCR results found.")

        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"OCR processing error: {str(e)}")

def object_detection_process(frame_queue, yolo_results_queue, stop_event):
    """
    Process frames for object detection in a separate process.
    """
    object_detector = ObjectDetection('yolo11n.pt')
    while not stop_event.is_set():
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = frame_queue.get(timeout=0.1)
            yolo_results = object_detector.detect_objects(frame)

            if yolo_results:
                if yolo_results_queue.full():
                    try:
                        yolo_results_queue.get_nowait()
                    except:
                        pass
                yolo_results_queue.put(yolo_results)
                logger.debug("YOLO results added to queue.")
            else:
                logger.debug("No YOLO results found.")

        except Exception as e:
            if not stop_event.is_set():
                logger.error(f"Object Detection error: {str(e)}")

def generate_frames():
    """
    Generator function to yield video frames with error handling and reconnection logic.
    """
    cap = None

    # Initialize processing components
    object_detector = ObjectDetection('yolo11x.pt')
    ocr_reader = OCR()
    frame_annotator = FrameAnnotator()  # Initialize without parameters as per the class definition

    # Initialize multiprocessing components
    frame_queue = Queue(maxsize=1)
    ocr_results_queue = Queue(maxsize=1)
    yolo_results_queue = Queue(maxsize=1)
    stop_event = Event()

    # Initialize results
    latest_ocr_results = None
    latest_yolo_results = None

    # Start processing processes
    ocr_proc = Process(target=ocr_process, args=(frame_queue, ocr_results_queue, stop_event))
    yolo_proc = Process(target=object_detection_process, args=(frame_queue, yolo_results_queue, stop_event))

    ocr_proc.start()
    yolo_proc.start()

    try:
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

                # Send frame to processing queues
                if not frame_queue.full():
                    frame_queue.put(frame)

                # Check for latest processing results
                if not ocr_results_queue.empty():
                    latest_ocr_results = ocr_results_queue.get_nowait()
                if not yolo_results_queue.empty():
                    latest_yolo_results = yolo_results_queue.get_nowait()

                # Annotate Frame with the correct method signature
                annotated_frame, closest_object = frame_annotator.annotate_frame(
                    frame=frame,
                    yolo_results=latest_yolo_results,
                    ocr_results=latest_ocr_results,
                    filter_tracked=True,
                    annotation_mode=AnnotationMode.BOX_ONLY
                )

                # Convert to base64
                frame_base64 = convert_frame_to_base64(annotated_frame)
                if frame_base64 is None:
                    continue

                # Create response with additional data
                response_data = {
                    'frame': frame_base64,
                    'timestamp': time.time(),
                    'closest_object': closest_object
                }
                print(closest_object)
                response = json.dumps(response_data)
                #print(response)
                yield f"data: {response}\n\n"

                # Small delay to prevent overwhelming the system
                time.sleep(0.01)  # 10ms delay between frames

            except Exception as e:
                logger.error(f"Error in frame generation: {str(e)}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(reconnect_timeout)

    finally:
        # Cleanup
        stop_event.set()
        time.sleep(0.1)

        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                break

        ocr_proc.join(timeout=1.0)
        yolo_proc.join(timeout=1.0)

        if ocr_proc.is_alive():
            ocr_proc.terminate()
        if yolo_proc.is_alive():
            yolo_proc.terminate()

        if cap is not None:
            cap.release()

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
    #app.run(debug=False, host='192.168.0.30', threaded=True)
    app.run(debug=False, threaded=True)