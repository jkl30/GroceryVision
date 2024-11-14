# main.py

import logging
import time
import cv2
import numpy as np
import base64
import json
from object_detection import ObjectDetection
from ocr import OCR
from frame_annotator import FrameAnnotator
from logger_config import configure_logging

logger = logging.getLogger(__name__)

def convert_frame_to_base64(frame):
    """
    Convert an OpenCV frame to a base64 encoded string.
    """
    _, buffer = cv2.imencode('.png', frame)  # You can use .jpg or .png
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    return frame_base64

def main():
    """
    Main function to run the object detection and OCR pipeline.
    """
    configure_logging()

    # Initialize components
    object_detector = ObjectDetection('yolo11n.pt')
    ocr_reader = OCR()
    frame_annotator = FrameAnnotator(start_time=time.time(), frame_count=0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera")
        return 1

    logger.info("Starting processing loop...")
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            # Object Detection
            yolo_results = object_detector.detect_objects(frame)

            # OCR Detection (Optional)
            ocr_results = ocr_reader.detect_text(frame)

            # Annotate Frame
            annotated_frame = frame_annotator.annotate_frame(frame, yolo_results, ocr_results)

            # Convert annotated frame to base64
            annotated_frame_base64 = convert_frame_to_base64(annotated_frame)

            # Create JSON response
            json_response = json.dumps({
                'frame': annotated_frame_base64,
            })

            # Display the annotated frame
            cv2.imshow('Object Detection & OCR', annotated_frame)

            # You can print or send the json_response as needed for your Flask app
            # For example, you might want to return it from a Flask route or log it
            logger.debug(f"JSON Response: {json_response}")  # This line logs the JSON response

            frame_count += 1
            logger.info(f"Processed frame {frame_count}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in processing loop: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup complete")

    return 0

if __name__ == "__main__":
    exit(main())
