# main_mp.py

import logging
import time
import cv2
from multiprocessing import Process, Queue, Event
from object_detection import ObjectDetection
from ocr import OCR
from frame_annotator import FrameAnnotator
from logger_config import configure_logging

logger = logging.getLogger(__name__)

def ocr_process(frame_queue, ocr_results_queue, stop_event):
    """
    Processes frames from the frame_queue, applies OCR, and stores results in ocr_results_queue.
    The process stops when stop_event is set.
    """
    ocr_reader = OCR()
    while not stop_event.is_set():
        try:
            if frame_queue.empty():
                time.sleep(0.01)  # Short sleep to prevent CPU spinning
                continue
                
            frame = frame_queue.get(timeout=0.1)
            logger.info("Received frame for OCR.")
            ocr_results = ocr_reader.detect_text(frame)

            if ocr_results:
                if ocr_results_queue.full():
                    try:
                        ocr_results_queue.get_nowait()  # Remove oldest result if queue is full
                    except:
                        pass
                ocr_results_queue.put(ocr_results)
                logger.info("OCR results added to queue.")
            else:
                logger.info("No OCR results found.")

        except Exception as e:
            if not stop_event.is_set():  # Only log errors if we're not stopping
                logger.error(f"OCR processing error: {e}", exc_info=True)

def object_detection_process(frame_queue, yolo_results_queue, stop_event):
    """
    Processes frames from the frame_queue, applies object detection, and stores results in yolo_results_queue.
    The process stops when stop_event is set.
    """
    object_detector = ObjectDetection('yolo11x.pt')
    while not stop_event.is_set():
        try:
            if frame_queue.empty():
                time.sleep(0.01)  # Short sleep to prevent CPU spinning
                continue
                
            frame = frame_queue.get(timeout=0.1)
            logger.info("Received frame for Object Detection.")
            yolo_results = object_detector.detect_objects(frame)

            if yolo_results:
                if yolo_results_queue.full():
                    try:
                        yolo_results_queue.get_nowait()  # Remove oldest result if queue is full
                    except:
                        pass
                yolo_results_queue.put(yolo_results)
                logger.info("YOLO results added to queue.")
            else:
                logger.info("No YOLO results found.")

        except Exception as e:
            if not stop_event.is_set():  # Only log errors if we're not stopping
                logger.error(f"Object Detection error: {e}", exc_info=True)

def main():
    """
    Main function for initializing and managing OCR and Object Detection processes, video capture,
    and frame annotation. Handles graceful shutdown on user interruption.
    """
    configure_logging()
    frame_queue = Queue(maxsize=1)
    ocr_results_queue = Queue(maxsize=1)
    yolo_results_queue = Queue(maxsize=1)
    stop_event = Event()

    # Start OCR and Object Detection processes
    ocr_proc = Process(target=ocr_process, args=(frame_queue, ocr_results_queue, stop_event))
    yolo_proc = Process(target=object_detection_process, args=(frame_queue, yolo_results_queue, stop_event))
    ocr_proc.start()
    yolo_proc.start()

    # Initialize variables for OCR and YOLO results
    ocr_results = None
    yolo_results = None

    # Set up frame annotator and video capture
    frame_annotator = FrameAnnotator(start_time=time.time(), frame_count=0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open camera.")
        stop_event.set()  # Signal other processes to stop
        ocr_proc.join()
        yolo_proc.join()
        return 1

    logger.info("Processing loop started...")
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame.")
                break

            # Send the latest frame for processing
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # Discard old frame
                except:
                    pass
            frame_queue.put(frame)

            # Get OCR and YOLO results if available
            try:
                if not ocr_results_queue.empty():
                    ocr_results = ocr_results_queue.get_nowait()
                if not yolo_results_queue.empty():
                    yolo_results = yolo_results_queue.get_nowait()
            except:
                pass

            # Annotate frame if results are available
            if yolo_results:
                annotated_frame = frame_annotator.annotate_frame(frame, yolo_results, ocr_results)
                cv2.imshow('Object Detection & OCR', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit signal received.")
                break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

    finally:
        # Signal processes to exit and ensure clean shutdown
        stop_event.set()
        time.sleep(0.1)  # Give processes time to notice the stop event
        
        # Clean up queues without blocking
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except:
                break
                
        ocr_proc.join(timeout=1.0)  # Wait up to 1 second for processes to finish
        yolo_proc.join(timeout=1.0)
        
        # Terminate processes if they haven't finished
        if ocr_proc.is_alive():
            ocr_proc.terminate()
        if yolo_proc.is_alive():
            yolo_proc.terminate()
            
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info("Cleanup complete.")

    return 0

if __name__ == "__main__":
    exit(main())