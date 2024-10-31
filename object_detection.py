# object_detection.py

import logging
from logger_config import configure_logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Configure logging
configure_logging()  # Call the logging configuration function

class ObjectDetection:
    def __init__(self, model_path: str):
        """
        Initialize the object detection model.

        Args:
            model_path (str): The path to the YOLO model.
        """
        self.model_path = Path(model_path)
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the YOLO model.
        """
        try:
            logger.info("Initializing YOLO model...")
            self.model = YOLO(self.model_path)
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}")
            raise

    def detect_objects(self, frame: np.ndarray) -> dict:
        """
        Detect objects in a given frame.

        Args:
            frame (np.ndarray): The input frame.

        Returns:
            dict: A dictionary containing the centers of bounding boxes, probabilities, labels, and bounding box coordinates of detected objects.
                - centers (list of tuples): Centers of the bounding boxes (x, y).
                - probabilities (list of floats): Probabilities of the detected objects.
                - labels (list of str): Labels of the detected objects.
                - bboxes (list of tuples): Bounding box coordinates (x1, y1, x2, y2).
        """
        try:
            results = self.model(frame)
            result = results[0]  # Assuming single frame input

            centers = []
            probabilities = []
            labels = []
            bboxes = []

            for box, prob, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                centers.append((x_center, y_center))
                probabilities.append(prob.item())
                labels.append(result.names[int(class_id)])
                bboxes.append((box[0].item(), box[1].item(), box[2].item(), box[3].item()))

            return {
                'centers': centers,
                'probabilities': probabilities,
                'labels': labels,
                'bboxes': bboxes
            }
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            return {}
