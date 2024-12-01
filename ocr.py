# ocr.py

import logging
import numpy as np
import easyocr

logger = logging.getLogger(__name__)

class OCR:
    def __init__(self):
        """
        Initialize the OCR reader.
        """
        self.reader = None
        self._initialize_reader()

    def _initialize_reader(self):
        """
        Initialize the EasyOCR reader.
        """
        try:
            logger.info("Initializing EasyOCR reader...")
            self.reader = easyocr.Reader(['en','fr'],gpu=True)
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {str(e)}")
            raise

    def detect_text(self, frame: np.ndarray) -> dict:
        """
        Detect text in a given frame.

        Args:
            frame (np.ndarray): The input frame.

        Returns:
            dict: A dictionary containing the centers of bounding boxes, probabilities, words, and bounding box coordinates of detected texts.
                - centers (list of tuples): Centers of the bounding boxes (x, y).
                - probabilities (list of floats): Probabilities of the detected texts.
                - words (list of str): Words detected.
                - bboxes (list of tuples): Bounding box coordinates (x1, y1, x2, y2).
        """
        try:
            results = self.reader.readtext(frame)

            centers = []
            probabilities = []
            words = []
            bboxes = []

            for result in results:
                bbox, text, confidence = result
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                centers.append((x_center, y_center))
                probabilities.append(confidence)
                words.append(text)
                bboxes.append((bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]))

            return {
                'centers': centers,
                'probabilities': probabilities,
                'words': words,
                'bboxes': bboxes
            }
        except Exception as e:
            logger.error(f"Text detection failed: {str(e)}")
            return {}
