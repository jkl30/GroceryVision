# frame_annotator.py

import cv2
import logging
import numpy as np
import time
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (tuple): Bounding box coordinates (x1, y1, x2, y2).
        box2 (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        float: The IoU value.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def match_ocr_to_yolo(yolo_bboxes, ocr_bboxes, ocr_words, ocr_probabilities):
    """
    Match each OCR word to the YOLO bounding box with the highest IoU.

    Args:
        yolo_bboxes (List[tuple]): List of YOLO bounding boxes.
        ocr_bboxes (List[tuple]): List of OCR bounding boxes.
        ocr_words (List[str]): List of OCR detected words.
        ocr_probabilities (List[float]): List of OCR detection probabilities.

    Returns:
        List[Dict]: Matched OCR information with YOLO bounding box index, word, and probability.
    """
    matched_results = []

    for ocr_bbox, ocr_word, ocr_prob in zip(ocr_bboxes, ocr_words, ocr_probabilities):
        best_iou = 0
        best_yolo_index = -1

        # Find the YOLO bbox with the highest IoU for this OCR bbox
        for i, yolo_bbox in enumerate(yolo_bboxes):
            iou = calculate_iou(ocr_bbox, yolo_bbox)
            if iou > best_iou:
                best_iou = iou
                best_yolo_index = i

        # Add to matched results only if a matching YOLO box is found
        if best_yolo_index >= 0:
            matched_results.append({
                'yolo_index': best_yolo_index,
                'word': ocr_word,
                'probability': ocr_prob
            })

    return matched_results

class FrameAnnotator:
    def __init__(self, start_time: float, frame_count: int):
        """
        Initialize the frame annotator.

        Args:
            start_time (float): The start time of the processing.
            frame_count (int): The initial frame count.
        """
        self.start_time = start_time
        self.frame_count = frame_count
        self.last_fps_update_time = start_time
        self.current_fps = 0

    def annotate_frame(self, frame: np.ndarray, yolo_results: Optional[Dict], ocr_results: Optional[Dict] = None) -> np.ndarray:
        """
        Annotates the frame with detection results from YOLO and optional OCR results.

        Args:
            frame (np.ndarray): The input frame.
            yolo_results (Optional[Dict]): The YOLO detected results.
            ocr_results (Optional[Dict]): The OCR detection results.

        Returns:
            np.ndarray: The annotated frame.
        """
        annotated = frame.copy()

        # Check if yolo_results is provided
        if yolo_results is None:
            logger.warning("YOLO results are None, skipping annotation.")
            return annotated  # Return the original frame if no results

        # Annotate YOLO detections with circles at the center and display information
        for i, (center, probability, label, yolo_bbox) in enumerate(
            zip(
                yolo_results.get('centers', []), 
                yolo_results.get('probabilities', []), 
                yolo_results.get('labels', []), 
                yolo_results.get('bboxes', [])
            )
        ):
            center_x, center_y = map(int, center)
            cv2.circle(annotated, (center_x, center_y), 5, (255, 255, 255), -1)

            # Display the YOLO text beside the circle
            yolo_text = f"[{probability:.2f}] {label}"
            text_position = (center_x + 10, center_y)
            cv2.putText(annotated, yolo_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # If OCR results exist, match OCR boxes to YOLO boxes and annotate
        if ocr_results:
            matched_ocr = match_ocr_to_yolo(
                yolo_results.get('bboxes', []),
                ocr_results.get('bboxes', []),
                ocr_results.get('words', []),
                ocr_results.get('probabilities', [])
            )

            # Group matched OCR results by YOLO index
            ocr_groups = {}
            for match in matched_ocr:
                yolo_index = match['yolo_index']
                if yolo_index not in ocr_groups:
                    ocr_groups[yolo_index] = []
                ocr_groups[yolo_index].append(match)

            # Display matched OCR text under the corresponding YOLO text
            for yolo_index, matches in ocr_groups.items():
                center_x, center_y = map(int, yolo_results['centers'][yolo_index])
                for j, match in enumerate(matches):
                    ocr_word = match['word']
                    ocr_probability = match['probability']
                    ocr_text = f"[{ocr_probability:.2f}] {ocr_word}"
                    ocr_text_position = (center_x + 10, center_y + 15 + 15 * j)  # Positioning OCR text below YOLO text

                    cv2.putText(annotated, ocr_text, ocr_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red color

        # Calculate and display FPS over a rolling 1-second window
        current_time = time.time()
        elapsed_since_last_update = current_time - self.last_fps_update_time
        self.frame_count += 1

        if elapsed_since_last_update >= 1.0:
            self.current_fps = self.frame_count / elapsed_since_last_update
            self.last_fps_update_time = current_time
            self.frame_count = 0  # Reset frame count for the next interval

        # Display the current FPS
        cv2.putText(annotated, f'FPS: {self.current_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated
