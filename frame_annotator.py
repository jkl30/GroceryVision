# frame_annotator.py

import cv2
import logging
from logger_config import configure_logging
import numpy as np
import time
from typing import List, Optional, Dict, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)
# Configure logging
configure_logging()
logger.setLevel(logging.INFO)  # Ensure the logger respects the configured level

class AnnotationMode(Enum):
    """Defines the available annotation visualization modes."""
    DOT_ONLY = "dot_only"
    DOT_AND_TEXT = "dot_and_text" 
    BOX_ONLY = "box_only"
    BOX_AND_TEXT = "box_and_text"

@dataclass
class TrackedObject:
    """Represents a tracked object with its properties."""
    id: int
    name: str
    aliases: Set[str]

    def matches(self, label: str) -> bool:
        """
        Check if a label matches this tracked object.
        
        Args:
            label (str): The label to check.
            
        Returns:
            bool: True if the label matches any of the object's aliases.
        """
        return label.lower() in self.aliases

    def to_dict(self) -> Dict:
        """
        Convert TrackedObject to a dictionary for JSON serialization.
        
        Returns:
            Dict: Dictionary representation of the TrackedObject.
        """
        return {
            'id': self.id,
            'name': self.name,
            'aliases': list(self.aliases)  # Convert set to list for JSON serialization
        }

# Define tracked objects using the dataclass
TRACKED_OBJECTS = {
    'yogurt': TrackedObject(
        id=1,
        name='yogurt',
        aliases={'yogurt', 'greek', 'kirkland'}
    ),
    'honey': TrackedObject(
        id=2,
        name='honey',
        aliases={'honey', 'organic', 'miel', 'cremeux','brut', 'creamed', 'raw'}
    ),
    'curry': TrackedObject(
        id=3,
        name='curry',
        aliases={'curry', 'powder', 'curry sauce', 'curry powder', 'poudre', 'cari','poudre de cari','village'}
    )
}

def calculate_distance_to_center(point: Tuple[float, float], center: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between a point and center.

    Args:
        point (Tuple[float, float]): The point coordinates (x, y).
        center (Tuple[float, float]): The center coordinates (x, y).

    Returns:
        float: The Euclidean distance between the points.
    """
    return np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)

def calculate_iou(box1: Tuple[float, float, float, float], 
                 box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (Tuple[float, float, float, float]): First bounding box (x1, y1, x2, y2).
        box2 (Tuple[float, float, float, float]): Second bounding box (x1, y1, x2, y2).

    Returns:
        float: The IoU value between 0 and 1.
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
    return inter_area / union_area if union_area > 0 else 0

def match_ocr_to_yolo(yolo_bboxes: List[Tuple[float, float, float, float]],
                     ocr_bboxes: List[Tuple[float, float, float, float]],
                     ocr_words: List[str],
                     ocr_probabilities: List[float]) -> List[Dict]:
    """
    Match OCR detected words to YOLO bounding boxes based on IoU.

    Args:
        yolo_bboxes (List[Tuple]): List of YOLO bounding boxes (x1, y1, x2, y2).
        ocr_bboxes (List[Tuple]): List of OCR bounding boxes (x1, y1, x2, y2).
        ocr_words (List[str]): List of OCR detected words.
        ocr_probabilities (List[float]): List of OCR detection probabilities.

    Returns:
        List[Dict]: List of dictionaries containing matched OCR information.
    """
    matched_results = []
    
    for ocr_bbox, word, prob in zip(ocr_bboxes, ocr_words, ocr_probabilities):
        best_iou = 0
        best_yolo_index = -1
        
        for i, yolo_bbox in enumerate(yolo_bboxes):
            iou = calculate_iou(ocr_bbox, yolo_bbox)
            if iou > best_iou:
                best_iou = iou
                best_yolo_index = i
        
        if best_iou > 0:
            matched_results.append({
                'yolo_index': best_yolo_index,
                'word': word,
                'probability': prob
            })
    
    return matched_results

class FrameAnnotator:
    """
    A class for annotating video frames with object detection and OCR results.
    """
    
    def __init__(self):
        """Initialize the frame annotator."""
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.last_tracked_object = None
        self.object_missing_frames = 0
        self.max_missing_frames = 100  # Object persists for ~1 second at 30fps
        self.detection_history = []
        self.history_size = 15  # Keep track of last 5 frames
        self.last_positions = {}  # {object_id: (x, y)}
        self.alpha = 0.05  # Smoothing factor (0-1), lower = more smoothing
        

    def _smooth_position(self, obj_id: int, new_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Apply exponential moving average to position"""
        if obj_id not in self.last_positions:
            self.last_positions[obj_id] = new_pos
            return new_pos
            
        old_x, old_y = self.last_positions[obj_id]
        new_x, new_y = new_pos
        
        # Calculate smoothed position
        smooth_x = old_x + self.alpha * (new_x - old_x)
        smooth_y = old_y + self.alpha * (new_y - old_y)
        
        # Update stored position
        self.last_positions[obj_id] = (smooth_x, smooth_y)
        return (smooth_x, smooth_y)

    def _get_tracked_object(self, label: str, ocr_words: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Get the tracked object that matches the label or OCR words.

        Args:
            label (str): The YOLO detection label.
            ocr_words (Optional[List[str]]): List of OCR words associated with this detection.

        Returns:
            Optional[Dict]: Dictionary representation of matching TrackedObject if found, None otherwise.
        """
        # Check YOLO label
        label = label.lower()
        for tracked_obj in TRACKED_OBJECTS.values():
            if tracked_obj.matches(label):
                return tracked_obj.to_dict()
                
        # Check OCR words if available
        if ocr_words:
            for word in ocr_words:
                for tracked_obj in TRACKED_OBJECTS.values():
                    if tracked_obj.matches(word.lower()):
                        return tracked_obj.to_dict()
                        
        return None

    def _update_fps(self) -> None:
        """Update the FPS counter."""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        self.fps_counter += 1

        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.last_fps_time = current_time

    def _find_closest_tracked_object(self, 
                                   frame: np.ndarray,
                                   detections: List[Tuple[Tuple[float, float], str, Optional[List[str]]]]) -> Optional[Dict]:
        """
        Find the tracked object closest to the frame center.

        Args:
            frame (np.ndarray): The input frame.
            detections (List[Tuple]): List of (center, label, ocr_words) for each detection.

        Returns:
            Optional[Dict]: Dictionary representation of the closest tracked object, or None if no tracked objects.
        """
        if not detections:
            return None

        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width / 2, frame_height / 2)
        
        # Add current frame detections to history
        self.detection_history.append(detections)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
            
        # Count occurrences of each tracked object in history
        object_counts = {}
        
        for frame_detections in self.detection_history:
            for center, label, ocr_words in frame_detections:
                tracked_obj = self._get_tracked_object(label, ocr_words)
                if tracked_obj:
                    obj_id = tracked_obj['id']
                    if obj_id not in object_counts:
                        object_counts[obj_id] = {
                            'count': 0,
                            'object': tracked_obj,
                            'last_center': center
                        }
                    object_counts[obj_id]['count'] += 1
                    object_counts[obj_id]['last_center'] = center

        # Find most consistent object (present in most frames)
        max_count = 0
        most_consistent = None
        
        for obj_data in object_counts.values():
            if obj_data['count'] > max_count:
                max_count = obj_data['count']
                most_consistent = obj_data['object']
            elif obj_data['count'] == max_count and most_consistent:
                # If equal counts, prefer the one closer to center
                current_distance = calculate_distance_to_center(obj_data['last_center'], frame_center)
                existing_distance = calculate_distance_to_center(object_counts[most_consistent['id']]['last_center'], frame_center)
                if current_distance < existing_distance:
                    most_consistent = obj_data['object']

        if most_consistent:
            self.last_tracked_object = most_consistent
            self.object_missing_frames = 0
            return most_consistent
            
        # No consistent object found
        self.object_missing_frames += 1
        if self.object_missing_frames > self.max_missing_frames:
            self.last_tracked_object = None
            
        return self.last_tracked_object

    def _find_closest_tracked_object(self, frame: np.ndarray, detections):
        if not detections:
            return None

        frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)
        min_distance = float('inf')
        closest_object = None
        
        for center, label, ocr_words in detections:
            tracked_obj = self._get_tracked_object(label, ocr_words)
            if tracked_obj is None:
                continue
                
            # Smooth the position
            smooth_center = self._smooth_position(tracked_obj['id'], center)
            distance = calculate_distance_to_center(smooth_center, frame_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_object = tracked_obj
        
        return closest_object

    def _draw_annotations(self,
                        frame: np.ndarray,
                        center: Tuple[float, float],
                        probability: float,
                        label: str,
                        bbox: Tuple[float, float, float, float],
                        ocr_matches: Optional[List[Dict]] = None,
                        annotation_mode: AnnotationMode = AnnotationMode.DOT_AND_TEXT) -> None:
        """
        Draw annotations for a single detection on the frame.

        Args:
            frame (np.ndarray): The frame to annotate.
            center (Tuple[float, float]): Center point of the detection.
            probability (float): Detection probability.
            label (str): Detection label.
            bbox (Tuple[float, float, float, float]): Bounding box coordinates (x1, y1, x2, y2).
            ocr_matches (Optional[List[Dict]]): Matching OCR results.
            annotation_mode (AnnotationMode): The annotation visualization mode.
        """
        center_x, center_y = int(center[0]), int(center[1])
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        if annotation_mode in [AnnotationMode.DOT_ONLY, AnnotationMode.DOT_AND_TEXT]:
            # Draw the dot
            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

        if annotation_mode in [AnnotationMode.BOX_ONLY, AnnotationMode.BOX_AND_TEXT]:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        if annotation_mode in [AnnotationMode.DOT_AND_TEXT, AnnotationMode.BOX_AND_TEXT]:
            # Draw YOLO detection text
            yolo_text = f"[{probability:.2f}] {label}"
            text_y = y1 - 10 if annotation_mode == AnnotationMode.BOX_AND_TEXT else center_y
            cv2.putText(frame, yolo_text, 
                        (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw OCR matches if available
            if ocr_matches:
                for idx, match in enumerate(ocr_matches):
                    ocr_text = f"[{match['probability']:.2f}] {match['word']}"
                    text_y = y1 - 25 - (15 * idx) if annotation_mode == AnnotationMode.BOX_AND_TEXT else center_y + 15 + (15 * idx)
                    cv2.putText(frame, ocr_text,
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def annotate_frame(self,
                      frame: np.ndarray,
                      yolo_results: Optional[Dict],
                      ocr_results: Optional[Dict] = None,
                      filter_tracked: bool = False,
                      annotation_mode: AnnotationMode = AnnotationMode.DOT_AND_TEXT) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Annotate the frame with detection results.

        Args:
            frame (np.ndarray): The input frame.
            yolo_results (Optional[Dict]): YOLO detection results containing 'centers',
                'probabilities', 'labels', and 'bboxes'.
            ocr_results (Optional[Dict]): OCR results containing 'bboxes', 'words',
                and 'probabilities'.
            filter_tracked (bool): If True, only annotate tracked objects.
            annotation_mode (AnnotationMode): The annotation visualization mode.

        Returns:
            Tuple[np.ndarray, Optional[Dict]]: Annotated frame and dictionary representation of closest tracked object.
        """
        if yolo_results is None:
            logger.warning("No YOLO results provided, skipping annotation.")
            return frame.copy(), None

        annotated = frame.copy()
        detections_with_ocr = []

        # Process YOLO detections with corresponding OCR results
        for i, (center, probability, label, yolo_bbox) in enumerate(
            zip(
                yolo_results.get('centers', []),
                yolo_results.get('probabilities', []),
                yolo_results.get('labels', []),
                yolo_results.get('bboxes', [])
            )
        ):
            # Get matching OCR results for this detection
            ocr_matches = []
            if ocr_results:
                ocr_matches = [
                    match for match in match_ocr_to_yolo(
                        yolo_results.get('bboxes', []),
                        ocr_results.get('bboxes', []),
                        ocr_results.get('words', []),
                        ocr_results.get('probabilities', [])
                    )
                    if match['yolo_index'] == i
                ]

            # Get OCR words for tracking check
            ocr_words = [match['word'] for match in ocr_matches] if ocr_matches else None
            
            # Store detection info for closest object calculation
            detections_with_ocr.append((center, label, ocr_words))
            
            # Skip if filtering and not a tracked object
            tracked_obj = self._get_tracked_object(label, ocr_words)
            if filter_tracked and tracked_obj is None:
                continue

            # Draw annotations for this detection
            self._draw_annotations(
                annotated, center, probability, label,
                yolo_bbox, ocr_matches, annotation_mode
            )

        # Update and draw FPS
        self._update_fps()
        cv2.putText(annotated, f'FPS: {self.current_fps:.2f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Find closest tracked object
        closest_object = self._find_closest_tracked_object(frame, detections_with_ocr)

        return annotated, closest_object