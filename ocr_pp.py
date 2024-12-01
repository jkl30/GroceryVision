import logging
from logger_config import configure_logging
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

# Configure logging
configure_logging()  # Call the logging configuration function
logger.setLevel(logging.INFO)  # Ensure the logger respects the configured level

class OCR:
    def __init__(self):
        """
        Initialize the OCR reader.
        """
        self.reader = None
        self._initialize_reader()

    def _initialize_reader(self):
        """
        Initialize the PaddleOCR reader.
        """
        try:
            logger.info("Initializing PaddleOCR reader...")
            self.reader = PaddleOCR(use_angle_cls=True, lang='en',show_log=False)
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR reader: {str(e)}")
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
            # Convert frame to RGB for PaddleOCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform OCR on the frame
            result = self.reader.ocr(rgb_frame, cls=True)

            centers = []
            probabilities = []
            words = []
            bboxes = []

            if result and len(result) > 0 and result[0] is not None:
                # Get the first result page
                page_result = result[0]

                # Check if there are any detections on this page
                if page_result:
                    for line in page_result:
                        bbox, (text, confidence) = line
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

if __name__ == "__main__":
    import cv2

    ocr = OCR()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break

            # Detect text in the frame
            results = ocr.detect_text(frame)

            # Draw the results on the frame
            if results:
                for bbox, text, confidence in zip(results['bboxes'], results['words'], results['probabilities']):
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{text} ({confidence:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Webcam OCR', frame)

            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
