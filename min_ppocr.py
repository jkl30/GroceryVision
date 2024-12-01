import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageFont
import numpy as np
import os

def get_default_font():
    """Get a default font that should work on most systems"""
    try:
        # Try to use Arial on Windows
        if os.name == 'nt':  # Windows
            font_path = "C:/Windows/Fonts/arial.ttf"
            if os.path.exists(font_path):
                return font_path
        
        # Try common paths on Linux
        linux_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
            "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch Linux
            "/usr/share/fonts/dejavu/DejaVuSans.ttf"  # Fedora
        ]
        for font_path in linux_fonts:
            if os.path.exists(font_path):
                return font_path
        
        # If no system font found, use PIL's default font
        default_font = ImageFont.load_default()
        return default_font
        
    except Exception as e:
        print(f"Warning: Could not load system font: {e}")
        return None

def main():
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get default font
    font_path = get_default_font()
    if font_path is None:
        print("Warning: No font found. Text may not display properly.")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break
            
            # Convert frame to RGB for PaddleOCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform OCR on the frame
            result = ocr.ocr(rgb_frame, cls=True)
            
            # Process results if text is detected
            if result and len(result) > 0 and result[0] is not None:
                # Get the first result page
                page_result = result[0]
                
                # Check if there are any detections on this page
                if page_result:
                    try:
                        # Extract boxes, texts and scores
                        boxes = [line[0] for line in page_result]
                        txts = [line[1][0] for line in page_result]
                        scores = [line[1][1] for line in page_result]
                        
                        # Draw the results on the frame
                        if boxes:
                            # Convert numpy array to PIL Image
                            pil_img = Image.fromarray(rgb_frame)
                            
                            # Draw OCR results
                            drawn_img = draw_ocr(pil_img, boxes, txts, scores, font_path=font_path)
                            
                            # Convert back to BGR for display
                            frame = cv2.cvtColor(np.array(drawn_img), cv2.COLOR_RGB2BGR)
                        
                        # Print detected text
                        for line in page_result:
                            print(f"Detected: {line[1][0]} (Confidence: {line[1][1]:.2f})")
                            
                    except Exception as e:
                        print(f"Error processing OCR results: {e}")
            
            # Display the frame
            cv2.imshow('Webcam OCR', frame)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()