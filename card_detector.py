import cv2
import numpy as np
import os

def detect_and_crop_cards(image_path: str, debug=False) -> list:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from: {image_path}")

    original = img.copy()
    debug_img = img.copy()

    # Grayscale and adaptive threshold 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 71, 7)
    thresh = cv2.bitwise_not(thresh)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Contours 
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 0.2 * img.shape[1] or h < 0.2 * img.shape[0]:
            continue

        aspect_ratio = w / h if h > 0 else 0
        squareness = 1 - abs(1 - aspect_ratio)
        score = area * squareness

        candidates.append((score, img[y:y+h, x:x+w], (x, y, w, h), cnt))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    # Decision logic
    if len(candidates) >= 2:
        cropped_cards = [c[1] for c in candidates[:2]]
        boxes = [c[2] for c in candidates[:2]]
    elif len(candidates) == 1:
        cropped_cards = [candidates[0][1]]  # just one card
        boxes = [candidates[0][2]]
    else:
        raise ValueError("No card detected.")

    # Debug display
    for _, _, (x, y, w, h), cnt in candidates[:2]:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 1)

    if debug:
        cv2.imshow("Threshold", thresh)
        cv2.waitKey(0)
        cv2.imshow("Morph Cleaned", clean)
        cv2.waitKey(0)
        cv2.imshow("Detected Cards", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cropped_masks = [clean[y:y+h, x:x+w] for (_, _, (x, y, w, h), _) in candidates[:len(cropped_cards)]]
    
    return cropped_cards, cropped_masks
