import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from card_detector import detect_and_crop_cards
from segment_icons import segment_icons
from torchvision.transforms import ToPILImage
import random
from collections import Counter

# Load class names from text file 
def load_class_names(path="card_classes.txt"):
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Class names file not found at {path}")
        return []

# Load model
def load_model(model_path, num_classes, device):
    if num_classes == 0:
        print("Error: No class names loaded, cannot determine model output size.")
        return None
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except RuntimeError as e:
        print(f"Error: Could not load model state_dict from {model_path}. Details: {e}")
        return None
    model.to(device)
    model.eval()
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_icon(model, icon_pil_img, class_names, device, threshold=0.3):
    if not class_names:
        return 'error_no_classes', 0.0
    icon_tensor = transform(icon_pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(icon_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        
        predicted_class_name = 'unknown'
        confidence_score = conf.item()

        if confidence_score >= threshold:
            if pred_idx.item() < len(class_names):
                predicted_class_name = class_names[pred_idx.item()]
            else:
                print(f"Warning: Predicted index {pred_idx.item()} out of bounds for {len(class_names)} classes.")
                predicted_class_name = 'error_pred_idx_oob'
        
        return predicted_class_name, confidence_score

# Main 
def process_test_images(test_dir='spot-it',
                        model_path='models/resnet_icon.pt',
                        class_txt='card_classes.txt',
                        output_csv='Gupta_Keskar.csv',
                        debug_main=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    class_names = load_class_names(class_txt)
    if not class_names:
        print("Halting: Class names could not be loaded.")
        return

    model = load_model(model_path, len(class_names), device)
    if model is None:
        print("Halting: Model could not be loaded.")
        return

    if debug_main:
        os.makedirs("debug_icons", exist_ok=True)
        print("Debug mode is ON. Segmented icons may be saved to 'debug_icons/'.")

    results = []
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in directory: {test_dir}")
        return
        
    for filename in sorted(image_files):
        print(f"\n📷 Processing {filename}...")
        image_path = os.path.join(test_dir, filename)
        match_symbol = 'unknown' 

        try:
            # detected_masks from here will now be ignored for icon segmentation on cropped cards
            detected_cards, _ = detect_and_crop_cards(image_path, debug=False) 
        except ValueError as e_val: 
            print(f"Card detection raised ValueError for {filename}: {e_val}. Marking as no cards detected.")
            detected_cards = []
        except Exception as e_card_detect:
            print(f"CRITICAL ERROR during card detection for {filename}: {e_card_detect}. Skipping.")
            results.append((filename, 'error_critical_card_detection'))
            continue

        if len(detected_cards) >= 2:
            print(f"Found {len(detected_cards)} card regions, processing first two for comparison.")
            cards_to_process = detected_cards[:2]
            
            per_card_icon_sets = []
            for cidx in range(len(cards_to_process)):
                card_image = cards_to_process[cidx]
                # current_mask = masks_to_process[cidx] # OLD LOGIC

                if card_image is None or card_image.size == 0:
                    print(f"      Card {cidx+1}: Image is empty. Adding empty icon set.")
                    per_card_icon_sets.append(set())
                    continue
                
                print(f"Processing Card {cidx+1} (cropped region)...")
                # Always pass None for card_mask_
                # It just works :shrug:
                icon_cv_images, _ = segment_icons(card_image, card_mask_provided=None, debug=False)
                
                icon_predictions_for_this_card  = []
                print(f"Found {len(icon_cv_images)} candidate icons for Card {cidx+1}.")
                if not icon_cv_images:
                    per_card_icon_sets.append(set())
                    continue

                for i, individual_icon_cv_img in enumerate(icon_cv_images):
                    try:
                        rgb_icon = cv2.cvtColor(individual_icon_cv_img, cv2.COLOR_BGR2RGB)
                        pil_icon = ToPILImage()(rgb_icon)
                    except Exception as e_conversion:
                        print(f"  Error converting icon {i+1} for Card {cidx+1}: {e_conversion}")
                        continue
                    
                    predicted_class, confidence = predict_icon(model, pil_icon, class_names, device, threshold=0.6)
                    if predicted_class not in ['unknown', 'error_no_classes', 'error_pred_idx_oob'] and \
                       predicted_class not in icon_predictions_for_this_card:
                        icon_predictions_for_this_card.append(predicted_class)
                    if debug_main:
                        safe_cls_name = predicted_class.replace("/", "_").replace("\\", "_")
                        fn_debug = f"{filename}_card{cidx+1}_icon{i+1}_{safe_cls_name}_{confidence:.2f}.jpg"
                        cv2.imwrite(os.path.join("debug_icons", fn_debug), individual_icon_cv_img)
                per_card_icon_sets.append(set(icon_predictions_for_this_card))

            if len(per_card_icon_sets) == 2:
                common_symbols = per_card_icon_sets[0].intersection(per_card_icon_sets[1])
                if len(common_symbols) > 1:
                    print(f"      ⚠️ Found multiple common symbols: {common_symbols}. Picking one randomly.")
                    match_symbol = random.choice(list(common_symbols))
                elif len(common_symbols) == 1:
                    match_symbol = common_symbols.pop()
                else:
                    match_symbol = 'two_cards_no_common_symbol' 
            else: 
                match_symbol = 'error_processing_two_cards'

        elif len(detected_cards) == 1:
            print(f" Found 1 card region. Processing entire image to find frequently occurring icons.")
            full_image = cv2.imread(image_path) 
            if full_image is None:
                print(f"Error reloading full image {image_path} for single card logic. Skipping.")
                match_symbol = 'error_reloading_full_image_for_single_card'
            else:
                print(f"Processing entire image ({filename})...")
                # Pass None as mask, so segment_icons uses its default for the whole image
                icon_cv_images, _ = segment_icons(full_image, card_mask_provided=None, debug=False)
                
                all_icon_predictions_in_image = []
                print(f" Found {len(icon_cv_images)} candidate icons in the entire image.")

                if icon_cv_images:
                    for i, individual_icon_cv_img in enumerate(icon_cv_images):
                        try:
                            rgb_icon = cv2.cvtColor(individual_icon_cv_img, cv2.COLOR_BGR2RGB)
                            pil_icon = ToPILImage()(rgb_icon)
                        except Exception as e_conversion:
                            print(f"         ⚠️ Error converting icon {i+1} from full image: {e_conversion}")
                            continue
                        
                        predicted_class, confidence = predict_icon(model, pil_icon, class_names, device, threshold=0.6)
                        if predicted_class not in ['unknown', 'error_no_classes', 'error_pred_idx_oob']:
                            all_icon_predictions_in_image.append(predicted_class)
                        if debug_main:
                            safe_cls_name = predicted_class.replace("/", "_").replace("\\", "_")
                            fn_debug = f"{filename}_fullimage_icon{i+1}_{safe_cls_name}_{confidence:.2f}.jpg"
                            cv2.imwrite(os.path.join("debug_icons", fn_debug), individual_icon_cv_img)
                    
                    if all_icon_predictions_in_image:
                        icon_counts = Counter(all_icon_predictions_in_image)
                        duplicate_icons = [item for item, count in icon_counts.items() if count > 1]
                        if duplicate_icons:
                            most_frequent_duplicate = ""
                            max_count = 1 
                            for item, count in icon_counts.items():
                                if count > max_count:
                                    max_count = count
                                    most_frequent_duplicate = item
                                elif count == max_count and count > 1: 
                                    if not most_frequent_duplicate : most_frequent_duplicate = item 
                            if most_frequent_duplicate: 
                                match_symbol = most_frequent_duplicate
                                print(f"Most frequent duplicated icon in full image: {match_symbol} (count: {max_count})")
                            else: 
                                match_symbol = 'single_image_no_duplicates_found'
                        else:
                            match_symbol = 'single_image_no_duplicates_found'
                    else:
                        match_symbol = 'single_image_no_valid_icons_predicted'
                else:
                    match_symbol = 'single_image_no_icons_segmented'
        else: # 0 cards detected
            print(f"No card regions detected in {filename}.")
            match_symbol = 'no_cards_detected_match_unknown'

        print(f"Match for {filename}: {match_symbol}")
        results.append((filename, match_symbol))

    df = pd.DataFrame(results, columns=['ID','Class'])
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(results)} results to {output_csv}")
    if not results:
        print("No images were processed or no results to save.")

if __name__ == "__main__":
    process_test_images(debug_main=False)
