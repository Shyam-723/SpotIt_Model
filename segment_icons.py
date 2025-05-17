import cv2
import numpy as np
from scipy.spatial import distance_matrix 

# calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Boxes are expected in (x1, y1, x2, y2) format.
    """
    # Ensure coordinates are integers
    boxA = tuple(map(int, boxA))
    boxB = tuple(map(int, boxB))
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Handle cases where area is zero
    if boxAArea <= 0 or boxBArea <= 0: return 0.0
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0: return 0.0
        
    iou = interArea / unionArea
    return iou

# Remove Nested and Overlapping Boxes 
def remove_nested_and_overlapping_boxes(boxes, iou_threshold=0.3, containment_threshold=0.9, debug=False):
    """
    1. If IoU > iou_threshold, removes the box with the smaller area.
    2. If box B is smaller than box A and its area is largely contained within box A 
       (intersection_area / area(B) > containment_threshold), removes box B.
    Boxes are expected in (x1, y1, x2, y2) format.
    """
    if not boxes:
        return []
        
    num_boxes = len(boxes)
    if num_boxes <= 1:
        return boxes

    # Store boxes with original index and area, filtering out zero-area boxes
    boxes_info = []
    for i, box in enumerate(boxes):
        try:
            int_box = tuple(map(int, box))
            width = max(0, int_box[2] - int_box[0])
            height = max(0, int_box[3] - int_box[1])
            area = width * height
            if area > 0:
                boxes_info.append({'id': i, 'box': int_box, 'area': area, 'keep': True})
            # else:
            #     if debug: print(f"  Skipping box idx {i} {int_box} due to zero area.")
        except (TypeError, ValueError, IndexError) as e:
            print(f"Overlap Removal Warning: Skipping invalid box format {box}. Error: {e}")
            continue
            
    if not boxes_info:
        return []

    num_valid_boxes = len(boxes_info)
    if debug: print(f"Overlap/Nested Removal Start: {num_valid_boxes} valid boxes initially.")

    # Compare all unique pairs of boxes
    for i in range(num_valid_boxes):
        info_i = boxes_info[i]
        # If box i has already been marked for removal, skip its comparisons
        if not info_i['keep']:
            continue
            
        for j in range(i + 1, num_valid_boxes):
            info_j = boxes_info[j]
            # If box j has already been marked for removal, skip comparison
            if not info_j['keep']:
                continue

            box_i = info_i['box']
            area_i = info_i['area']
            box_j = info_j['box']
            area_j = info_j['area']

            # Calculate IoU
            iou = calculate_iou(box_i, box_j)
            if debug: print(f"  Comparing box idx {info_i['id']} ({area_i:.0f}) and idx {info_j['id']} ({area_j:.0f}): IoU = {iou:.4f}")

            # Check 1: High IoU overlap -> remove smaller 
            if iou > iou_threshold:
                if area_i >= area_j: # Keep box i, discard box j
                    info_j['keep'] = False
                    if debug: print(f"    Discarding box idx {info_j['id']} (smaller area, IoU > {iou_threshold})")
                else: # Keep box j, discard box i
                    info_i['keep'] = False
                    if debug: print(f"    Discarding box idx {info_i['id']} (smaller area, IoU > {iou_threshold})")
                    break # Box i is removed, no need to compare it further
                continue # Move to next j comparison if one was discarded (or i was discarded)

            # Check 2: Nested boxes -> remove smaller if significantly contained
            # Calculate intersection area (needed for containment check)
            xA = max(box_i[0], box_j[0])
            yA = max(box_i[1], box_j[1])
            xB = min(box_i[2], box_j[2])
            yB = min(box_i[3], box_j[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)

            # If box j is smaller and largely contained within box i
            # Check containment: intersection_area / area_of_smaller_box
            if area_j < area_i:
                containment_j_in_i = interArea / float(area_j) if area_j > 0 else 0
                if containment_j_in_i > containment_threshold:
                    info_j['keep'] = False
                    if debug: print(f"    Discarding box idx {info_j['id']} (contained within idx {info_i['id']} > {containment_threshold*100}%)")
                    continue # Move to next j

            # If box i is smaller and largely contained within box j
            elif area_i < area_j:
                containment_i_in_j = interArea / float(area_i) if area_i > 0 else 0
                if containment_i_in_j > containment_threshold:
                    info_i['keep'] = False
                    if debug: print(f"    Discarding box idx {info_i['id']} (contained within idx {info_j['id']} > {containment_threshold*100}%)")
                    break # Box i is removed
            
            # Note: If areas are equal, containment check isn't needed unless IoU was high.
            
    # Collect the boxes that were marked to be kept
    final_boxes = [info['box'] for info in boxes_info if info['keep']]

    if debug: print(f"Overlap/Nested Removal End: {len(final_boxes)} boxes kept.")
    return final_boxes


# merges nearby contours before checking IoU
def merge_nearby_contours(boxes, merge_distance=80):
    """
    Merge bounding boxes whose centers are within `merge_distance` px.
    Uses a connected components approach to ensure all related boxes are merged.
    """
    if not boxes: return []

    if not all(isinstance(b, (list, tuple)) and len(b) == 4 for b in boxes):
        print(f"Warning: Invalid structure in boxes for merge_nearby_contours. Boxes: {boxes}")
        return boxes
    num_boxes = len(boxes)
    if num_boxes <= 1: return boxes
    centers = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for x1, y1, x2, y2 in boxes])
    if centers.shape[0] < 1: return boxes
    try:
        D = distance_matrix(centers, centers)
        np.fill_diagonal(D, np.inf)
    except ValueError as e:
        print(f"Error creating distance matrix: {e}. Centers: {centers}")
        return boxes
    merged_flags = [False] * num_boxes
    final_merged_boxes = []
    for i in range(num_boxes):
        if merged_flags[i]: continue
        current_group_indices = []
        queue = [i]; merged_flags[i] = True; head = 0
        while head < len(queue):
            current_box_idx = queue[head]; head += 1
            current_group_indices.append(current_box_idx)
            for j in range(num_boxes):
                if not merged_flags[j] and D[current_box_idx, j] < merge_distance:
                    merged_flags[j] = True; queue.append(j)
        if not current_group_indices: continue
        group_boxes = [boxes[idx] for idx in current_group_indices]
        xs = [b[0] for b in group_boxes]; ys = [b[1] for b in group_boxes]
        Xe = [b[2] for b in group_boxes]; Ye = [b[3] for b in group_boxes]
        final_merged_boxes.append((min(xs), min(ys), max(Xe), max(Ye)))
    return final_merged_boxes


# Update segment_icons to use the new overlap/nested removal function 
def segment_icons(card_img, card_mask_provided=None, debug=False):
    """
    Segments icons from a card image using an edge-based approach.
    Includes merging nearby contours and removal of nested/overlapping boxes.

    If `card_mask_provided` is None, it generates a large default circular mask 
    intended to cover most of the input `card_img`.

    This works for some reason then making an individual work
    """
    # Step 0: Validate input image
    if card_img is None:
        print("Error: segment_icons - card_img is None.")
        return [], []
    h_img, w_img = card_img.shape[:2] 
    if h_img == 0 or w_img == 0:
        print(f"Error: segment_icons - card_img has zero dimension: {card_img.shape}")
        return [], []

    # Convert input image to grayscale
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

    # Step 1: Determine the final mask to use
    final_card_mask = card_mask_provided
    if final_card_mask is None:
        cx, cy = w_img // 2, h_img // 2
        R = 100000 # User's setting
        final_card_mask = np.zeros((h_img, w_img), np.uint8)
        cv2.circle(final_card_mask, (cx, cy), R, 255, -1) 
        if debug: print(f"Segment_icons: Generated default mask with R={R} for image size {w_img}x{h_img}")
    
    if debug:
        display_mask = cv2.resize(final_card_mask, (min(w_img, 500), min(h_img, 500)))
        cv2.imshow("1. Card Mask (Provided or Generated)", display_mask)
        cv2.waitKey(0)

    # Step 2: Pre-processing for Canny Edge Detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3) 
    if debug:
        cv2.imshow("2. Initial Canny Edges", edges)
        cv2.waitKey(0)

    # Step 4: Morphological Operations
    dilate_kernel_size = (3,3) 
    dilate_iterations = 2 
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_kernel_size)
    dilated_edges = cv2.dilate(edges, dilate_kernel, iterations=dilate_iterations)
    if debug:
        cv2.imshow("3. Dilated Edges (Iterations=2)", dilated_edges)
        cv2.waitKey(0)

    use_closing = True 
    if use_closing:
        close_kernel_size = (5,5) 
        close_iterations = 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
        processed_edges_for_contours = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
        if debug:
            cv2.imshow("4. Closed Edges (for contour finding)", processed_edges_for_contours)
            cv2.waitKey(0)
    else:
        processed_edges_for_contours = dilated_edges 

    # Step 5: Apply the final card mask
    edges_within_card_mask = cv2.bitwise_and(processed_edges_for_contours, processed_edges_for_contours, mask=final_card_mask)
    if debug:
        cv2.imshow("5. Processed Edges within Card Mask", edges_within_card_mask)
        cv2.waitKey(0)
    
    # Step 6: Find Contours using RETR_LIST
    final_contours, hierarchy = cv2.findContours(edges_within_card_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_img_contours_on_edges = card_img.copy()
        cv2.drawContours(debug_img_contours_on_edges, final_contours, -1, (0,0,255), 1) 
        cv2.imshow("6. Contours Found on Masked Edges (RETR_LIST)", debug_img_contours_on_edges)
        cv2.waitKey(0)

    # Step 7: Filter Contours into Raw Bounding Boxes
    raw_boxes = [] 
    min_icon_area_ratio = 0.003 
    max_icon_area_ratio = 0.25 
    min_icon_dim_ratio = 0.05
    max_box_dim_ratio = 0.40 
    card_area_for_filter = float(h_img * w_img)
    min_dim_for_filter = float(min(h_img, w_img))

    if debug:
        debug_img_filtered_boxes = card_img.copy()

    for i_cnt, cnt in enumerate(final_contours):
        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        
        if w_b > max_box_dim_ratio * w_img or h_b > max_box_dim_ratio * h_img: continue
        bbox_area = float(w_b * h_b)
        if not (min_icon_area_ratio * card_area_for_filter * 0.5 < bbox_area < max_icon_area_ratio * card_area_for_filter * 1.5): continue
        if w_b < min_icon_dim_ratio * min_dim_for_filter or h_b < min_icon_dim_ratio * min_dim_for_filter: continue
        aspect_ratio = float(w_b) / h_b if h_b > 0 else 0
        if not (0.3 < aspect_ratio < 3.0): continue
            
        raw_boxes.append((x_b, y_b, x_b + w_b, y_b + h_b))
        if debug:
            cv2.rectangle(debug_img_filtered_boxes, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 255, 0), 2)

    if debug:
        cv2.imshow("7. Filtered Bounding Boxes (RETR_LIST)", debug_img_filtered_boxes)
        cv2.waitKey(0)

    # Step 8: Merge Nearby Raw Bounding Boxes
    enable_merging = True 
    merge_distance_ratio = 0.05 
    merge_distance_px = int(merge_distance_ratio * w_img)
    
    if enable_merging and len(raw_boxes) > 1 :
        merged_boxes = merge_nearby_contours(raw_boxes, merge_distance=merge_distance_px)
    else:
        merged_boxes = raw_boxes 

    if debug and merged_boxes:
        debug_img_merged_boxes = card_img.copy()
        for x1,y1,x2,y2 in merged_boxes:
             cv2.rectangle(debug_img_merged_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2) 
        cv2.imshow("8. Final Merged Boxes (Before Overlap Removal)", debug_img_merged_boxes)
        cv2.waitKey(0)

    # Step 8.5: Apply Nested and Overlap Removal 
    iou_overlap_threshold = 0.3     # Remove smaller if IoU > 0.3
    containment_check_threshold = 0.9 # Remove smaller if > 90% contained in larger
    
    final_boxes_filtered = remove_nested_and_overlapping_boxes(
        merged_boxes, 
        iou_threshold=iou_overlap_threshold, 
        containment_threshold=containment_check_threshold, 
        debug=debug
    ) 
    
    if debug and final_boxes_filtered:
        debug_img_overlap_removed_boxes = card_img.copy()
        print(f"Segment_icons: Boxes before Removal: {len(merged_boxes)}, Boxes after Removal (IoU>{iou_overlap_threshold}, Contain>{containment_check_threshold}): {len(final_boxes_filtered)}")
        for x1,y1,x2,y2 in final_boxes_filtered:
             cv2.rectangle(debug_img_overlap_removed_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw final boxes in Blue
        cv2.imshow("8.5 Final Boxes after Nested/Overlap Removal", debug_img_overlap_removed_boxes)
        cv2.waitKey(0)

    # Step 9: Crop and Resize Icons based on the final filtered boxes
    icons = [] 
    for x1, y1, x2, y2 in final_boxes_filtered: # Use boxes after filtering
        if x1 >= x2 or y1 >= y2: continue
        icon_crop = card_img[y1:y2, x1:x2] 
        if icon_crop.size == 0: continue
        icons.append(cv2.resize(icon_crop, (128, 128), interpolation=cv2.INTER_AREA))

    if debug:
        print(f"Segment_icons: Number of final icons extracted: {len(icons)}")
        cv2.destroyAllWindows() 
        
    return icons, final_boxes_filtered
