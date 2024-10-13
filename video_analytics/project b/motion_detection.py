import cv2
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def calc_optical_flow(prev_frame, next_frame):
    """
    Calculate dense optical flow using Farneback's method.

    Args:
        prev_frame (np.array): The previous frame.
        next_frame (np.array): The next frame.

    Returns:
        np.array: The optical flow field.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Adjust parameters for better optical flow calculation
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow


def background_subtraction(flow):
    """
    Apply background subtraction using optical flow with adaptive thresholding.

    Args:
        flow (np.array): The optical flow.

    Returns:
        np.array: The binary mask indicating the detected motion regions.
    """
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude to the range [0, 255]
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to 8-bit image
    magnitude_uint8 = np.uint8(magnitude_normalized)
    
    # Use Otsu's method to find the optimal threshold
    _, mask = cv2.threshold(magnitude_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return mask


def reduce_noise(mask):
    """
    Reduce noise in the motion mask using bilateral filtering.

    Args:
        mask (np.array): The binary mask indicating motion regions.

    Returns:
        np.array: The noise-reduced binary mask.
    """
    # Apply bilateral filter for noise reduction
    noise_reduced_mask = cv2.bilateralFilter(mask, d=9, sigmaColor=75, sigmaSpace=75)
    
    return noise_reduced_mask


def apply_morphology(mask):
    """
    Apply morphological operations to clean up the mask.

    Args:
        mask (np.array): The noise-reduced binary mask.

    Returns:
        np.array: The cleaned-up mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    # Apply opening (erosion followed by dilation) to remove small noise
    morph_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Apply closing (dilation followed by erosion) to remove small holes
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel)
    return morph_mask


def draw_ground_truth_boxes(frame, annotations):
    """
    Draw ground truth bounding boxes on the image.

    Args:
        frame (np.array): The image frame.
        annotations (list): List of ground truth annotations.

    Returns:
        np.array: The image with ground truth boxes drawn.
    """
    for annotation in annotations:
        bbox = annotation['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green for ground truth
    return frame


def load_kitti_ground_truth(annotation_file, frame_idx):
    """
    Load KITTI ground truth annotations for a specific frame.

    Args:
        annotation_file (str): Path to the annotation file.
        frame_idx (int): The index of the frame.

    Returns:
        list: List of ground truth annotations for the frame.
    """
    annotations = []
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            if int(data[0]) == frame_idx and data[2] == 'Car':
                bbox = list(map(int, map(float, data[6:10])))
                annotations.append({'bbox': bbox})
    return annotations


def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list): The first bounding box.
        boxB (list): The second bounding box.

    Returns:
        float: The IoU score.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calculate_metrics(detected_boxes, ground_truth_boxes, iou_threshold=0.3):
    """
    Calculate precision, recall, and F1-score based on detected and ground truth boxes.

    Args:
        detected_boxes (list): List of detected bounding boxes.
        ground_truth_boxes (list): List of ground truth bounding boxes.
        iou_threshold (float): IoU threshold for determining true positives.

    Returns:
        tuple: Precision, recall, and F1-score.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = len(ground_truth_boxes)
    
    for detected_bbox in detected_boxes:
        matched = False
        for gt_bbox in ground_truth_boxes:
            iou = calculate_iou(detected_bbox, gt_bbox)
            if iou >= iou_threshold:
                true_positives += 1
                false_negatives -= 1
                matched = True
                break
        if not matched:
            false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def add_labels_to_frames(frame, label):
    """
    Add a label to the image frame.

    Args:
        frame (np.array): The image frame.
        label (str): The label text.

    Returns:
        np.array: The image with the label added.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2

    labeled_frame = cv2.putText(frame, label, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    return labeled_frame


def process_kitti_sequence(image_sequence_path, annotation_file):
    """
    Process the KITTI image sequence and evaluate motion detection performance.

    Args:
        image_sequence_path (str): Path to the folder containing image sequence.
        annotation_file (str): Path to the ground truth annotation file.
    """
    images = sorted([os.path.join(image_sequence_path, img) for img in os.listdir(image_sequence_path)])

    prev_frame = cv2.imread(images[0])
    total_precision, total_recall, total_f1 = 0, 0, 0
    frame_count = 0
    
    for i in range(1, len(images)):
        next_frame = cv2.imread(images[i])

        # Load ground truth annotations for the current frame
        ground_truth = load_kitti_ground_truth(annotation_file, i)
        ground_truth_boxes = [annotation['bbox'] for annotation in ground_truth]

        # Step 1: Calculate optical flow between consecutive frames
        flow = calc_optical_flow(prev_frame, next_frame)
        
        # Step 2: Background subtraction using the flow
        motion_mask = background_subtraction(flow)
        
        # Step 3: Reduce noise in the motion mask
        noise_reduced_mask = reduce_noise(motion_mask)
        
        # Step 4: Apply morphological operations to clean up the mask
        clean_mask = apply_morphology(noise_reduced_mask)
        
        # Find contours of the detected motion regions
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        detected_boxes = [[x, y, x + w, y + h] for (x, y, w, h) in detected_boxes]

        # Filter out small bounding boxes or boxes with unusual aspect ratios
        detected_boxes = [box for box in detected_boxes if box[2] - box[0] > 15 and box[3] - box[1] > 15]

        # Draw ground truth bounding boxes on the original frame
        annotated_frame = draw_ground_truth_boxes(next_frame.copy(), ground_truth)

        # Calculate precision, recall, and F1-score for the current frame
        precision, recall, f1_score = calculate_metrics(detected_boxes, ground_truth_boxes)
        total_precision += precision
        total_recall += recall
        total_f1 += f1_score
        frame_count += 1

        # Draw detected bounding boxes on the annotated frame
        for box in detected_boxes:
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)  # Red for detected

        # Label the frames
        annotated_frame = add_labels_to_frames(annotated_frame, "Original Ground Truth (Green) | Predicted (Red)")
        motion_detected_frame = cv2.bitwise_and(next_frame, next_frame, mask=clean_mask)
        motion_detected_frame = add_labels_to_frames(motion_detected_frame, "Motion Detection")
        clean_mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        clean_mask_bgr = add_labels_to_frames(clean_mask_bgr, "Clean Mask")
        
        # Resize frames for displaying
        height, width = annotated_frame.shape[:2]
        resize_factor = 0.5
        annotated_frame = cv2.resize(annotated_frame, (int(width * resize_factor), int(height * resize_factor)))
        motion_detected_frame = cv2.resize(motion_detected_frame, (int(width * resize_factor), int(height * resize_factor)))
        clean_mask_bgr = cv2.resize(clean_mask_bgr, (int(width * resize_factor), int(height * resize_factor)))

        # Combine frames vertically
        combined_output = cv2.vconcat([annotated_frame, motion_detected_frame, clean_mask_bgr])
        cv2.imshow('Original | Motion Detected | Clean Mask', combined_output)
        
        # Wait for a key press and break if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        # Update the previous frame to the current frame for the next iteration
        prev_frame = next_frame
    
    cv2.destroyAllWindows()

    # Calculate average metrics
    avg_precision = total_precision / frame_count if frame_count > 0 else 0
    avg_recall = total_recall / frame_count if frame_count > 0 else 0
    avg_f1_score = total_f1 / frame_count if frame_count > 0 else 0

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1-Score: {avg_f1_score:.4f}")


def main():
    """
    Main function to run the motion detection on the KITTI image sequence.
    """
    # Path to the KITTI image sequence folder
    image_sequence_path = './kitti_dataset/object_tracking/training/image_02/0002'
    # Path to the KITTI annotations folder
    annotation_file = './kitti_dataset/object_tracking/training/label_02/0002.txt'

    # Run the motion detection on the KITTI image sequence
    process_kitti_sequence(image_sequence_path, annotation_file)


if __name__ == '__main__':
    main()
