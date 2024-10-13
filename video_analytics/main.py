import cv2
import numpy as np
import os
from motion_estimation import calculate_optical_flow_horn_schunck
from motion_detection import background_subtraction, reduce_noise, apply_morphology
from motion_tracking import Sort
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def load_kitti_images(image_sequence_path):
    """
    Load and sort image file paths from the specified directory.

    Args:
        image_sequence_path (str): Directory containing the image sequence.

    Returns:
        list: Sorted list of image file paths.
    """
    images = sorted([os.path.join(image_sequence_path, img) for img in os.listdir(image_sequence_path) if img.endswith('.png')])
    return images

def decision_making(tracked_objects):
    """
    Make decisions based on the width of the tracked objects.

    Args:
        tracked_objects (list): List of tracked objects, each represented by a bounding box (x1, y1, x2, y2) and an object ID.

    Returns:
        list: List of decisions for each tracked object.
    """
    decisions = []
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        width = x2 - x1
        if width > 100:
            decisions.append("Slow down")
        elif width < 50:
            decisions.append("Maintain speed")
        else:
            decisions.append("Change lanes")
    return decisions

def visualize_tracking(frame, tracked_objects, decisions, flow, scale=0.7):
    """
    Visualize the tracking results on the frame and display decisions.

    Args:
        frame (np.ndarray): The current frame image.
        tracked_objects (list): List of tracked objects, each represented by a bounding box (x1, y1, x2, y2) and an object ID.
        decisions (list): List of decisions corresponding to each tracked object.
    """

    for obj, decision in zip(tracked_objects, decisions):
        x1, y1, x2, y2, obj_id = obj
        track_motion_frame = frame.copy()
        cv2.rectangle(track_motion_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(track_motion_frame, f"ID: {int(obj_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw predicted trajectory
        trajectory = [(int(x1 + (x2 - x1) * t / 10), int(y1 + (y2 - y1) * t / 10)) for t in range(1, 11)]
        for point in trajectory:
            cv2.circle(track_motion_frame, point, 2, (255, 0, 0), -1)
        
        # Draw optical flow vectors
        op_flow_frame = frame.copy()
        step = 16
        for y in range(0, flow.shape[0], step):
            for x in range(0, flow.shape[1], step):
                fx, fy = flow[y, x]
                cv2.arrowedLine(op_flow_frame, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1, tipLength=0.3)
    
    # Display decisions in the top-left corner
    for i, decision in enumerate(decisions):
        cv2.putText(frame, f"Decision {i+1}: {decision}", (10, 52 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Resize images
    frame_resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    track_motion_frame_resized = cv2.resize(track_motion_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    op_flow_frame_resized = cv2.resize(op_flow_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Combine images vertically
    combined_image = np.vstack((frame_resized, track_motion_frame_resized, op_flow_frame_resized))

    # Add annotations
    cv2.putText(combined_image, 'Original Frames', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined_image, 'Motion Tracking and Trajectory', (10, frame_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined_image, 'Optical Flow motion', (10, frame_resized.shape[0] * 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Visualization of Raw Data', combined_image)


def main():
    """
    Main function to run the tracking system and visualize results.
    """
    image_sequence_path = './kitti_dataset/raw_data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data'
    images = load_kitti_images(image_sequence_path)
    
    # Initialize the SORT tracker
    mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.35)
    
    # Read the first frame
    prev_frame = cv2.imread(images[0])
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(images)):
        # Read the next frame
        next_frame = cv2.imread(images[i])
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = calculate_optical_flow_horn_schunck(prev_gray, next_gray)
        
        # Process the motion mask
        motion_mask = background_subtraction(flow)
        clean_mask = apply_morphology(reduce_noise(motion_mask))
        
        # Detect objects in the motion mask
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = np.array([cv2.boundingRect(cnt) for cnt in contours])
        detections = np.array([[x, y, x + w, y + h] for (x, y, w, h) in detections])
        
        # Update the tracker with the new detections
        tracked_objects = mot_tracker.update(detections)
        
        # Make decisions based on tracked objects
        decisions = decision_making(tracked_objects)
        
        # Visualize the tracking results and decisions
        visualize_tracking(next_frame, tracked_objects, decisions, flow)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        # Update previous frame
        prev_frame = next_frame
        prev_gray = next_gray
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
