import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment 
from filterpy.kalman import KalmanFilter
import os
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


# Step 1: Define the Kalman Filter class
class KalmanBoxTracker(object):
    """
    Class for tracking objects using a Kalman Filter.

    Attributes:
        kf (KalmanFilter): Kalman Filter instance for tracking.
        time_since_update (int): Number of frames since the last update.
        id (int): Unique identifier for the tracker.
        history (list): History of tracked states.
        hits (int): Number of successful hits.
        hit_streak (int): Number of consecutive hits.
        age (int): Age of the tracker.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialize a KalmanBoxTracker instance with a bounding box.

        Args:
            bbox (np.ndarray): Initial bounding box (x1, y1, x2, y2).
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0,0],
            [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0],
            [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = bbox.reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Update the tracker with a new bounding box.

        Args:
            bbox (np.ndarray): New bounding box (x1, y1, x2, y2).
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox)

    def predict(self):
        """
        Predict the next state and update history.

        Returns:
            np.ndarray: Predicted bounding box (x1, y1, x2, y2).
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.kf.x[:4].reshape((1,4))

    def get_state(self):
        """
        Get the current state of the tracker.

        Returns:
            np.ndarray: Current bounding box (x1, y1, x2, y2).
        """
        return self.kf.x[:4].reshape((1,4))

    def predict_occluded(self):
        predicted_state = self.predict()
        return predicted_state

    def predict_occluded(self):
        """
        Predict the state of the tracker when occluded.

        Returns:
            np.ndarray: Predicted bounding box (x1, y1, x2, y2).
        """
        predicted_state = self.predict()
        self.time_since_update += 1
        return predicted_state
    
    def estimate_velocity(self):
        """
        Estimate the velocity of the tracked object based on historical data.
        """
        if len(self.history) > 1:
            current_pos = self.kf.x[:4].flatten()
            prev_pos = self.history[-2][:4].flatten()
            velocity = current_pos - prev_pos
            self.kf.x[4:] = velocity.reshape((4, 1))

    def smooth_state(self, alpha=0.7):
        """
        Smooth the state estimate using exponential smoothing.

        Args:
            alpha (float): Smoothing factor between 0 and 1.
        """
        if len(self.history) > 1:
            smoothed_state = alpha * self.kf.x[:4].flatten() + (1 - alpha) * self.history[-2][:4].flatten()
            self.kf.x[:4] = smoothed_state.reshape((4, 1))


# Step 2: Implement the Hungarian algorithm for data association
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.35):
    """
    Associate detections to existing trackers using the Hungarian algorithm.

    Args:
        detections (np.ndarray): Array of detections, each represented by a bounding box (x1, y1, x2, y2).
        trackers (np.ndarray): Array of trackers, each represented by a bounding box (x1, y1, x2, y2).
        iou_threshold (float): IoU threshold for considering a detection and tracker as matched.

    Returns:
        tuple: A tuple containing:
            - matches (np.ndarray): Array of matched indices (detection index, tracker index).
            - unmatched_detections (np.ndarray): Indices of unmatched detections.
            - unmatched_trackers (np.ndarray): Indices of unmatched trackers.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.column_stack((row_ind, col_ind))

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# Step 3: Implement IOU calculation
def iou(bb_test, bb_gt):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        bb_test (np.ndarray): Bounding box (x1, y1, x2, y2) of the test object.
        bb_gt (np.ndarray): Bounding box (x1, y1, x2, y2) of the ground truth object.

    Returns:
        float: Intersection over Union (IoU) score.
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)


# Step 4: Implement RANSAC for robust tracking
def ransac_filter(detections, threshold=3.0):
    """
    Filter out outliers from detections using the RANSAC algorithm.

    Args:
        detections (np.ndarray): Array of detections, each represented by a bounding box (x1, y1, x2, y2).
        threshold (float): Distance threshold for RANSAC inlier definition.
        max_trials (int): Maximum number of RANSAC iterations.

    Returns:
        np.ndarray: Filtered array of detections.
    """
    if len(detections) < 2:
        return detections

    best_inliers = []

    # Check for zero standard deviation
    std_x = np.std(detections[:, 0])
    std_y = np.std(detections[:, 1])

    if std_x == 0 or std_y == 0:
        return detections

    # Normalize the data
    mean_x = np.mean(detections[:, 0])
    mean_y = np.mean(detections[:, 1])

    normalized_detections = np.copy(detections)
    normalized_detections[:, 0] = (detections[:, 0] - mean_x) / std_x
    normalized_detections[:, 1] = (detections[:, 1] - mean_y) / std_y

    for _ in range(100):  # Number of RANSAC iterations
        if len(normalized_detections) < 2:
            continue
        try:
            sample = normalized_detections[np.random.choice(len(normalized_detections), 2, replace=False)]
            model = np.polyfit(sample[:, 0], sample[:, 1], 1)
            distances = np.abs(normalized_detections[:, 1] - (model[0] * normalized_detections[:, 0] + model[1]))
            inliers = detections[distances < threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        except np.linalg.LinAlgError:
            continue

    return best_inliers if len(best_inliers) > 0 else detections


# Step 5: Main tracking function
class Sort(object):
    """
    SORT (Simple Online and Realtime Tracking) algorithm for tracking objects.

    Attributes:
        max_age (int): Maximum number of frames a tracker can be inactive before being deleted.
        min_hits (int): Minimum number of frames a tracker must be active to be considered valid.
        iou_threshold (float): IOU threshold for associating detections with trackers.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize a SORT tracker.

        Args:
            max_age (int): Maximum number of frames a tracker can be inactive before being deleted.
            min_hits (int): Minimum number of frames a tracker must be active to be considered valid.
            iou_threshold (float): IOU threshold for associating detections with trackers.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Update the trackers with new detections.

        Args:
            detections (np.ndarray): Array of detections, each represented by a bounding box (x1, y1, x2, y2).

        Returns:
            np.ndarray: Array of tracked objects, each represented by a bounding box (x1, y1, x2, y2) and an object ID.
        """
        self.frame_count += 1
        
        # Apply RANSAC to filter outliers
        if len(dets) > 0:
            dets = ransac_filter(dets)

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks[:, :4], self.iou_threshold)

        # Update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
                trk.estimate_velocity()
                trk.smooth_state()

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
            

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            elif trk.time_since_update < self.max_age:
                d = trk.predict_occluded()[0]
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    
# Step 6: Load KITTI dataset
def load_kitti_data(data_dir, sequence):
    """
    Load images and annotations from the KITTI dataset.

    Args:
        data_dir (str): Base directory of the KITTI dataset.
        sequence (str): Sequence number to load.

    Returns:
        tuple: A tuple containing:
            - images (list of str): List of image file paths.
            - labels (dict): Dictionary mapping frame indices to lists of ground truth boxes (track_id, bbox).
    """
    img_dir = os.path.join(data_dir, 'image_02', sequence)
    label_dir = os.path.join(data_dir, 'label_02', f'{sequence}.txt')
    
    images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    
    labels = defaultdict(list)
    with open(label_dir, 'r') as f:
        for line in f:
            data = line.strip().split()
            frame = int(data[0])
            track_id = int(data[1])
            bbox = [float(x) for x in data[6:10]]
            labels[frame].append((track_id, bbox))
    
    return images, labels

# Step 7: Evaluation metrics
def compute_metrics(gt_tracks, pred_tracks):
    """
    Compute evaluation metrics for tracking results.

    Args:
        gt_tracks (dict): Ground truth tracks, mapping frame indices to lists of (track_id, bbox).
        pred_tracks (dict): Predicted tracks, mapping frame indices to lists of (x1, y1, x2, y2, obj_id).

    Returns:
        dict: A dictionary with evaluation metrics including Precision, Recall, F1-Score, MOTA, and ID Switches.
    """
    total_gt = sum(len(tracks) for tracks in gt_tracks.values())
    total_pred = sum(len(tracks) for tracks in pred_tracks.values())
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    id_switches = 0
    prev_matches = {}
    
    for frame in sorted(gt_tracks.keys()):
        gt_boxes = np.array([box for _, box in gt_tracks[frame]])
        pred_boxes = np.array([box[:4] for box in pred_tracks.get(frame, [])])
        
        if len(gt_boxes) == 0:
            false_positives += len(pred_boxes)
            continue
        
        if len(pred_boxes) == 0:
            false_negatives += len(gt_boxes)
            continue
        
        # Compute IoU between all pairs of boxes
        ious = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                ious[i, j] = iou(gt_box, pred_box)
        
        # Match boxes using Hungarian algorithm
        matched_indices = linear_sum_assignment(-ious)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        # Count matches
        for i, j in matched_indices:
            if ious[i, j] >= 0.5:
                true_positives += 1
                gt_id = gt_tracks[frame][i][0]
                pred_id = int(pred_tracks[frame][j][4])
                if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                    id_switches += 1
                prev_matches[gt_id] = pred_id
            else:
                false_positives += 1
                false_negatives += 1
        
        false_positives += len(pred_boxes) - len(matched_indices)
        false_negatives += len(gt_boxes) - len(matched_indices)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    mota = 1 - (false_positives + false_negatives + id_switches) / total_gt if total_gt > 0 else 0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'MOTA': mota,
        'ID Switches': id_switches
    }


from collections import defaultdict

# Step 7: Main function to run the tracking system
def main():
    """
    Main function to run the tracking system, visualize the results, and evaluate metrics.
    """

    data_dir = './kitti_dataset/object_tracking/training/'
    sequence = '0002'  # Change this to test different sequences
    
    images, gt_labels = load_kitti_data(data_dir, sequence)
    mot_tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.35)
    
    pred_tracks = defaultdict(list)
    
    for frame, img_path in enumerate(tqdm(images)):
        img = cv2.imread(img_path)
        detections = np.array([box for _, box in gt_labels[frame]])
        
        if len(detections) > 0:
            tracked_objects = mot_tracker.update(detections)
            
            for obj in tracked_objects:
                x1, y1, x2, y2, obj_id = obj
                pred_tracks[frame].append(obj)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"ID: {int(obj_id)}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Tracking', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    # Evaluate the results
    metrics = compute_metrics(gt_labels, pred_tracks)
    print("Tracking Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()