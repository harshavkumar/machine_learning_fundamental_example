import cv2
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def calculate_optical_flow_lucas_kanade(frame1, frame2):
    """
    Calculate sparse optical flow using the Lucas-Kanade method.

    Args:
        frame1 (numpy.ndarray): The first grayscale frame.
        frame2 (numpy.ndarray): The second grayscale frame.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the new feature points
               and the second array contains the old feature points.
    """
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Detect feature points in the first frame
    p0 = cv2.goodFeaturesToTrack(frame1, mask=None, **feature_params)
    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)
    
    # Filter out good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    return good_new, good_old

def calculate_optical_flow_horn_schunck(frame1, frame2):
    """
    Calculate dense optical flow using the Horn-Schunck method (via Farneback in OpenCV).

    Args:
        frame1 (numpy.ndarray): The first grayscale frame.
        frame2 (numpy.ndarray): The second grayscale frame.

    Returns:
        numpy.ndarray: The computed dense optical flow.
    """
    flow_hs = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow_hs

def compute_epe_sparse(good_new, good_old, ground_truth, scale_factor=64):
    """
    Compute the Endpoint Error (EPE) for sparse optical flow.

    Args:
        good_new (numpy.ndarray): The new feature points.
        good_old (numpy.ndarray): The old feature points.
        ground_truth (numpy.ndarray): The ground truth flow field.
        scale_factor (int): Scaling factor for ground truth.

    Returns:
        float: The average EPE for the sparse optical flow.
    """
    epe_sparse = 0
    count = 0
    height, width = ground_truth.shape[:2]
    
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        
        if 0 <= int(a) < width and 0 <= int(b) < height:
            gt_flow = ground_truth[int(b), int(a), :2] / scale_factor
            estimated_flow = np.array([a - c, b - d])
            epe_sparse += np.linalg.norm(estimated_flow - gt_flow)
            count += 1
    
    return epe_sparse / count if count > 0 else 0

def compute_epe_dense(flow, ground_truth, scale_factor=64):
    """
    Compute the Endpoint Error (EPE) for dense optical flow.

    Args:
        flow (numpy.ndarray): The computed dense optical flow.
        ground_truth (numpy.ndarray): The ground truth flow field.
        scale_factor (int): Scaling factor for ground truth.

    Returns:
        float: The average EPE for the dense optical flow.
    """
    gt_flow_scaled = ground_truth[..., :2] / scale_factor
    epe = np.sqrt((flow[..., 0] - gt_flow_scaled[..., 0])**2 + (flow[..., 1] - gt_flow_scaled[..., 1])**2)
    return np.mean(epe)

def visualize_optical_flows(frame1, good_new, good_old, flow_hs, ground_truth, scale=0.5):
    """
    Visualize and compare optical flows with the ground truth.

    Args:
        frame1 (numpy.ndarray): The first grayscale frame.
        good_new (numpy.ndarray): The new feature points for Lucas-Kanade.
        good_old (numpy.ndarray): The old feature points for Lucas-Kanade.
        flow_hs (numpy.ndarray): The dense optical flow from Horn-Schunck method.
        ground_truth (numpy.ndarray): The ground truth flow field.
        scale (float): Scaling factor for visualization.
        delay (int): Delay in milliseconds for displaying the images.
    """
    # Prepare Lucas-Kanade visualization
    output_lk = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        output_lk = cv2.line(output_lk, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        output_lk = cv2.circle(output_lk, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Prepare Horn-Schunck visualization
    hsv = np.zeros_like(cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR))
    mag, ang = cv2.cartToPolar(flow_hs[..., 0], flow_hs[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow_hs = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Visualize ground truth flow
    gt_flow_visual = np.zeros_like(cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR))
    gt_flow_scaled = ground_truth[..., :2] / 64
    mag, ang = cv2.cartToPolar(gt_flow_scaled[..., 0], gt_flow_scaled[..., 1])
    gt_flow_visual[..., 0] = ang * 180 / np.pi / 2
    gt_flow_visual[..., 1] = 255
    gt_flow_visual[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_gt_flow = cv2.cvtColor(gt_flow_visual, cv2.COLOR_HSV2BGR)

    # Resize images
    output_lk_resized = cv2.resize(output_lk, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    rgb_flow_hs_resized = cv2.resize(rgb_flow_hs, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    rgb_gt_flow_resized = cv2.resize(rgb_gt_flow, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # Combine images vertically
    combined_image = np.vstack((output_lk_resized, rgb_flow_hs_resized, rgb_gt_flow_resized))

    # Add annotations
    cv2.putText(combined_image, 'Lucas-Kanade', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined_image, 'Horn-Schunck', (10, output_lk_resized.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined_image, 'Ground Truth', (10, output_lk_resized.shape[0] * 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the combined image
    cv2.imshow('Optical Flow Comparison', combined_image)


def process_kitti_dataset(image_dir, ground_truth_dir):
    """
    Process KITTI dataset to evaluate and visualize optical flow methods.

    Args:
        image_dir (str): Directory containing the image frames.
        ground_truth_dir (str): Directory containing the ground truth flow fields.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    ground_truth_files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.png')])

    total_epe_lk = 0
    total_epe_hs = 0
    num_pairs = len(image_files) // 2

    for i in range(num_pairs):
        # Read frames and ground truth
        frame1 = cv2.imread(os.path.join(image_dir, image_files[2 * i]), cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(os.path.join(image_dir, image_files[2 * i + 1]), cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(os.path.join(ground_truth_dir, ground_truth_files[i]), cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Calculate optical flows
        good_new, good_old = calculate_optical_flow_lucas_kanade(frame1, frame2)
        flow_hs = calculate_optical_flow_horn_schunck(frame1, frame2)

        # Evaluate EPE
        epe_lk = compute_epe_sparse(good_new, good_old, ground_truth)
        epe_hs = compute_epe_dense(flow_hs, ground_truth)
        total_epe_lk += epe_lk
        total_epe_hs += epe_hs

        # Visualize results
        visualize_optical_flows(frame1, good_new, good_old, flow_hs, ground_truth, scale=0.5)

        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

    print(f'Average EPE for Lucas-Kanade: {total_epe_lk / num_pairs:.4f}')
    print(f'Average EPE for Horn-Schunck: {total_epe_hs / num_pairs:.4f}')

def main():
    """
    Main function to run the optical flow evaluation.
    """
    # Set the paths to the KITTI dataset images and ground truth
    image_dir = './kitti_dataset/optical_flow/training/image_0'
    ground_truth_dir = './kitti_dataset/optical_flow/training/flow_occ'

    process_kitti_dataset(image_dir, ground_truth_dir)

if __name__ == "__main__":
    main()
