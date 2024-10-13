### README

#### Project Overview

This project focuses on developing a cohesive system for autonomous vehicle navigation by integrating motion detection, estimation, and tracking components. It processes sequences from the KITTI dataset to detect and track objects, estimate their motions, and predict trajectories.

#### Task
1. Motion Detection 
Implement a motion detection algorithm to identify moving objects in KITTI image sequences.
    * Use background subtraction techniques, adapting them for a moving camera scenario.
    * Implement noise reduction techniques to minimize false positives.
    * Apply morphological operations to improve the detection results.
    * Compare your results with KITTI's object detection ground truth.

    Deliverables:
    - Python code for the motion detection algorithm
    - discussing the results, and comparing with KITTI ground truth

2. Motion Estimation 
Develop motion estimation algorithms to determine the velocity and direction of detected moving objects.
    * Implement the Lucas-Kanade optical flow algorithm for sparse feature tracking.
    * Use the Horn-Schunck method for dense optical flow estimation.
    * Test both methods on KITTI's optical flow benchmark.
    * Compare the results of both methods and discuss their strengths and weaknesses in the context of autonomous vehicle navigation.

    Deliverables:
    - Python code for both Lucas-Kanade and Horn-Schunck algorithms
    - Including visualizations of the motion vectors and evaluation against KITTI's optical flow ground truth

3. Motion Tracking 
Create a motion tracking system that can follow multiple objects across frames, maintaining their identities and predicting their future positions.
    * Implement the Kalman filter for motion tracking.
    * Develop a method to associate detection with existing tracks (e.g., Hungarian algorithm).
    * Handle occlusions and new object entries/exits from the scene.
    * Use RANSAC to make the tracking robust against outliers.
    * Test your system on KITTI's object tracking benchmark.

    Deliverables:
    - Python code for the motion tracking system
    - Including how you handled challenging scenarios like occlusions, and comparing your results with KITTI's tracking ground truth

4. Integration and Application 
Combine the motion detection, estimation, and tracking components into a cohesive system for autonomous vehicle navigation.
    * Develop a simple decision-making module that uses the motion analysis results to make navigation decisions (e.g., slow down, change lanes).
    * Create a visualization that shows the detected objects, their estimated motions, and predicted trajectories.
    * Test your integrated system on a complete KITTI raw data sequence.
    * Discuss how this system could be extended to handle more complex scenarios.

#### Project Dependencies

1. **Python**: Ensure you have Python 3.9 or later installed.
2. **OpenCV**: Install OpenCV using pip:
   ```bash
   pip install opencv-python-headless
   ```
3. **NumPy**: Install NumPy using pip:
   ```bash
   pip install numpy
   ```
4. **Filterpy**: Install Filterpy using pip:
    ```
    pip install filterpy
    ```
5. **Scipy**: Install Scipy using pip:
    ```
    pip install scipy
    ```
6. **Tqdm**: Install Tqdm using pip:
    ```
    pip install tqdm
    ```
7. **KITT Dataset**: Download the KITTI dataset from the official website (https://www.cvlibs.net/datasets/kitti/) and extract it to a directory of your choice.


    * Object Tracking:
    ```
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip -P ./kitti_dataset/object_tracking

        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip -P ./kitti_dataset/object_tracking
    ```
    * Optical Flow 
    ```
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip -P ./kitti_dataset/optical_flow
    ```
    *  Object Detection
    ```
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -P ./kitti_dataset/object_detection
        
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip -P ./kitti_dataset/object_detection
    ```
    * Raw Data
    ```
        wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip -P ./kitti_dataset/raw_data

        after that extract from raw_data_downloader.zip and run shell script file raw_data_downloader.sh to download data in same directory
    ```

#### Setup

1. **Navigate to the Project Directory**: Move to the project directory where files and dataset have been extracted:
   ```bash
   cd /path/to/project/directory
   ```
2. **Download the Dataset subset**: unzip the Dataset Subset provided in Zip.
3. **Code Directory**: Extract code in same directory as of dataset 
4. **Download Dataset**: For large dataset download from KITTI Dataset website by using above provide scripts

#### Running the Code

1. **Run Individual files**: for benchmarking you can run individual files for usecase, just change path if required in the respective file.
    ```bash
   python3 individual_filename.py
   ```

2. **Run the Main Script**: Execute the main script:
   ```bash
   python3 main.py
   ```
3. **Specify Image Sequence Path**: enter the path to the KITTI image sequence directory in script file if required:
   ```bash
   Enter the path to the KITTI image sequence directory: /path/to/kitti/image/sequence
   ```
4. **Specify Annotation File**: enter the path to the KITTI annotation file in script file if required for evaluation:
   ```bash
   Enter the path to the KITTI annotation file: /path/to/kitti/annotation/file
   ```

#### Code Structure

1. **`motion_estimation.py`**: Contains functions for calculating optical flow using Lucas-Kanade and Horn-Schunck methods.
2. **`motion_detection.py`**: Includes functions for background subtraction, noise reduction, and morphological operations.
3. **`motion_tracking.py`**: Provides functions for tracking objects using the Kalman filter and data association.
4. **`main.py`**: The main script that integrates all components and processes the KITTI dataset.

#### Troubleshooting

1. **Ensure Correct Paths**: Verify that the paths to the KITTI dataset and annotation files are correct.
2. **Check Dependencies**: Ensure that all dependencies are installed and up-to-date.
3. **Consult Documentation**: Refer to the OpenCV and NumPy documentation for any issues related to these libraries.
4. **Warning * On entry to DLASCL parameter number  4 had an illegal value**: Its coming because of robust data formatting in RANSAC algorithm, it will not effect the current system


### Conclusion

By following these steps and ensuring that all dependencies are met, you should be able to successfully run the project and explore its capabilities in autonomous vehicle navigation.
