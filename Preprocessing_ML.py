import numpy as np
import pandas as pd
import cv2

def preprocess_video_frames_and_labels(video_path, labels_path, chunk_size=10):
    labels = pd.read_csv(labels_path, dtype={'Label': int}, low_memory=False)['Label'].values
    label_idx = 0
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_labels = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))
        normalized_frame = resized_frame.astype(np.float32) / 255.0

        frames.append(normalized_frame)
        frame_labels.append(labels[label_idx])
        label_idx += 1

        if len(frames) == chunk_size:
            yield np.array(frames), np.array(frame_labels).reshape(-1, 1)  # Reshape labels to 2D array
            frames = []
            frame_labels = []

    cap.release()

    # if len(frames) > 0:
    #     yield np.array(frames), np.array(frame_labels).reshape(-1, 1)  # Reshape labels to 2D array

for participant_id in range(8,9):
    participant_id = str(participant_id)
    left_eye_video_paths_test = [f'D:\\Carla_Scratch\\Data\\Eye_Tracker\\Eye_Tracker\\{participant_id}_M_Left.mp4']
    right_eye_video_paths_test = [f'D:\\Carla_Scratch\\Data\\Eye_Tracker\\Eye_Tracker\\{participant_id}_M_Right.mp4']
    labels_csv_paths_test = [f'D:\\Carla_Scratch\\Data\\Eye_Tracker\\Eye_Tracker\\{participant_id}_M_LabelsU.csv']

    batch_size = 10
    num_batches = int(np.ceil(len(left_eye_video_paths_test) / batch_size))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_left_paths = left_eye_video_paths_test[start_idx:end_idx]
        batch_right_paths = right_eye_video_paths_test[start_idx:end_idx]
        batch_labels_paths = labels_csv_paths_test[start_idx:end_idx]

        X_left_test = []
        X_right_test = []
        y_test = []
        for left_path, right_path, labels_path in zip(batch_left_paths, batch_right_paths, batch_labels_paths):
            left_frames_and_labels = [data for data in preprocess_video_frames_and_labels(left_path, labels_path) if len(data[0]) == 10]
            right_frames = [frames for frames, _ in preprocess_video_frames_and_labels(right_path, labels_path) if len(frames) == 10]
            
            # # Continue if frame count is mismatched
            # if len(left_frames_and_labels) != len(right_frames):
            #     print(f"Warning: Number of frames mismatch in {left_path} and {right_path}. Skipping these videos.")
            #     continue

             # Use the minimum number of frames between the two videos
            if len(left_frames_and_labels) != len(right_frames):
                print(f"Warning: Number of frames mismatch in {left_path} and {right_path}. Using the frames of the smallest one :D.")
                min_frame_count = min(len(left_frames_and_labels), len(right_frames))

                left_frames_and_labels = left_frames_and_labels[:min_frame_count]
                right_frames = right_frames[:min_frame_count]
                
            left_frames = [data[0] for data in left_frames_and_labels]
            labels = [data[1] for data in left_frames_and_labels]

            concatenated_left_frames = np.concatenate(left_frames, axis=0).astype(np.float16)
            concatenated_right_frames = np.concatenate(right_frames, axis=0).astype(np.float16)
            concatenated_labels = np.concatenate(labels, axis=0)

            print('left_frames shape:', concatenated_left_frames.shape)
            print('right_frames shape:', concatenated_right_frames.shape)
            print('labels shape:', concatenated_labels.shape)

            X_left_test.append(concatenated_left_frames)
            X_right_test.append(concatenated_right_frames)
            y_test.append(concatenated_labels)

        if len(X_left_test) > 0 and len(X_right_test) > 0:
            X_left_test = np.concatenate(X_left_test, axis=0)
            X_right_test = np.concatenate(X_right_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            print('Processing test batch id: ', batch_idx)
            print("Shape of true labels:", y_test.shape)
            print("Data type of true labels:", y_test.dtype)

            np.savez_compressed(f'test_data_batch{participant_id}.npz', X_left_test=X_left_test, X_right_test=X_right_test, y_test=y_test)

        
        # X_left_test = np.concatenate(X_left_test, axis=0)
        # X_right_test = np.concatenate(X_right_test, axis=0)
        # y_test = np.concatenate(y_test, axis=0)

        # print('Processing test batch id: ', batch_idx)
        # print("Shape of true labels:", y_test.shape)
        # print("Data type of true labels:", y_test.dtype)

        # np.savez_compressed(f'test_data_batch{participant_id}.npz', X_left_test=X_left_test, X_right_test=X_right_test, y_test=y_test)

    print("End")
