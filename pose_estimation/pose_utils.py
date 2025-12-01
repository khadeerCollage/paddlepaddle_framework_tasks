def draw_pose(image, keypoints, threshold=0.5):
    import cv2
    import numpy as np

    # Define the colors for the keypoints
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 255, 255)
    ]

    # Draw keypoints
    for i, (x, y, score) in enumerate(keypoints):
        if score > threshold:
            cv2.circle(image, (int(x), int(y)), 5, colors[i % len(colors)], -1)

    return image


def process_pose_results(results):
    keypoints = results['keypoint']
    scores = results['score']
    return [(kp[0], kp[1], score) for kp, score in zip(keypoints, scores)]


def visualize_pose(image, results, threshold=0.5):
    processed_results = process_pose_results(results)
    image_with_pose = draw_pose(image, processed_results, threshold)
    return image_with_pose