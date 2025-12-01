"""
Object Detection with PaddlePaddle
==================================
This module demonstrates object detection concepts and provides
utilities for working with pretrained detection models.

Key Concepts:
1. Backbone: Extracts features from images
2. Detection Head: Predicts bounding boxes and class labels
3. IoU: Measures overlap between boxes
4. NMS: Removes duplicate detections
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================================
# PART 1: BOUNDING BOX UTILITIES
# ============================================================================

class BoxUtils:
    """Utility functions for bounding box operations"""
    
    @staticmethod
    def xyxy_to_xywh(boxes):
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]"""
        boxes = np.array(boxes)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.stack([cx, cy, w, h], axis=1)
    
    @staticmethod
    def xywh_to_xyxy(boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        boxes = np.array(boxes)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)
    
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU)
        
        IoU is the key metric for object detection:
        - IoU > 0.5: Usually considered a good detection
        - IoU > 0.75: High quality detection
        """
        # Get intersection coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        inter_w = max(0, x2_inter - x1_inter)
        inter_h = max(0, y2_inter - y1_inter)
        intersection = inter_w * inter_h
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    @staticmethod
    def apply_nms(boxes, scores, iou_thresh=0.5):
        """
        Non-Maximum Suppression (NMS)
        
        Removes overlapping boxes, keeping only the best ones.
        This prevents multiple detections for the same object.
        """
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Sort by score (highest first)
        order = scores.argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            # Keep the highest scoring box
            idx = order[0]
            keep.append(idx)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with remaining boxes
            remaining = order[1:]
            ious = np.array([
                BoxUtils.calculate_iou(boxes[idx], boxes[i]) 
                for i in remaining
            ])
            
            # Keep boxes with IoU below threshold
            order = remaining[ious < iou_thresh]
        
        return keep


# ============================================================================
# PART 2: SIMPLE DETECTOR BACKBONE
# ============================================================================

class DetectorBackbone(nn.Layer):
    """
    Feature extraction backbone for object detection.
    
    Extracts multi-scale features:
    - Large feature maps: Good for small objects
    - Small feature maps: Good for large objects
    """
    
    def __init__(self):
        super(DetectorBackbone, self).__init__()
        
        # Stage 1: Input -> 1/2 size
        self.stage1 = nn.Sequential(
            nn.Conv2D(3, 32, 3, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.MaxPool2D(2, 2)
        )
        
        # Stage 2: 1/2 -> 1/4 size
        self.stage2 = nn.Sequential(
            nn.Conv2D(32, 64, 3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(2, 2)
        )
        
        # Stage 3: 1/4 -> 1/8 size
        self.stage3 = nn.Sequential(
            nn.Conv2D(64, 128, 3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.MaxPool2D(2, 2)
        )
        
        # Stage 4: 1/8 -> 1/16 size
        self.stage4 = nn.Sequential(
            nn.Conv2D(128, 256, 3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.MaxPool2D(2, 2)
        )
        
        # Stage 5: 1/16 -> 1/32 size
        self.stage5 = nn.Sequential(
            nn.Conv2D(256, 512, 3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.MaxPool2D(2, 2)
        )
    
    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


class DetectionHead(nn.Layer):
    """
    Detection head that predicts boxes and classes.
    
    For each grid cell, predicts:
    - 4 box coordinates (x, y, w, h)
    - 1 objectness score
    - N class probabilities
    """
    
    def __init__(self, in_ch, num_classes, num_anchors=3):
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Output: anchors * (4 coords + 1 objectness + num_classes)
        out_ch = num_anchors * (5 + num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2D(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2D(in_ch),
            nn.ReLU(),
            nn.Conv2D(in_ch, out_ch, 1)
        )
    
    def forward(self, x):
        return self.conv(x)


class SimpleDetector(nn.Layer):
    """
    Simple YOLO-style object detector.
    
    Multi-scale detection:
    - 13x13 grid: Large objects
    - 26x26 grid: Medium objects  
    - 52x52 grid: Small objects
    """
    
    def __init__(self, num_classes=80):
        super(SimpleDetector, self).__init__()
        self.backbone = DetectorBackbone()
        self.head_s = DetectionHead(512, num_classes)  # Small grid
        self.head_m = DetectionHead(256, num_classes)  # Medium grid
        self.head_l = DetectionHead(128, num_classes)  # Large grid
    
    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        out_s = self.head_s(c5)
        out_m = self.head_m(c4)
        out_l = self.head_l(c3)
        return out_s, out_m, out_l


# ============================================================================
# PART 3: VISUALIZATION
# ============================================================================

class DetectionVisualizer:
    """Visualize detection results"""
    
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    @staticmethod
    def draw_boxes(image, boxes, labels, scores, class_names=None):
        """Draw bounding boxes on image"""
        img = image.copy()
        
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = map(int, box)
            color = DetectionVisualizer.COLORS[label % len(DetectionVisualizer.COLORS)]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if class_names:
                text = f'{class_names[label]}: {score:.2f}'
            else:
                text = f'Class {label}: {score:.2f}'
            
            cv2.putText(img, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img
    
    @staticmethod
    def show_detections(image, boxes, labels, scores, class_names=None):
        """Display image with detections"""
        img = DetectionVisualizer.draw_boxes(image, boxes, labels, scores, class_names)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


# ============================================================================
# PART 4: USING PADDLEDETECTION (Recommended for Real Use)
# ============================================================================

def run_paddledetection_inference(image_path, model_dir):
    """
    Run inference using PaddleDetection pretrained model.
    
    This is the recommended way for real-world applications.
    """
    import sys
    sys.path.insert(0, 'PaddleDetection')
    
    from deploy.python.infer import Detector
    
    detector = Detector(model_dir=model_dir, device='GPU')
    image = cv2.imread(image_path)
    results = detector.predict_image([image[..., ::-1]], visual=False)
    
    return results


def download_and_setup_paddledetection():
    """Instructions for setting up PaddleDetection"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           PaddleDetection Setup Instructions                 ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Step 1: Clone Repository                                    ║
    ║  ─────────────────────────                                   ║
    ║  git clone https://github.com/PaddlePaddle/PaddleDetection   ║
    ║  cd PaddleDetection                                          ║
    ║  pip install -r requirements.txt                             ║
    ║                                                              ║
    ║  Step 2: Export Pretrained Model                             ║
    ║  ────────────────────────────────                            ║
    ║  python tools/export_model.py \\                              ║
    ║      -c configs/ppyoloe/ppyoloe_crn_l_300e_coco.yml \\        ║
    ║      --output_dir=./inference_model \\                        ║
    ║      -o weights=https://paddledet.bj.bcebos.com/models/\\     ║
    ║         ppyoloe_crn_l_300e_coco.pdparams                     ║
    ║                                                              ║
    ║  Step 3: Run Inference                                       ║
    ║  ─────────────────────                                       ║
    ║  python deploy/python/infer.py \\                             ║
    ║      --model_dir=./inference_model/ppyoloe_crn_l_300e_coco \\ ║
    ║      --image_file=./test.jpg                                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


# ============================================================================
# PART 5: DEMO AND TESTING
# ============================================================================

def demo_iou_calculation():
    """Demonstrate IoU calculation"""
    print("\n=== IoU Calculation Demo ===\n")
    
    box1 = [100, 100, 200, 200]  # Ground truth
    box2 = [150, 150, 250, 250]  # Prediction (partial overlap)
    box3 = [100, 100, 200, 200]  # Perfect match
    box4 = [300, 300, 400, 400]  # No overlap
    
    iou1 = BoxUtils.calculate_iou(box1, box2)
    iou2 = BoxUtils.calculate_iou(box1, box3)
    iou3 = BoxUtils.calculate_iou(box1, box4)
    
    print(f"Box1: {box1}")
    print(f"Box2 (partial overlap): {box2} -> IoU: {iou1:.4f}")
    print(f"Box3 (perfect match):   {box3} -> IoU: {iou2:.4f}")
    print(f"Box4 (no overlap):      {box4} -> IoU: {iou3:.4f}")


def demo_nms():
    """Demonstrate Non-Maximum Suppression"""
    print("\n=== NMS Demo ===\n")
    
    # Multiple overlapping detections for same object
    boxes = [
        [100, 100, 200, 200],  # Detection 1
        [105, 105, 205, 205],  # Detection 2 (overlaps with 1)
        [110, 110, 210, 210],  # Detection 3 (overlaps with 1, 2)
        [300, 300, 400, 400],  # Detection 4 (different object)
    ]
    scores = [0.9, 0.85, 0.7, 0.95]
    
    print(f"Before NMS: {len(boxes)} boxes")
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  Box {i}: {box}, Score: {score}")
    
    keep_indices = BoxUtils.apply_nms(boxes, scores, iou_thresh=0.5)
    
    print(f"\nAfter NMS: {len(keep_indices)} boxes")
    for idx in keep_indices:
        print(f"  Box {idx}: {boxes[idx]}, Score: {scores[idx]}")


def demo_detector_model():
    """Demonstrate simple detector forward pass"""
    print("\n=== Simple Detector Demo ===\n")
    
    model = SimpleDetector(num_classes=80)
    
    # Create dummy input (batch=1, channels=3, height=416, width=416)
    dummy_input = paddle.randn([1, 3, 416, 416])
    
    # Forward pass
    out_s, out_m, out_l = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shapes:")
    print(f"  Small objects (13x13):  {out_s.shape}")
    print(f"  Medium objects (26x26): {out_m.shape}")
    print(f"  Large objects (52x52):  {out_l.shape}")


def create_sample_detection_image():
    """Create a sample image with detections for visualization"""
    print("\n=== Visualization Demo ===\n")
    
    # Create blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some shapes to detect
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)   # Red square
    cv2.rectangle(img, (200, 100), (350, 250), (0, 255, 0), -1) # Green rectangle
    cv2.circle(img, (500, 200), 80, (255, 0, 0), -1)            # Blue circle
    
    # Simulated detections
    boxes = [
        [45, 45, 155, 155],   # Red square detection
        [195, 95, 355, 255],  # Green rectangle detection
        [415, 115, 585, 285], # Blue circle detection
    ]
    labels = [0, 1, 2]
    scores = [0.95, 0.88, 0.92]
    class_names = ['square', 'rectangle', 'circle']
    
    # Visualize
    DetectionVisualizer.show_detections(img, boxes, labels, scores, class_names)
    print("Detection visualization displayed!")


def main():
    """Main function to run all demos"""
    print("=" * 60)
    print("   OBJECT DETECTION TUTORIAL WITH PADDLEPADDLE")
    print("=" * 60)
    
    # Run demos
    demo_iou_calculation()
    demo_nms()
    demo_detector_model()
    
    # Show PaddleDetection setup instructions
    download_and_setup_paddledetection()
    
    # Uncomment to show visualization (requires display)
    # create_sample_detection_image()
    
    print("\n" + "=" * 60)
    print("   Tutorial Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
