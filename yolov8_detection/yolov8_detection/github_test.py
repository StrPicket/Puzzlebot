import os
print(os.getcwd())

from ultralytics import YOLO
import cv2
import sys

try:
    # Load model
    model = YOLO("yolov8_detection/models/best.pt")

    # Load test image
    image_path = "test.jpeg"
    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Test image not found")

    # Run inference
    results = model(image)

    detected_classes = []

    with open("detection_results.txt", "w") as f:

        for r in results:

            boxes = r.boxes

            if boxes is not None:

                for box in boxes:

                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    class_name = model.names[cls_id]

                    line = f"Detected: {class_name} | confidence={conf:.2f}"

                    print(line)
                    f.write(line + "\n")

                    detected_classes.append(class_name)

    if len(detected_classes) == 0:
        print("No objects detected")
        sys.exit(1)

    print("YOLO test completed successfully")
    sys.exit(0)

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)