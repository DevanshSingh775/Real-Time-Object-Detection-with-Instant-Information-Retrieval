import cv2
import requests
from ultralytics import YOLO
from PIL import Image

# Load model and image
model = YOLO('yolov8n.pt')
img_path = "demo_image.jpg"  # Replace with your image
image = cv2.imread(img_path)

# Perform detection
results = model(image, verbose=False)

# Process results
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf < 0.5:
            continue

        # Get info from DuckDuckGo
        try:
            response = requests.get(
                f"https://api.duckduckgo.com/?q={class_name}&format=json&no_html=1",
                timeout=2
            )
            summary = response.json().get('AbstractText', 'No summary available.')
        except:
            summary = "API request failed"

        # Draw bounding box and text
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{class_name}: {summary[:60]}..."  # Truncate long text
        cv2.putText(image, text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Save and display output
cv2.imwrite("output.jpg", image)
Image.open("output.jpg").show()  # Display using PIL
