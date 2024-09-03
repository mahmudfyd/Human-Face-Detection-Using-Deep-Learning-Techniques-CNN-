import cv2
from keras.models import model_from_json
import numpy as np
import sys

# Force UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Load the model
json_file = open("facedetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facedetector.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    image = cv2.resize(image, (96, 96))
    feature = np.array(image)
    feature = feature.reshape(1, 96, 96, 3)
    return feature / 255.0

# Start video capture
webcam = cv2.VideoCapture(0)

labels = {0: 'Human', 1: 'non_Human'}

# Define the fixed position and size for the bounding box
frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fixed_box_size = 200
fixed_x = (frame_width - fixed_box_size) // 2
fixed_y = (frame_height - fixed_box_size) // 2

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Draw the fixed bounding box in the middle
    cv2.rectangle(frame, (fixed_x, fixed_y), (fixed_x + fixed_box_size, fixed_y + fixed_box_size), (0, 0, 255), 2)

    # Detect faces in the entire frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Initialize flag to check if a human face is detected within the bounding box
    detected_human = False

    for (x, y, w, h) in faces:
        # Check if the detected face is within the fixed bounding box
        if (fixed_x <= x <= fixed_x + fixed_box_size) and (fixed_y <= y <= fixed_y + fixed_box_size) and \
           (fixed_x <= x + w <= fixed_x + fixed_box_size) and (fixed_y <= y + h <= fixed_y + fixed_box_size):
            detected_human = True
            face_region = frame[y:y+h, x:x+w]
            img = extract_features(face_region)
            pred = model.predict(img)
            
            label = labels[np.argmax(pred)]
            
            # Put label inside the fixed bounding box
            cv2.putText(frame, label, (fixed_x, fixed_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # If no human detected in the bounding box, show "non_Human"
    if not detected_human:
        cv2.putText(frame, 'non_Human', (fixed_x, fixed_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

webcam.release()
cv2.destroyAllWindows()
