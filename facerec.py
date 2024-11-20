

import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, img = webcam.read()
    
    # Check if frame is captured successfully
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    
    # Draw rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Display the result
    cv2.imshow("Face Detection", img)
    
    # Exit on 'ESC' key press
    key = cv2.waitKey(10)
    if key == 27:  # ESC key
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
