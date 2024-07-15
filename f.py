#pip install opencv-python deepface
#pip install tf-keras

import cv2
from deepface import DeepFace

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        try:
            # Analyze the face using DeepFace, with enforce_detection=False
            analysis = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)

            # Access the first element of the analysis list
            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]

                # Correctly access the dominant emotion from the analysis result
                emotion = analysis['dominant_emotion']
                
                # Display the detected emotion on the frame
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error analyzing face:", e)
    
    cv2.imshow('Facial Expression Detection', frame)
    
    # Check for user input to quit (press 'q' key)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




