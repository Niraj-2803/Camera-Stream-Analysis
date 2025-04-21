import cv2

def blur_faces_from_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    window_name = "RTSP Stream"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0
    faces = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No frame received")
            break

        frame_count += 1

        # Resize for faster face detection
        small_frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect every 3rd frame
        if frame_count % 3 == 0:
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # Scale face coords to original size
        x_scale = frame.shape[1] / 640
        y_scale = frame.shape[0] / 360

        for (x, y, w, h) in faces:
            x = int(x * x_scale)
            y = int(y * y_scale)
            w = int(w * x_scale)
            h = int(h * y_scale)

            face_region = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)
            frame[y:y+h, x:x+w] = blurred_face

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
