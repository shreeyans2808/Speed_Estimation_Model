import cv2

video_path = '/Users/shreeyansarora/Downloads/Internship_Work/Task5/traffic_video.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    ret, frame = cap.read()

    if ret:
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video Resolution: {video_width}x{video_height}")

        output_path = 'frame_0.jpg'
        cv2.imwrite(output_path, frame)

        saved_frame = cv2.imread(output_path)
        print(f"Saved Frame Resolution: {saved_frame.shape[1]}x{saved_frame.shape[0]}")
    else:
        print("Error: Could not read frame.")

cap.release()

