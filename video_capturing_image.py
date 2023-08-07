import cv2

video_path = ""
output_image_path = "captured_frame.jpg"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Set the current frame position to the desired frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 320)  # Change '30' to the desired frame number

# Read the frame at the current position
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error reading video frame")
    exit()

# Save the captured frame as an image
cv2.imwrite(output_image_path, frame)

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
