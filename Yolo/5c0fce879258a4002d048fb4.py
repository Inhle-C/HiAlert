import cv2
import numpy as np
import json
from ultralytics import YOLO
import math
from datetime import datetime

# Load the JSON data
with open('../Json files/5c0fce879258a4002d048fb4.json', 'r') as f:json_data = json.load(f)

# Get all actors and their trajectories
actors = json_data['actors']

# Create a mapping of actor types to colors
actor_colors = {
    'TRUCK': (0, 0, 255),  # Red
    'CAR': (0, 255, 0),  # Green
    'MOTORCYCLE': (255, 0, 0),  # Blue
    'BICYCLE': ( 555,0,0),
     'PEOPLE':(0,0,0)
}

# Create a mapping of involvement to thickness
involvement_thickness = {
    'CAUSER': 3,
    'VICTIM': 2,
    'NONE': 1
}


# Function to convert position [longitude, latitude, z] to pixel coordinates
def pos_to_pixel(pos, min_long, max_long, min_lat, max_lat, frame_width, frame_height):
    # Extract longitude and latitude
    long, lat = pos[0], pos[1]

    # Scale to pixel coordinates
    x = int((long - min_long) / (max_long - min_long) * frame_width)
    y = int((1 - (lat - min_lat) / (max_lat - min_lat)) * frame_height)

    return x, y


# Function to render a frame of the accident visualization
def render_accident_frame(frame_num, min_long, max_long, min_lat, max_lat, frame_width, frame_height):
    # Create a blank canvas
    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Draw each actor on the canvas
    for actor in actors:
        trajectory = actor['trajectory']

        # Get the actor's position at this frame (if it exists)
        frame_data = None
        for point in trajectory:
            if point['frame'] == frame_num:
                frame_data = point
                break

        if frame_data:
            # Get actor type and involvement
            actor_type = actor['type']
            involvement = actor['involvement']

            # Get color and thickness
            color = actor_colors.get(actor_type, (255, 255, 255))  # Default to white
            thickness = involvement_thickness.get(involvement, 1)

            # Get position and convert to pixel coordinates
            pos = frame_data['position']
            x, y = pos_to_pixel(pos, min_long, max_long, min_lat, max_lat, frame_width, frame_height)

            # Draw a circle to represent the actor
            cv2.circle(canvas, (x, y), 5, color, thickness)

            # Draw rotated triangle to indicate heading (rotation[1] represents yaw)
            rotation = frame_data['rotation']
            yaw = rotation[1]  # Rotation about y-axis

            # Calculate points of triangle
            r = 15  # Size of triangle
            angle = yaw
            x1 = int(x + r * math.cos(angle))
            y1 = int(y + r * math.sin(angle))

            angle_left = angle + 2.5
            x2 = int(x + r * 0.6 * math.cos(angle_left))
            y2 = int(y + r * 0.6 * math.sin(angle_left))

            angle_right = angle - 2.5
            x3 = int(x + r * 0.6 * math.cos(angle_right))
            y3 = int(y + r * 0.6 * math.sin(angle_right))

            triangle_pts = np.array([[x1, y1], [x2, y2], [x3, y3]])
            cv2.fillPoly(canvas, [triangle_pts], color)

            # Add label
            label = f"{actor_type} ({involvement})"
            cv2.putText(canvas, label, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Add frame info
    cv2.putText(canvas, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add time info if available
    if 'timestamps' in json_data['scenario']:
        t0 = json_data['scenario']['timestamps']['T0']
        t6 = json_data['scenario']['timestamps']['T6']
        duration = t6 - t0
        time_in_seconds = t0 + (frame_num / max_frame_num) * duration
        time_str = f"Time: {time_in_seconds:.1f}s"
        cv2.putText(canvas, time_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return canvas


# Function to find min/max coordinates
def find_coordinate_bounds(actors):
    all_longs = []
    all_lats = []
    max_frame = 0

    for actor in actors:
        trajectory = actor['trajectory']
        for point in trajectory:
            all_longs.append(point['position'][0])
            all_lats.append(point['position'][1])
            max_frame = max(max_frame, point['frame'])

    # Apply a margin to the bounding box
    margin = 0.00005  # Adjust as needed
    min_long = min(all_longs) - margin
    max_long = max(all_longs) + margin
    min_lat = min(all_lats) - margin
    max_lat = max(all_lats) + margin

    return min_long, max_long, min_lat, max_lat, max_frame


# Find coordinate bounds and max frame
min_long, max_long, min_lat, max_lat, max_frame_num = find_coordinate_bounds(actors)

# Set up visualization parameters
viz_width = 600  # Visualization window width
viz_height = 400  # Visualization window height

# Load YOLO model
model = YOLO('../Yolo-Weights/yolov8l.pt')

# Video file or stream URL
video_source = "https://rr-traffic-video-stream.s3.us-east-1.amazonaws.com/2023/video-1680521897662.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIA2MJP5IYU7K4O4LPV%2F20250307%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250307T213130Z&X-Amz-Expires=600000&X-Amz-Signature=e9b1c3594202dc0c1497f98d53456b7b06e8b01bfe06d739eda5440aca972422&X-Amz-SignedHeaders=host&response-content-type=video%2Fmp4&x-id=GetObject"

# Create video capture
cap = cv2.VideoCapture(video_source)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Get video properties
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate accident animation speed
# Match the animation frames to the video duration
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
animation_step = max_frame_num / total_video_frames

# Frame counter
frame_counter = 0
accident_frame = 0

print(f"Video has {total_video_frames} frames at {video_fps} FPS")
print(f"Animation has {max_frame_num} frames, step size: {animation_step}")

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("End of video stream.")
        break

    # Process frame with YOLO
    results = model.predict(frame, verbose=False)[0]  # Set verbose=False to reduce console output

    # Visualize results on the frame
    annotated_frame = results.plot()

    # Get current accident frame
    current_accident_frame = int(frame_counter * animation_step)
    if current_accident_frame > max_frame_num:
        current_accident_frame = max_frame_num

    # Render accident visualization
    accident_viz = render_accident_frame(
        current_accident_frame,
        min_long, max_long,
        min_lat, max_lat,
        viz_width, viz_height
    )

    # Add timestamp to both visualizations
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    cv2.putText(annotated_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add frame counter to both visualizations
    frame_info = f"Video Frame: {frame_counter}/{total_video_frames}"
    cv2.putText(annotated_frame, frame_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Create side-by-side display
    display_height = max(video_height, viz_height)
    display = np.zeros((display_height, video_width + viz_width, 3), dtype=np.uint8)

    # Place the YOLO processed frame on the left
    display[0:video_height, 0:video_width] = annotated_frame

    # Place the accident visualization on the right
    viz_y_offset = (display_height - viz_height) // 2  # Center vertically if needed
    display[viz_y_offset:viz_y_offset + viz_height, video_width:video_width + viz_width] = accident_viz

    # Add dividing line
    cv2.line(display, (video_width, 0), (video_width, display_height), (255, 255, 255), 2)

    # Show the combined display
    cv2.imshow("Traffic Analysis", display)

    # Increment frame counter
    frame_counter += 1

    # Wait for key press (adjust for target FPS)
    if cv2.waitKey(int(1000 / video_fps)) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()