import torch
import cv2
import mediapipe as mp
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower

MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 2500

device = torch.device("mps")  # or "cuda" or "cpu"

# Configuration for the three toy policies
TOYS = [
    {"name": "grey", "model_id": "path/to/grey-toy-model", "task": "pick up the grey toy on the right side and place it on the left side"},
    {"name": "purple", "model_id": "path/to/purple-toy-model", "task": "pick up the purple toy on the right side and place it on the left side"},
    {"name": "orange", "model_id": "path/to/orange-toy-model", "task": "pick up the orange toy on the right side and place it on the left side"},
]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture hold duration requirement (seconds)
HOLD_DURATION = 2.0

def detect_thumbs_gesture(webcam_index=0, timeout=15):
    """
    Detect thumbs up or thumbs down from webcam using MediaPipe.
    Requires the gesture to be held continuously for HOLD_DURATION seconds.
    
    Args:
        webcam_index: Index of the laptop webcam (usually 0)
        timeout: Maximum time to wait for gesture detection (seconds)
    
    Returns:
        "thumbs_up", "thumbs_down", or None if timeout
    """
    cap = cv2.VideoCapture(webcam_index)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam at index {webcam_index}")
        return None
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        start_time = time.time()
        detected_gesture = None
        
        # Variables to track gesture holding
        current_gesture = None
        gesture_start_time = None
        
        print(f"Show thumbs up or thumbs down gesture and HOLD for {HOLD_DURATION} seconds...")
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = hands.process(rgb_frame)
            
            frame_gesture = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    # Detect thumbs up/down
                    frame_gesture = classify_thumb_gesture(hand_landmarks)
            
            # Track gesture holding
            if frame_gesture:
                # New gesture detected or same gesture continues
                if frame_gesture == current_gesture:
                    # Same gesture - check if held long enough
                    hold_time = time.time() - gesture_start_time
                    
                    # Display progress
                    progress_text = f"Hold {frame_gesture.replace('_', ' ').upper()}: {hold_time:.1f}/{HOLD_DURATION}s"
                    cv2.putText(
                        frame, progress_text,
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 0), 2
                    )
                    
                    # Draw progress bar
                    bar_width = int((hold_time / HOLD_DURATION) * 600)
                    cv2.rectangle(frame, (10, 70), (610, 100), (255, 255, 255), 2)
                    cv2.rectangle(frame, (10, 70), (10 + bar_width, 100), (0, 255, 0), -1)
                    
                    if hold_time >= HOLD_DURATION:
                        detected_gesture = frame_gesture
                        cv2.putText(
                            frame, f"CONFIRMED: {frame_gesture.replace('_', ' ').upper()}", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.2, (0, 255, 0), 3
                        )
                        cv2.imshow('Gesture Detection', frame)
                        cv2.waitKey(1)
                        time.sleep(1)  # Show confirmation for 1 second
                        break
                else:
                    # Different gesture - restart tracking
                    current_gesture = frame_gesture
                    gesture_start_time = time.time()
                    print(f"Detected {frame_gesture.replace('_', ' ')} - hold for {HOLD_DURATION} seconds...")
            else:
                # No gesture detected - reset tracking
                if current_gesture is not None:
                    print("Gesture lost - please try again")
                current_gesture = None
                gesture_start_time = None
                
                cv2.putText(
                    frame, "Show gesture...",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2
                )
            
            # Display frame
            cv2.imshow('Gesture Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Gesture detection cancelled by user")
                break
            
            if detected_gesture:
                break
        
        # Clean up - CRITICAL: Release camera and close windows
        cap.release()
        cv2.destroyAllWindows()
        
        # Give a moment for windows to close properly
        cv2.waitKey(1)
        
        if detected_gesture:
            print(f"✓ Gesture confirmed: {detected_gesture.replace('_', ' ').upper()}")
        else:
            print("✗ No gesture detected within timeout period")
        
        return detected_gesture

def classify_thumb_gesture(hand_landmarks):
    """
    Classify thumb gesture based on hand landmarks.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
    
    Returns:
        "thumbs_up", "thumbs_down", or None
    """
    # Get landmark coordinates
    landmarks = hand_landmarks.landmark
    
    # Key points: thumb tip, thumb ip, wrist, and other fingertips
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    
    # Check if other fingers are folded (y-coordinates close to palm)
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]
    
    # Check if fingers are folded (tips are below/close to MCPs)
    fingers_folded = (
        abs(index_tip.y - index_mcp.y) < 0.1 and
        abs(middle_tip.y - middle_mcp.y) < 0.1 and
        abs(ring_tip.y - ring_mcp.y) < 0.1 and
        abs(pinky_tip.y - pinky_mcp.y) < 0.1
    )
    
    if fingers_folded:
        # Thumbs up: thumb tip is above wrist
        if thumb_tip.y < wrist.y - 0.1:
            return "thumbs_up"
        # Thumbs down: thumb tip is below wrist
        elif thumb_tip.y > wrist.y + 0.1:
            return "thumbs_down"
    
    return None

def load_policy(model_id, device):
    """Load a SmolVLA policy and its pre/post processors."""
    print(f"Loading policy: {model_id}")
    model = SmolVLAPolicy.from_pretrained(model_id)
    
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    
    return model, preprocess, postprocess

def run_policy(robot, model, preprocess, postprocess, task, robot_type, dataset_features, device):
    """Run a policy for one episode."""
    print(f"Running policy for task: {task}")
    
    for step in range(MAX_STEPS_PER_EPISODE):
        obs = robot.get_observation()
        obs_frame = build_inference_frame(
            observation=obs, 
            ds_features=dataset_features, 
            device=device, 
            task=task, 
            robot_type=robot_type
        )

        obs = preprocess(obs_frame)
        action = model.select_action(obs)
        action = postprocess(action)
        action = make_robot_action(action, dataset_features)
        robot.send_action(action)
    
    print(f"Policy execution finished for: {task}")

# Main execution
def main():
    # Robot setup
    follower_port = "/dev/tty.usbmodem5AB01814981"
    follower_id = "so101_right_follower"
    
    camera_config = {
        "overhead": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
        "wrist": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
    }
    
    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO100Follower(robot_cfg)
    robot.connect()
    
    robot_type = "so101_follower"
    
    # Setup dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    # Load all three policies
    policies = {}
    for toy in TOYS:
        model, preprocess, postprocess = load_policy(toy["model_id"], device)
        policies[toy["name"]] = {
            "model": model,
            "preprocess": preprocess,
            "postprocess": postprocess,
            "task": toy["task"]
        }
    
    print("\n" + "="*50)
    print("Starting toy pickup sequence")
    print("="*50 + "\n")
    
    # Main loop: ask about each toy
    for toy in TOYS:
        print(f"\n--- {toy['name'].upper()} TOY ---")
        print(f"Do you want to pick up the {toy['name']} toy?")
        print(f"Show and HOLD THUMBS UP for YES or THUMBS DOWN for NO ({HOLD_DURATION} seconds)")
        
        gesture = detect_thumbs_gesture(webcam_index=0, timeout=15)
        
        if gesture == "thumbs_up":
            print(f"✓ Thumbs up confirmed! Running {toy['name']} toy policy...")
            policy = policies[toy["name"]]
            run_policy(
                robot, 
                policy["model"], 
                policy["preprocess"], 
                policy["postprocess"],
                policy["task"],
                robot_type,
                dataset_features,
                device
            )
        elif gesture == "thumbs_down":
            print(f"✗ Thumbs down confirmed! Skipping {toy['name']} toy...")
        else:
            print(f"⚠ No gesture detected in time. Skipping {toy['name']} toy...")
        
        time.sleep(2)  # Brief pause between toys
    
    print("\n" + "="*50)
    print("All toys processed!")
    print("="*50)
    
    robot.disconnect()

if __name__ == "__main__":
    main()