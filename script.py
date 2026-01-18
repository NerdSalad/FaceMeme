import cv2
import os
import threading
from deepface import DeepFace
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# Global variables for thread communication
current_expression = "Neutral"
current_confidence = 0.0
is_running = True
latest_face_roi = None
lock = threading.Lock()

# Default Settings
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360
WINDOW_NAME_CAM = "FaceMeme Camera"
WINDOW_NAME_PRED = "FaceMeme Prediction"
MEME_DIR = r"d:\Comdur\Main Projects\FaceMeme\meme"
MODEL_PATH = r"d:\Comdur\Main Projects\FaceMeme\face_meme_model.h5"

# Global Model Variables
custom_model = None
using_custom_model = False
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preload Memes

# Preload Memes
memes = {}
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def load_memes():
    global memes
    print("Loading memes...")
    for emo in emotions:
        folder_path = os.path.join(MEME_DIR, emo)
        if os.path.exists(folder_path):
            # Find first image file
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(folder_path, f)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize to fit window once to save performance
                        img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
                        memes[emo] = img
                        print(f"Loaded meme for {emo}: {f}")
                        break
    print("Meme loading complete.")

def load_custom_model():
    global custom_model, using_custom_model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Found custom model at {MODEL_PATH}. Loading...")
            custom_model = load_model(MODEL_PATH)
            using_custom_model = True
            print("Custom model loaded successfully!")
        except Exception as e:
            print(f"Error loading custom model: {e}")
            using_custom_model = False
    else:
        print("Custom model not found. Using DeepFace fallback.")
        using_custom_model = False

def face_detection(frame, face_cascade):
    """
    Detects the most prominent face in the frame.
    Returns (x, y, w, h) of the face or None if no face found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None

    # Find the largest face (most prominent)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    return largest_face

def expression_prediction():
    """
    Thread function that continuously predicts expression from latest_face_roi.
    """
    global current_expression, current_confidence, latest_face_roi, is_running

    while is_running:
        roi_to_process = None
        
        with lock:
            if latest_face_roi is not None:
                roi_to_process = latest_face_roi.copy()
        
        if roi_to_process is not None:
            try:
                if using_custom_model and custom_model is not None:
                    # Custom Model Inference (MobileNetV2: 224x224, RGB, Normalized)
                    img = cv2.resize(roi_to_process, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
                    img = img.astype("float32") / 255.0
                    img = np.expand_dims(img, axis=0) # Batch dim
                    
                    preds = custom_model.predict(img, verbose=0)
                    idx = np.argmax(preds)
                    
                    current_expression = CLASSES[idx]
                    current_confidence = float(np.max(preds)) * 100
                    
                else:
                    # DeepFace Inference
                    # actions=['emotion'] enforces emotion anaylsis
                    # enforce_detection=False because we already detected the face
                    result = DeepFace.analyze(
                        img_path=roi_to_process, 
                        actions=['emotion'], 
                        enforce_detection=False, 
                        detector_backend='opencv',
                        silent=True
                    )
                    
                    if result:
                        # DeepFace returns a list of dicts or a dict
                        first_result = result[0] if isinstance(result, list) else result
                        current_expression = first_result['dominant_emotion']
                        # Confidence is not always directly available for the emotion classification in simple mode,
                        # but we can get it from the emotion distribution if needed. 
                        # For now just using dominant string. 
                        # first_result['emotion'] is a dict like {'angry': 0.1, ...}
                        current_confidence = first_result['emotion'][current_expression]

            except Exception as e:
                # If analysis fails (e.g., face too small/blurry for model), keep previous
                # print(f"Prediction error: {e}")
                pass
        
        # dynamic sleep to prevent CPU hogging, but fast enough for updates
        time.sleep(0.1)

def render_camera_window(frame, face_coords):
    """
    Draws bounding box and annotation on the camera frame.
    Resizes the frame to fixed window size.
    """
    # Resize frame to fixed size (640x480)
    # Most webcams are 4:3, so this should look natural.
    # If the source is 16:9, this might squish slightly vertically, but acceptable.
    # To completely avoid distortion, we'd need letterboxing, but 640x480 is a good baseline.
    render_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Calculate scale factor to map face coords correctly 
    h_orig, w_orig = frame.shape[:2]
    scale_x = WINDOW_WIDTH / w_orig
    scale_y = WINDOW_HEIGHT / h_orig

    if face_coords is not None:
        x, y, w_box, h_box = face_coords
        # Scale coordinates
        x = int(x * scale_x)
        y = int(y * scale_y)
        w_box = int(w_box * scale_x)
        h_box = int(h_box * scale_y)
        
        # Draw rectangle
        cv2.rectangle(render_frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
        
    return render_frame

def render_prediction_window(expression, confidence):
    """
    Displays the meme corresponding to the expression.
    If no meme is found, falls back to text.
    """
    # Normalize expression string (DeepFace returns lower case usually, but we ensure consistency)
    emo_key = expression.lower()
    
    # Check if we have a meme for this emotion
    if emo_key in memes:
        img = memes[emo_key].copy()
    else:
        # Fallback to black screen with text
        img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        text = expression.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (WINDOW_WIDTH - text_size[0]) // 2
        text_y = (WINDOW_HEIGHT + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Always overlay confidence/label slightly so user knows what's happening
    # Draw a semi-transparent overlay at the bottom? Or just simple text.
    # Simple text at top or bottom is good.
    label_text = f"{expression.upper()} ({confidence:.1f}%)"
    
    # Indicate which model is being used
    model_src = "Custom" if using_custom_model else "DeepFace"
    cv2.putText(img, f"[{model_src}]", (WINDOW_WIDTH - 120, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    
    cv2.putText(img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 255), 2) # Yellow text
    
    return img

def main():
    global latest_face_roi, is_running

    # Initialize Face Detector (Haar Cascade is fast/reliable enough for bounding box)
    # Using cv2.data.haarcascades to find the xml file
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Load Memes
    load_memes()
    
    # Try to load custom model
    load_custom_model()

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Start Prediction Thread
    pred_thread = threading.Thread(target=expression_prediction, daemon=True)
    pred_thread.start()

    # Setup Windows
    cv2.namedWindow(WINDOW_NAME_CAM)
    cv2.namedWindow(WINDOW_NAME_PRED)
    
    # Move windows to be side-by-side
    # Coordinates (x, y) on screen
    cv2.moveWindow(WINDOW_NAME_CAM, 100, 100)
    cv2.moveWindow(WINDOW_NAME_PRED, 100 + WINDOW_WIDTH + 20, 100)

    print("Starting Main Loop... Press 'q' to quit.")

    while True:
        # Check if windows are closed (clicked X button)
        try:
            if cv2.getWindowProperty(WINDOW_NAME_CAM, cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty(WINDOW_NAME_PRED, cv2.WND_PROP_VISIBLE) < 1:
                is_running = False
                break
        except Exception:
            pass

        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame (horizontal flip)
        frame = cv2.flip(frame, 1)

        # 1. Detect Face
        face = face_detection(frame, face_cascade)

        # 2. Update Global Face ROI
        with lock:
            if face is not None:
                x, y, w, h = face
                h_img, w_img = frame.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w_img, x + w)
                y2 = min(h_img, y + h)
                latest_face_roi = frame[y1:y2, x1:x2]
            else:
                latest_face_roi = None

        # 3. Render Windows
        cam_view = render_camera_window(frame, face)
        cv2.imshow(WINDOW_NAME_CAM, cam_view)

        if face is None:
            pred_view = render_prediction_window("NO FACE", 0.0)
        else:
            pred_view = render_prediction_window(current_expression, current_confidence)
            
        cv2.imshow(WINDOW_NAME_PRED, pred_view)

        # FPS Control / Input
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pred_thread.join()

if __name__ == "__main__":
    main()
