import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
from datetime import datetime
import logging
import pandas as pd
import io
import base64
import tempfile
import time
from streamlit_option_menu import option_menu

# Configure page settings
st.set_page_config(
    page_title="Face Analysis Pro",
    page_icon="👤",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #9370DB !important;
        font-size: 3rem !important;
        font-weight: bold !important;
        padding-bottom: 2rem;
    }
    .stSubheader {
        color: #6A5ACD !important;
        font-weight: bold !important;
    }
    .stButton button {
        background-color: #9370DB;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #6A5ACD;
    }
    .metrics-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'stop_video' not in st.session_state:
    st.session_state.stop_video = False

def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def analyze_face(image):
    try:
        results = DeepFace.analyze(image, 
                                 actions=["age", "gender", "emotion", "race"],
                                 enforce_detection=False)
        return results if isinstance(results, list) else [results]
    except Exception as e:
        logger.error(f"Face analysis error: {str(e)}")
        return []

def process_frame(frame):
    try:
        results = analyze_face(frame)
        
        for result in results:
            x, y, w, h = result["region"]["x"], result["region"]["y"], result["region"]["w"], result["region"]["h"]
            
            # Get the gender with the highest probability
            gender, probability = max(result['gender'].items(), key=lambda x: x[1])
            
            # Create color based on gender and emotion
            gender_color = (211, 147, 255) if gender == "Male" else (255, 147, 211)
            
            emotion_color = {
                "angry": (255, 0, 0),
                "disgust": (0, 255, 0),
                "fear": (0, 0, 255),
                "happy": (255, 255, 0),
                "sad": (255, 0, 255),
                "surprise": (0, 255, 255),
                "neutral": (128, 128, 128)
            }.get(result["dominant_emotion"], (128, 128, 128))
            
            # Draw rectangle and add text
            cv2.rectangle(frame, (x, y), (x+w, y+h), gender_color, 2)
            cv2.rectangle(frame, (x, y-80), (x+w, y), gender_color, -1)
            
            # Add text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{gender}: {probability:.2f}%", (x+5, y-60), font, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Age: {int(result['age'])}", (x+5, y-40), font, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Emotion: {result['dominant_emotion']}", (x+5, y-20), font, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Race: {result['dominant_race']}", (x+5, y), font, 0.6, (255,255,255), 1)
            
        return frame, results
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}")
        return frame, []

def process_image(image):
    np_img = np.array(image.convert("RGB"))
    image_rgb = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    
    results = analyze_face(np_img)
    
    if not results:
        st.warning("No faces detected in the image. Please try another image.")
        return None, []
    
    for result in results:
        x, y, w, h = result["region"]["x"], result["region"]["y"], result["region"]["w"], result["region"]["h"]
        
        # Get the gender with the highest probability
        gender, probability = max(result['gender'].items(), key=lambda x: x[1])
        
        gender_color = (211, 147, 255) if gender == "Male" else (255, 147, 211)
        
        emotion_color = {
            "angry": (255, 0, 0),
            "disgust": (0, 255, 0),
            "fear": (0, 0, 255),
            "happy": (255, 255, 0),
            "sad": (255, 0, 255),
            "surprise": (0, 255, 255),
            "neutral": (128, 128, 128)
        }.get(result["dominant_emotion"], (128, 128, 128))
        
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), gender_color, 2)
        cv2.rectangle(image_rgb, (x, y-80), (x+w, y), gender_color, -1)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_rgb, f"{gender}: {probability:.2f}%", (x+5, y-60), font, 0.6, (255,255,255), 1)
        cv2.putText(image_rgb, f"Age: {int(result['age'])}", (x+5, y-40), font, 0.6, (255,255,255), 1)
        cv2.putText(image_rgb, f"Emotion: {result['dominant_emotion']}", (x+5, y-20), font, 0.6, (255,255,255), 1)
        cv2.putText(image_rgb, f"Race: {result['dominant_race']}", (x+5, y), font, 0.6, (255,255,255), 1)
    
    processed_image = Image.fromarray(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))
    return processed_image, results

# Sidebar navigation
with st.sidebar:
    st.image("https://www.svgrepo.com/show/529279/user-circle.svg", width=100)
    with st.container():
        st.markdown("<h2 style='color: #9370DB; font-weight: bold;'>Face Analysis Pro</h2>", unsafe_allow_html=True)
        st.write("Analyze faces in images and video for age, gender, emotion, and race.")
        selected = option_menu(
            menu_title=None,
            options=["Home", "Video Analysis", "Analysis History", "About"],
            icons=["house", "camera-video", "clock-history", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5!important"},
                "icon": {"color": "#9370DB", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color": "#6A5ACD"},
                "nav-link-selected": {"background-color": "#9370DB", "color": "white"},
            }
        )

# Video analysis function
def video_analysis():
    st.title("Real-time Video Analysis")
    
    # Video capture options
    if st.button("Start Webcam"):
        st.session_state.video_capture = cv2.VideoCapture(0)
    
    # Create placeholder for video feed
    video_placeholder = st.empty()
    
    # Create stop button
    stop_button = st.button("Stop Video")
    
    # Video processing loop
    while st.session_state.video_capture is not None and st.session_state.video_capture.isOpened():
        ret, frame = st.session_state.video_capture.read()
        
        if not ret or stop_button:
            st.session_state.video_capture.release()
            break
        
        # Process frame
        processed_frame, results = process_frame(frame)
        
        # Convert BGR to RGB for display
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Add small delay to reduce CPU usage
        time.sleep(0.1)

# Main application logic
if selected == "Home":
    st.title("Face Analysis Pro")
    st.write("Upload an image to analyze faces for age, gender, emotion, and race.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        with st.spinner('Processing image...'):
            col1, col2 = st.columns(2)
            
            image = Image.open(uploaded_file)
            col1.subheader("Original Image")
            col1.image(image, use_container_width=True)
            
            processed_image, results = process_image(image)
            
            if processed_image and results:
                col2.subheader("Analyzed Image")
                col2.image(processed_image, use_container_width=True)
                
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'results': results,
                    'processed_image': processed_image
                })
                
                st.markdown(
                    get_image_download_link(processed_image, "analyzed_image.png", "Download Analyzed Image"),
                    unsafe_allow_html=True
                )
                
                st.subheader("Analysis Results")
                
                for i, result in enumerate(results):
                    cols = st.columns(4)
                    
                    # Display the gender with the highest probability
                    gender, probability = max(result['gender'].items(), key=lambda x: x[1])
                    with cols[0]:
                        st.metric("Gender", f"{gender}: {probability:.2f}%")
                    with cols[1]:
                        st.metric("Age", f"{int(result['age'])} years")
                    with cols[2]:
                        st.metric("Emotion", result['dominant_emotion'])
                    with cols[3]:
                        st.metric("Race", result['dominant_race'])

elif selected == "Video Analysis":
    video_analysis()

elif selected == "Analysis History":
    st.title("Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history available yet. Try analyzing some images first!")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Analysis {i+1} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                st.image(entry['processed_image'], caption="Analyzed Image", use_container_width=True)
                for result in entry['results']:
                    cols = st.columns(4)
                    
                    # Display the gender with the highest probability
                    gender, probability = max(result['gender'].items(), key=lambda x: x[1])
                    with cols[0]:
                        st.write(f"Gender: {gender} ({probability:.2f}%)")
                    with cols[1]:
                        st.write(f"Age: {int(result['age'])}")
                    with cols[2]:
                        st.write(f"Emotion: {result['dominant_emotion']}")
                    with cols[3]:
                        st.write(f"Race: {result['dominant_race']}")

elif selected == "About":
    st.title("About Face Analysis Pro")
    st.write("""
    Face Analysis Pro is an advanced facial analysis application that uses deep learning to detect and analyze faces in images and video.
    
    ### Features:
    - Face Detection
    - Real-time Video Analysis
    - Webcam Support
    - Age Estimation
    - Gender Classification
    - Emotion Recognition
    - Ethnicity Detection
    - Analysis History
    - Downloadable Results
    
    ### Technologies Used:
    - Streamlit
    - DeepFace
    - OpenCV
    - Python
    
    ### Privacy Note:
    All image and video processing is done locally and no data is stored permanently.
    """)

# Cleanup on app exit
def cleanup():
    if st.session_state.video_capture is not None:
        st.session_state.video_capture.release()

# Register cleanup handler
import atexit
atexit.register(cleanup)
