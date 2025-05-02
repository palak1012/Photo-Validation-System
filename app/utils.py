import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

REQUIRED_DIMENSIONS = (413, 531)

def validate_image(image_path):
    
    feedback = {} #to store results of validation

    #Dimension check
    image_pil = Image.open(image_path)
    feedback['Dimensions'] = "Pass" if image_pil.size == REQUIRED_DIMENSIONS else f"Fail (Got {image_pil.size})"

    #Load image and convert to RGB
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    #Face detection by mediapipe
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_results = face_detection.process(img_rgb)
    if not face_results.detections:
        feedback['Face Detection'] = "Fail (No face found)"
        return feedback
    feedback['Face Detection'] = "Pass (Face detected)"

    #Background check
    tolerance = 220
    white_mask = np.all(img_cv >= tolerance, axis=2)
    white_ratio = np.sum(white_mask) / (img_cv.shape[0] * img_cv.shape[1])
    feedback['Background'] = "Pass (White background)" if white_ratio > 0.50 else f"Fail ({white_ratio * 100:.1f}% white)"

    #Expression and accessories
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        mesh_results = face_mesh.process(img_rgb)
        if not mesh_results.multi_face_landmarks:
            feedback['Expression'] = "Fail (No landmarks)"
            feedback['Accessories'] = "Fail (Cannot detect features)"
        else:
            landmarks = mesh_results.multi_face_landmarks[0].landmark

            #to check if teeth visible
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            lip_dist = np.sqrt((upper_lip.x - lower_lip.x)**2 + (upper_lip.y - lower_lip.y)**2)
            feedback['Expression'] = "Fail (Teeth visible)" if lip_dist > 0.02 else "Pass (Teeth not visible)"

            #Glasses detection
            try:
                x1 = int(min(landmarks[33].x, landmarks[133].x) * img_cv.shape[1])
                x2 = int(max(landmarks[33].x, landmarks[133].x) * img_cv.shape[1])
                y1 = int(min(landmarks[159].y, landmarks[145].y) * img_cv.shape[0])
                y2 = int(max(landmarks[159].y, landmarks[145].y) * img_cv.shape[0])
                eye_region = img_cv[y1:y2, x1:x2]
                if eye_region.size > 0:
                    brightness = np.mean(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY))
                    feedback['Accessories'] = "Fail (Glasses detected)" if brightness < 100 else "Pass (No glasses)"
                else:
                    feedback['Accessories'] = "Fail (Eye region missing)"
            except:
                feedback['Accessories'] = "Fail (Eye region error)"
    
    return feedback