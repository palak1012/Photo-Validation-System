import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

REQUIRED_DIMENSIONS = (413, 531)

def validate_image(image_path):
    
    feedback = {} #to store results of validation

    #Dimension check
    image_pil = Image.open(image_path)
    if image_pil.size == REQUIRED_DIMENSIONS:
        feedback['Dimensions'] = {"status": "Pass", "confidence": 1.0}
    else:
        feedback['Dimensions'] = {
            "status": f"Fail (Dimensions got {image_pil.size})",
            "confidence": 0.0
        }


    #Load image and convert to RGB
    img_cv = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image_height, image_width = img_cv.shape[:2]

    #Face detection by mediapipe
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_results = face_detection.process(img_rgb)
    if not face_results.detections:
        feedback['Face Detection'] = {"status": "Fail (No face found)", "confidence": 0.0}
        return feedback
    else:
        detection_conf = face_results.detections[0].score[0]
        feedback['Face Detection'] = {"status": "Pass", "confidence": round(detection_conf, 2)}


    #Background check
    tolerance = 220
    white_mask = np.all(img_cv >= tolerance, axis=2)
    white_ratio = np.sum(white_mask) / (img_cv.shape[0] * img_cv.shape[1])
    status = "Pass (white background)" if white_ratio > 0.50 else f"Fail ({white_ratio*100:.1f}% white)"
    feedback['Background'] = {"status": status, "confidence": round(white_ratio, 2)}

    #Expression and accessories
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        mesh_results = face_mesh.process(img_rgb)
        if not mesh_results.multi_face_landmarks:
            feedback['Expression'] = {"status": "Fail (No landmarks)", "confidence": 0.0}
            feedback['Accessories'] = {"status": "Fail (Cannot detect)", "confidence": 0.0}
        else:
            landmarks = mesh_results.multi_face_landmarks[0].landmark

            #smile score
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            lip_dist = abs(upper_lip.y - lower_lip.y)

            # normalize, 0 = no smile, 1 = big smile
            smile_score = min(lip_dist * 15, 1.0) 
            expression_status = "Fail (Teeth visible)" if smile_score > 0.5 else "Pass (Teeth not visible)"
            expression_conf = 1 - smile_score
            feedback['Expression'] = {
                "status": expression_status,
                "confidence": round(expression_conf, 2)
            }

            #glasses score
            try:
                x1 = int(min(landmarks[33].x, landmarks[133].x) * image_width)
                x2 = int(max(landmarks[33].x, landmarks[133].x) * image_width)
                y1 = int(min(landmarks[159].y, landmarks[145].y) * image_height)
                y2 = int(max(landmarks[159].y, landmarks[145].y) * image_height)
                eye_region = img_cv[y1:y2, x1:x2]

                if eye_region.size > 0:
                    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                    brightness_std = np.std(gray_eye)
                     # 1 = glasses detected, 0 = no glasses
                    glasses_score = min(brightness_std / 100, 1.0)

                    glasses_status = "Fail (Glasses detected)" if glasses_score > 0.5 else "Pass (No glasses)"
                    feedback['Accessories'] = {
                        "status": glasses_status,
                        "confidence": round(1 - glasses_score, 2)
                    }
                else:
                    feedback['Accessories'] = {"status": "Fail (Eye region missing)", "confidence": 0.0}
            except:
                feedback['Accessories'] = {"status": "Fail", "confidence": 0.0}
                
    #to calculate overall confidence score
    confidences = []

    for key, value in feedback.items():
        if isinstance(value, dict) and 'confidence' in value:
            confidences.append(value['confidence'])

    if confidences:
        overall_confidence_score = round(sum(confidences) / len(confidences), 2)
        feedback['Overall Confidence'] = f"{overall_confidence_score} / 1.0"
    else:
        feedback['Overall Confidence'] = "N/A"


    return feedback