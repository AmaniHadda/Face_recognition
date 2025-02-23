import threading
import cv2
import os
import json
from deepface import DeepFace

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
counter = 0
face_match = False
matched_name = "Unknown"
thread_running = False
dataset_folder = "DataSet"
matching_json_path = os.path.join(dataset_folder, "matching.json")

if not os.path.exists(matching_json_path):
    print(f"Erreur : Le fichier {matching_json_path} n'existe pas.")
    exit()

with open(matching_json_path, "r") as f:
    matching_data = json.load(f)

reference_images = []

for filename in os.listdir(dataset_folder):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(dataset_folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            reference_images.append((img, matching_data.get(filename, "Unknown")))

if not reference_images:
    print("Erreur : Aucun visage trouvé dans le dossier 'DataSet'.")
    exit()

reference_images_rgb = [(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), name) for img, name in reference_images]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def check_face(frame):
    global face_match, matched_name, thread_running
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for reference_img_rgb, name in reference_images_rgb:
            result = DeepFace.verify(frame_rgb, reference_img_rgb, model_name='Facenet512', enforce_detection=False, threshold=0.5, detector_backend='opencv')
            print(result)
            if result.get('verified', False):
                face_match = True
                matched_name = name
                print(f"MATCH trouvé : {name}")
                break
        else:
            face_match = False
            matched_name = "Unknown"
    except Exception as e:
        print("Erreur DeepFace :", str(e))
        face_match = False
        matched_name = "Unknown"
    finally:
        thread_running = False

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 10 == 0 and not thread_running:
            thread_running = True
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
        counter += 1

        message = f"MATCH: {matched_name}" if face_match else "UNKNOWN"
        color = (0, 255, 0) if face_match else (0, 0, 255)
        cv2.putText(frame, message, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        cv2.imshow("video", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
