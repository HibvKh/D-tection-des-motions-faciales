import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Charger le modèle Keras
classifier = load_model("C:\\Users\\jamal\\Downloads\\model.h5")

# Charger le classificateur en cascade pour la détection des visages
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Définir les étiquettes des émotions
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Définir le titre et le sous-titre de l'interface
st.title('Emotion Detector')
st.subheader('Analysez vos émotions en temps réel')

# Section 'À propos de nous'
st.sidebar.title('À propos de nous')
st.sidebar.write("Bienvenue dans notre application d'analyse d'émotions en temps réel ! \
                  Nous utilisons la technologie de l'apprentissage profond pour détecter les émotions à partir de votre webcam.")

# Section 'Image d'exemple'
st.subheader('Image d\'exemple')
image_path = r"C:\Users\jamal\Downloads\OIP.jpeg"  # Chemin de votre image d'exemple

st.image(image_path, caption='Exemple d\'image')

# Boutons pour importer une photo et prendre une photo en direct
option = st.radio('Choisissez une option :', ('Prendre une photo en direct', 'Importer une photo'))

if option == 'Importer une photo':
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Afficher l'image dans Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

elif option == 'Prendre une photo en direct':
    # Début de la capture vidéo
    cap = cv2.VideoCapture(0)
    
    # Bouton pour prendre une photo
    if st.button('Prendre une photo'):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Afficher l'image dans Streamlit
        st.image(frame, channels="BGR", use_column_width=True)

    # Libérer la capture vidéo lorsque l'utilisateur ferme l'application
    cap.release()
