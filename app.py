import io
import os
import numpy as np
import pickle
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.applications import ResNet50
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import streamlit as st

detector = MTCNN()
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')
total_params = model.count_params()
print("Total number of features learned:", total_params)
total_layers = len(model.layers)
print("Total number of layers:", total_layers)

print(model.summary())
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path, model, detector):
    img = Image.open(img_path)
    img = np.array(img.convert('RGB'))

    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # save the image in a directory
    with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
        f.write(uploaded_image.getbuffer())
        saved_image_path = os.path.join('uploads', uploaded_image.name)
        # Save the uploaded image to the specified directory
        with open(saved_image_path, 'wb') as f:
            f.write(uploaded_image.read())
        # Load the image
        display_image = Image.open(saved_image_path)

        # Display the image


        # extract the features
        features = extract_features(saved_image_path, model, detector)

        # recommend
        index_pos = recommend(feature_list, features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

        # display
        col1, col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)

        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos], width=300)


else:
    st.write("No image file uploaded.")
