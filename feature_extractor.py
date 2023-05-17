# !pip install mtcnn==0.1.0
# !pip install tensorflow==2.3.1
# !pip install keras==2.4.3
# !pip install keras-vggface==0.6
# !pip install keras_applications==1.0.8

# import os
# import pickle
#
# actors = os.listdir('data')
#
# filenames = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('data',actor)):
#         filenames.append(os.path.join('data',actor,file))
#
# pickle.dump(filenames,open('filenames.pkl','wb'))

#
from keras.applications.vgg16 import preprocess_input
from keras.applications import ResNet50
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')

def feature_extractor(img_path, model):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

pickle.dump(features, open('embedding.pkl', 'wb'))
