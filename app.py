import streamlit as st
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import efficientnet.tfkeras as efn

# Title and Description
st.title('Plant Disease Detection')
st.write('Just upload your Plant\'s Leaf!')

# In case of GPU issues:
# gpus = tf.config.experimental.list_physical_devices('GPU')

# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)

# Loading model
model = tf.keras.models.load_model('model.h5')

# Upload the image
uploaded_file = st.file_uploader('Choose your image', type=['png', 'jpg'])

predictions_map = {0:'is healthy', 1:'hast Multiple Diseases', 2:'has rust', 3:'has scab'}

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, use_column_width=True)
    # Image preprocessing
    resized_image = np.array(image.resize((512,512)))/255. # Resize image and divide pixel number by 255. for having values between 0 and 1 (normalize it)
    # Adding batch dimensions
    image_batch = resized_image[np.newaxis, :, :, :]
    # Getting the predictions from the model
    predictions_arr = model.predict(image_batch)
    
    predictions = np.argmax(predictions_arr)

    result_text = f'The plant leaf {predictions_map[predictions]} with {int(predictions_arr[0][predictions]*100)}% probability'

    if predictions == 0:
        st.success(result_text)
    else:
        st.error(result_text)