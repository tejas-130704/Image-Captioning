import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from tqdm import tqdm
import tensorflow as tf



model1 = tf.keras.models.load_model("models/model1.h5")

def feature_extraction(img):
    # img=image.load_img(img,target_size=(224,224,3))
    array_img=image.img_to_array(img)
    exp_img=np.expand_dims(array_img,axis=0)
    preprocessed_img=preprocess_input(exp_img)
    result=model1.predict(preprocessed_img).flatten()
    norm_result=result/np.linalg.norm(result)
    return norm_result

def predict_caption(model, img, tokenizer, max_length):

    extracted_feature = feature_extraction(img)
    
    # Ensure it's a NumPy array and reshape to (1, feature_dim)
    extracted_feature = np.array(extracted_feature).reshape(1, -1)

    # Start caption generation
    in_text = "startseq"

    for _ in range(max_length):
        # Convert words to sequences
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        
        # Pad sequence
        sequence = pad_sequences([sequence], maxlen=max_length, padding='pre')

        # Ensure it's a NumPy array and reshape properly
        sequence = np.array(sequence).reshape(1, max_length)

        # Predict next word
        yhat = model.predict([extracted_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)  # Get index of highest probability word
        
        # Convert index to word
        word = tokenizer.index_word.get(yhat, None)  # If index not found, return None

        
        if word is None:  # Handle unknown words
            break
        
        # Append word to generated sequence
        in_text += " " + word
        
        if word == "endseq":  # Stop if end token is reached
            break

    # Remove 'startseq' and 'endseq' for a clean caption
    final_caption = in_text.replace("startseq", "").replace("endseq", "").strip()
    
    return final_caption
