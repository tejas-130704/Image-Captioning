# Image Captioning using ResNet50 and LSTM

This project implements an Image Captioning model using **ResNet50** for feature extraction and **LSTM** for sequence generation. The model is trained on 8K images, which results in decent but not highly accurate captions due to the limited dataset size.

## Preview

![Screenshot 2025-02-23 210222](https://github.com/user-attachments/assets/5b809ef5-02c4-4d7a-a765-2b6c8dc7ccea)


![Screenshot 2025-02-23 210004](https://github.com/user-attachments/assets/f2e710ed-3025-47a2-a698-e78cc2781e9f)


## Features
- Uses **ResNet50** as the feature extractor
- LSTM-based caption generation
- Trained on **8K images** with a limited vocabulary
- **Streamlit application** provided for easy testing

## Model Download & Usage
The trained model is **too large to be pushed to GitHub**. Instead, you can train your own model using the provided Jupyter Notebook. Alternatively, you can use the Streamlit application to test the model by adding the only `model.h5` and `tokenizer.json` file.

### Steps to Use the Streamlit App
1. Clone this repository:
   ```bash
   git clone https://github.com/tejas-130704/Image-Captioning.git
   cd Image-Captioning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained **`model.h5`** file in models folder and **`tokenizer.json`** file in the project directory.
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Upload an image and get captions!

## Limitations & Future Improvements
- **Limited dataset size (8K images)** results in **less accurate captions**.
- Accuracy can be significantly improved by training on **larger datasets** such as **COCO** or **Flickr30k**.
- Future work can involve adding an **Attention Mechanism** for better caption quality.

## Dataset
The model is trained using [Flickr 8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

## Contributions
Feel free to fork this repository and contribute improvements. Pull requests are welcome!



