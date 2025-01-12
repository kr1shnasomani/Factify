<h1 align="center">Factify</h1>
The code demonstrates text classification using a Convolutional Neural Network (CNN) and LSTM model. It processes the fake or real news dataset by tokenizing titles, applying word embeddings and training a model to classify news as fake or real. The model uses GloVe word embeddings for improved performance.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install tensorflow numpy pandas scikit-learn matplotlib nltk keras glove-python-binary
   ```

2. Download the dataset (link to the dataset: **https://www.kaggle.com/datasets/hassanamin/textdb3**)

3. Upon running the code, it also saves an addition file named model.keras (this file stores the trained model)

4. Enter the news text in the code and it will provide the prediction

## Accuracy & Loss Over Epochs:

![image](https://github.com/user-attachments/assets/abf80397-acc7-44b6-ad81-bae6671560b0)

![image](https://github.com/user-attachments/assets/adc04adf-5653-4937-84db-561862d6b499)

## Overview:
The code trains a text classification model using a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) network with pre-trained word embeddings. Based on the code, here are some points:

### Key Steps:
1. **Dataset Download and Extraction:** You successfully downloaded and extracted the dataset from Kaggle.
   
2. **Data Preprocessing:**
   - You loaded the data from the CSV file and preprocessed the text and labels.
   - The labels are encoded using `LabelEncoder` for binary classification (`REAL` or `FAKE`).
   
3. **Tokenization:** The titles of the articles are tokenized using the Keras `Tokenizer` and padded to the same length.

4. **Word Embeddings:** You used GloVe (Global Vectors for Word Representation) embeddings (`glove.6B.50d.txt`), which are loaded and indexed for the training model.

5. **Model Architecture:** The model consists of an `Embedding` layer initialized with the GloVe embeddings, followed by:
     - `Dropout` for regularization.
     - `Conv1D` and `MaxPooling1D` for feature extraction.
     - `LSTM` for learning sequence dependencies.
     - `Dense` layer for the binary classification task (`sigmoid` activation).

6. **Model Training:** The model is trained for 50 epochs, with accuracy and loss being monitored during training.

7. **Model Performance:** The trained the model for 20 epochs and achieved the following:
- Training accuracy: 94.81% 
- Validation accuracy: ~75-78%
