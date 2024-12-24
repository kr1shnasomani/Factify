<h1 align="center">Factify</h1>
This project uses a fine-tuned RoBERTa model for fake news detection. It analyzes the title and content of news articles, predicting "Real" or "Fake" with a confidence score. Built with Hugging Face Transformers and PyTorch for NLP tasks.

## Execution Guide:
1. Run the following command line in the terminal:
   ```
   pip install transformers torch
   ```

2. Enter the news article in a `.txt` file and paste the path of this file into the code

3. Upon running the code now, it will output the prediction with the confidence score

## Model Prediction:

1. `fake.txt`:

   ![image](https://github.com/user-attachments/assets/f0273857-b498-4168-bb30-e151982b8c9c)

2. `real.txt`:

   ![image](https://github.com/user-attachments/assets/797f1d11-7a2d-448d-ba35-82a7b4d80a94)

## Overview:
This project implements a **fake news detection system** using a fine-tuned **RoBERTa** model for sequence classification. It processes news articles, evaluates their content, and predicts whether they are "Real" or "Fake" with a confidence score.

1. **Model and Tokenizer Initialization**: The pretrained model and tokenizer are loaded from the Hugging Face model repository (`hamzab/roberta-fake-news-classification`).

2. **Prediction Function**:
   - Combines the news title and content into a single formatted string.
   - Tokenizes the input string and feeds it to the model.
   - Applies softmax to compute probabilities and determines the prediction and confidence score.

3. **Input Processing**: Reads a news article from a text file where the first line is assumed to be the title, and the subsequent lines form the content.

4. **Output**: Prints the prediction (`Real` or `Fake`) and the confidence percentage.
