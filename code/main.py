# Import the required libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

def predict_fake(title, text):
    # Format the input string
    input_str = "<title>" + title + "<content>" + text + "<end>"
    
    # Tokenize the input
    input_ids = tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    
    # Ensure the model uses CPU
    device = 'cpu'
    model.to(device)
    
    # Perform prediction without tracking gradients
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.Softmax(dim=1)(output.logits)[0]
    
    # Get the prediction and the corresponding confidence
    prediction = "Real" if probabilities[1] > probabilities[0] else "Fake"
    confidence = probabilities[1].item() if prediction == "Real" else probabilities[0].item()
    
    # Return the prediction and confidence
    return {"Prediction": prediction, "Confidence": round(confidence * 100, 2)}

# Read the news article from the file
news_file_path = r"C:\Users\krish\OneDrive\Desktop\Projects\Fake News Detection\fake.txt"

with open(news_file_path, "r", encoding="utf-8") as file:
    news_content = file.read()

# Assuming the first line is the title and the rest is the content
title = news_content.splitlines()[0]
content = "\n".join(news_content.splitlines()[1:])

# Predict and print the result
result = predict_fake(title, content)
print(f"Prediction: {result['Prediction']}\nConfidence: {result['Confidence']}%")