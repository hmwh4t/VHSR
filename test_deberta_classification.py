"""
Test script for mDeBERTa-v2 hate speech classification model.
This script loads the fine-tuned model and performs text classification.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to the fine-tuned model
MODEL_PATH = "model/checkpoint-25000"

def load_model():
    """Load the tokenizer and model."""
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    return tokenizer, model, device


def classify_text(text: str, tokenizer, model, device) -> dict:
    """
    Classify a single text input.
    
    Args:
        text: Input text to classify
        tokenizer: The tokenizer
        model: The classification model
        device: The device (CPU/GPU)
    
    Returns:
        dict with label and confidence scores
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    
    # Get predicted label
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    # Map to label names (handle both int and str keys)
    id2label = model.config.id2label
    def get_label(idx):
        if idx in id2label:
            return id2label[idx]
        return id2label[str(idx)]
    
    predicted_label = get_label(predicted_class)
    
    return {
        "text": text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": {
            get_label(i): prob.item() 
            for i, prob in enumerate(probabilities[0])
        }
    }


def classify_batch(texts: list, tokenizer, model, device) -> list:
    """
    Classify a batch of texts.
    
    Args:
        texts: List of input texts
        tokenizer: The tokenizer
        model: The classification model
        device: The device (CPU/GPU)
    
    Returns:
        List of classification results
    """
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
    
    # Process results
    id2label = model.config.id2label
    def get_label(idx):
        if idx in id2label:
            return id2label[idx]
        return id2label[str(idx)]
    
    results = []
    
    for i, text in enumerate(texts):
        predicted_class = torch.argmax(probabilities[i]).item()
        confidence = probabilities[i][predicted_class].item()
        
        results.append({
            "text": text,
            "predicted_label": get_label(predicted_class),
            "confidence": confidence,
            "probabilities": {
                get_label(j): prob.item() 
                for j, prob in enumerate(probabilities[i])
            }
        })
    
    return results


def main():
    # Load model
    tokenizer, model, device = load_model()
    while True:
        try:
            user_input = input("\nEnter text to classify: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not user_input:
                continue
                
            result = classify_text(user_input, tokenizer, model, device)
            print(f"Label: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
            print(f"Probabilities: clean={result['probabilities']['clean']:.4f}, hate={result['probabilities']['hate']:.4f}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
