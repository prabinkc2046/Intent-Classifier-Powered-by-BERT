from load_data import load_data
from encode_labels import encode_label
from load_tokenizer import load_tokenizer
from load_pretrained_model import load_pretrained_model
from data_set import IntentDataset
from train_model import train_model
from evaluate_model import evaluate_model
from save_model import save_model
from torch.utils.data import DataLoader
import torch

file_name = "intent-data"
file_path=f"./{file_name}.csv"

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data(file_path)
    data, label_encoder = encode_label(data, column_label="intent")

    # Tokenizer
    tokenizer = load_tokenizer()

    # Split data
    from sklearn.model_selection import train_test_split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"], data["intent"], test_size=0.2, random_state=42
    )

    # Load model
    model = load_pretrained_model(num_labels=len(label_encoder.classes_))

    # Create datasets and dataloaders
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    test_dataset = IntentDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Train
    train_model(model, train_loader, optimizer, num_epochs=3)

    # Evaluate
    evaluate_model(model, test_loader)

    save_model(tokenizer, model_name="intent_classifier_model")
