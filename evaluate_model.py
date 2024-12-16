import torch
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    for batch in test_loader:
        with torch.no_grad():
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

    print(f"Accuracy: {correct / total}")