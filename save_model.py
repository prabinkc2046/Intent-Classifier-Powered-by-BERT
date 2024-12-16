def save_model(model, tokenizer, model_name="intent_classifier_model"):
    model.save_pretrained(model_name)
    tokenizer.save_pretrained(model_name)