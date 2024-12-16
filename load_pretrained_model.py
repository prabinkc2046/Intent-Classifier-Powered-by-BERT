from transformers import BertForSequenceClassification

def load_pretrained_model(pretrained_model_name="bert-base-uncase", num_labels=10):
    return BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels)
