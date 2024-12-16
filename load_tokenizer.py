from transformers import BertTokenizer

def load_tokenizer(pretrained_model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(pretrained_model_name)

load_tokenizer()
