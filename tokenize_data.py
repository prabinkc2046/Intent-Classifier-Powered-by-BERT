def tokenize_data(tokenizer, sentences, max_length=64):
    return tokenizer(
        list(sentences),
        truncation = True,
        max_length= max_length,
        padding="max_length",
        return_tensors="pt"

    )


