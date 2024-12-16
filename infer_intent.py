def infer_intent(classifier, text, label_encoder):
    result = classifier(text)
    predicted_label = result[0]["label"]
    predicted_index = int(predicted_label.split("_")[1])
    real_label = label_encoder.inverse_transform([predicted_index])[0]
    print(f"Predicted Intent: {real_label}, Confidence: {result[0]['score']}")