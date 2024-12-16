import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Fetch host and port from environment variables
host ="0.0.0.0"
port = 8000  # Default to 8000 if not specified

print("env var", host, port)
# Define a request body schema
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow connections from any origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define intents
intents = [
    'contact', 'education', 'experiences',
    'general', 'intent', 'introduction',
    'projects', 'services', 'skills', 'userWantsToSendEmail'
]

# Load your fine-tuned model and tokenizer
model_path = "./intent_classifier_model"  # Path to your saved model
classifier = pipeline("text-classification", model=model_path)

@app.get("/")
def home():
    """
    Root endpoint to verify API is working.
    """
    return {"message": "Welcome to the Intent Classification API!"}

@app.post("/classify")
def classify_text(input: TextInput):
    """
    Endpoint to classify the intent of the provided text.
    """
    text = input.text

    # Validate input text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Perform classification
    result = classifier(text)
    print(result)  # Debugging: log the model output

    # Extract label and map it back to the real intent
    raw_label = result[0]["label"]  # e.g., "LABEL_6"
    label_index = int(raw_label.split("_")[-1])  # Extract the numeric part, e.g., 6
    if label_index >= len(intents):
        raise HTTPException(status_code=500, detail="Invalid label index returned by the model.")

    return {
        "intent": intents[label_index],
    }

# Entry point to run the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, reload=True)
