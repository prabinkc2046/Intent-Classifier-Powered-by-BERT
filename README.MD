# Intentify: A FastAPI-Based Intent Classifier Powered by BERT

## Overview

Intentify is a FastAPI-powered application designed to classify the intent of a given text input. Leveraging a fine-tuned **BERT-based** machine learning model, it can identify specific intents, such as contact, education, skills, and more. This project demonstrates the power of FastAPI for creating efficient and scalable REST APIs combined with the capabilities of state-of-the-art natural language processing using BERT.

## Features

- Classify text into predefined intents with high accuracy using a fine-tuned BERT model.
- Fast and scalable API built using FastAPI.
- Easy-to-deploy using Render or other cloud platforms.
- CORS-enabled to accept connections from any origin.

## Predefined Intents

The app currently supports the following intents:

- `contact`
- `education`
- `experiences`
- `general`
- `intent`
- `introduction`
- `projects`
- `services`
- `skills`
- `userWantsToSendEmail`

## Requirements

To run the application, you need the following:

- Python 3.8 or higher
- A fine-tuned **BERT-based** text classification model saved in a directory (default: `intent_classifier_model`).
- Required Python libraries (listed in `requirements.txt`).

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/prabinkc2046/Intent-Classifier-Powered-by-BERT
   cd intentify
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the Model is Available:**
   Place your fine-tuned **BERT-based** text classification model in the `intent_classifier_model` directory (or update the `model_path` in `app.py`).

## Running the Application

To run the application locally:

```bash
python3 app.py
```

The app will start on `http://0.0.0.0:8000`. You can access the endpoints using a REST client (e.g., Postman) or your browser.

Alternatively, use `uvicorn` directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### 1. **Home Endpoint**

- **URL:** `/`
- **Method:** `GET`
- **Description:** Returns a welcome message to verify the API is running.
- **Response:**
  ```json
  {
    "message": "Welcome to the Intent Classification API!"
  }
  ```

### 2. **Classify Text Endpoint**

- **URL:** `/classify`
- **Method:** `POST`
- **Description:** Classifies the intent of the provided text input.
- **Request Body:**
  ```json
  {
    "text": "Sample text to classify"
  }
  ```
- **Response:**
  ```json
  {
    "intent": "education"
  }
  ```
- **Error Handling:**
  - Returns a `400` error if the input text is empty.
  - Returns a `500` error if the model returns an invalid label index.

## Deployment

### Deploying on Render

To deploy the app on Render:

1. Ensure you have a `requirements.txt` file and the code is pushed to a Git repository.

2. Follow these steps in Render:

   - Create a new **Web Service** in the Render dashboard.
   - Link your repository.
   - Set the build command:
     ```bash
     pip install -r requirements.txt
     ```
   - Set the start command:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```

3. Deploy the app, and Render will automatically set it up.

### Running in Production

For production deployment, use:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

This will run the app with multiple workers for better performance.

## Folder Structure

```
.
├── app.py
├── data_set.py
├── encode_labels.py
├── evaluate_model.py
├── full-intent-data.csv
├── infer_intent.py
├── intent_classifier_model
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── intent_classifier_model-20241213T061801Z-001.zip
├── intent-data.csv
├── load_data.py
├── load_pretrained_model.py
├── load_tokenizer.py
├── main.py
├── preprocessing
│   ├── cleanedData.csv
│   └── validate.py
├── README.MD
├── requirements.txt
├── save_model.py
├── tokenize_data.py
└── train_model.py
```

## License

This project is licensed under the MIT License.
