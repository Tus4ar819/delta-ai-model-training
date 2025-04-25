from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from flask import Flask, request, jsonify
from transformers import pipeline
import json

# Initialize Flask app
app = Flask(__name__)

# Define training function
def train_model(training_file, num_labels, epochs, batch_size):
    # Load data from JSONL file
    with open(training_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Prepare dataset
    queries = [item['prompt'] for item in data]
    targets = [item['target '] for item in data]
    dataset = Dataset.from_dict({"text": queries, "labels": targets})

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(preprocess, batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    # Define training arguments
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_dir="logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset
    )

    # Train model
    trainer.train()

    # Return metrics
    return trainer.state.log_history

# Define /train endpoint
@app.route("/train", methods=["POST"])
def train():
    data = request.json
    training_file = data.get("training_file", "training_600.jsonl")
    num_labels = data.get("num_labels", 2)
    epochs = data.get("epochs", 3)
    batch_size = data.get("batch_size", 8)

    # Train the model
    metrics = train_model(training_file, num_labels, epochs, batch_size)
    return jsonify({"metrics": metrics})

# Define /predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    result = nlp(text)
    return jsonify(result)

# Initialize prediction pipeline
nlp = pipeline("text-classification", model="distilbert-base-uncased", tokenizer="distilbert-base-uncased", device=0)

# Inference pipeline
ner_pipeline = pipeline("ner", model="my_trained_ner_model", tokenizer="my_trained_ner_model", aggregation_strategy="simple")

# Example queries covering all entity types
example_queries = [
    "Alice Smith adopted a cat named Leo in Paris.",  # PERSON, ANIMAL, CITY
    "Tesla launched its new Taj Mahal in New York.",  # ORG, THING, CITY
    "Emma Davis, originally from Australia, now lives in Paris.",  # PERSON, COUNTRY, CITY
    "A wild lion was spotted near the Great Wall of China in Canada.",  # ANIMAL, THING, COUNTRY
    "During the Comic-Con, John Doe gave a speech at the Great Wall of China.",  # EVENT, PERSON, THING
    "The Amazon headquarters are located in Paris, Italy.",  # ORG, CITY, COUNTRY
    "In Japan, people celebrate Olympic Games with great enthusiasm.",  # COUNTRY, EVENT
    "The Statue of Liberty has become a symbol of New York's history.",  # THING, CITY
    "Researchers at IBM discovered a new species of dolphin in Italy.",  # ORG, ANIMAL, COUNTRY
    "Every year, the Cannes Festival is held in Sydney, attracting visitors worldwide."  # EVENT, CITY
]

for query in example_queries:
    result = ner_pipeline(query)
    print(f"Query: {query}")
    if result:
        for ent in result:
            print(f"  Entity: '{ent['word']}' | Type: {ent['entity_group']} | Score: {ent['score']:.2f}")
    else:
        print("  No entities found.")
    print("-")

# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)

