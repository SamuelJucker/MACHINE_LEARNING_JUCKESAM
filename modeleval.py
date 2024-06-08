import json
import pandas as pd
from datasets import load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

# Load the JSON files
train_json_file_path = 'path/to/your/train.jsonl'
val_json_file_path = 'path/to/your/val.jsonl'
test_json_file_path = 'path/to/your/test.jsonl'

# Read the JSONL files
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

train_data = read_jsonl(train_json_file_path)
val_data = read_jsonl(val_json_file_path)
test_data = read_jsonl(test_json_file_path)

# Convert to DataFrames for better handling
train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
test_df = pd.DataFrame(test_data)

# Combine the datasets for preprocessing and fine-tuning
combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Retrieve a random sample
random_index = random.randint(0, len(combined_df) - 1)
initial_sample = combined_df.iloc[random_index]
print("Random sample from the initial dataset (before processing):")
print(initial_sample)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Load the tokenizer
model_name = "google/pegasus-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define a preprocessing function
def preprocess_function(examples):
    inputs = [doc for doc in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the preprocessing function to the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Retrieve the processed sample
processed_sample = train_dataset[random_index]
print("\nRandom sample from the processed dataset (after processing):")
print(processed_sample)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Generate summaries for the validation set
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, min_length=30, num_beams=5, length_penalty=2.0, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Evaluate the model
rouge = load_metric("rouge")

def evaluate_model(dataset):
    summaries = []
    references = []
    for example in dataset:
        summaries.append(generate_summary(example['input']))
        references.append(example['summary'])
    results = rouge.compute(predictions=summaries, references=references)
    return results

# Calculate ROUGE scores for the validation set
validation_results = evaluate_model(val_dataset)
print(validation_results)

# Prepare data for Doc2Vec
documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(combined_df['input'])]

# Train Doc2Vec model
doc2vec_model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, workers=4)

# Generate embeddings
def get_embedding(text):
    return doc2vec_model.infer_vector(text.split())

# Calculate cosine similarity
def calculate_cosine_similarity(original, summary):
    original_embedding = get_embedding(original)
    summary_embedding = get_embedding(summary)
    return cosine_similarity([original_embedding], [summary_embedding])[0][0]

# Example usage
example_text = combined_df.iloc[0]['input']
example_summary = generate_summary(example_text)
cosine_sim = calculate_cosine_similarity(example_text, example_summary)
print(f"Cosine Similarity: {cosine_sim}")
