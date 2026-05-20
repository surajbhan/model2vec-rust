import json
import numpy as np
from model2vec import StaticModel

def main():
    print("Loading model...")
    # Load model from local folder
    model = StaticModel.from_pretrained("model")
    
    sentences = [
        "Hello world",
        "Rust is awesome",
        "This is an extremely performant and simple model2vec implementation.",
        "Model2Vec distilled models are up to 500x faster than sentence transformers on CPU.",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "", # Test empty sentence
        "Very long sentence: " + " ".join(["word"] * 100)
    ]
    
    print("Encoding sentences...")
    embeddings = model.encode(sentences)
    
    # Let's save the sentences and embeddings to json
    data = []
    for s, emb in zip(sentences, embeddings):
        # emb is a numpy array, convert to list of floats
        emb_list = emb.tolist()
        data.append({
            "sentence": s,
            "embedding": emb_list
        })
        
    output_path = "ground_truth.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved ground truth to {output_path}")

if __name__ == "__main__":
    main()
