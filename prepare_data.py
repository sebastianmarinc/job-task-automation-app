import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle # For saving/loading DataFrame with embeddings
import os # For path manipulation

# --- Configuration ---
DATA_FILE_PATH = 'job_tasks_data.csv'
# Paths for saving pre-computed data
PRECOMPUTED_DF_PATH = 'data/precomputed_df_with_embeddings.pkl'
PRECOMPUTED_FAISS_INDEX_PATH = 'data/precomputed_faiss_index.bin'
INITIAL_SEARCH_MODEL_NAME = 'BAAI/bge-small-en-v1.5'

def prepare_data():
    print(f"--- Data Preparation Started ({pd.Timestamp.now()}) ---")

    # Create 'data' directory if it doesn't exist
    os.makedirs(os.path.dirname(PRECOMPUTED_DF_PATH), exist_ok=True)

    print(f"Loading initial search model: {INITIAL_SEARCH_MODEL_NAME}...")
    model = SentenceTransformer(INITIAL_SEARCH_MODEL_NAME)
    print("Model loaded.")

    print(f"Loading raw data from {DATA_FILE_PATH}...")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
        df.columns = df.columns.str.strip()

        metric_cols = ['feedback_loop', 'directive', 'task_iteration', 'validation', 'learning', 'filtered']
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        print("Raw data loaded and cleaned.")
    except FileNotFoundError:
        print(f"Error: Raw data file '{DATA_FILE_PATH}' not found. Please ensure it's in the same directory as 'prepare_data.py'.")
        return
    except Exception as e:
        print(f"An error occurred while loading or processing raw data: {e}")
        return

    print("Generating embeddings for task descriptions... This may take a while.")
    # Add instruction to query before encoding for BGE models
    df['encoded_task_name'] = df['task_name'].apply(lambda x: "Represent this sentence for searching relevant passages: " + x if pd.notna(x) else None)
    df['embedding'] = df['encoded_task_name'].apply(lambda x: model.encode(x) if pd.notna(x) else None)
    df = df.dropna(subset=['embedding'])
    print("Embeddings generated.")

    print("Building FAISS index...")
    embeddings_matrix = np.vstack(df['embedding'].values).astype('float32')
    dimension = embeddings_matrix.shape[1]

    faiss.normalize_L2(embeddings_matrix) # Normalize for IP index (cosine similarity)
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings_matrix)
    print(f"FAISS index built with {faiss_index.ntotal} vectors.")

    print(f"Saving pre-computed DataFrame to {PRECOMPUTED_DF_PATH}...")
    # Drop the temporary 'encoded_task_name' column
    df = df.drop(columns=['encoded_task_name'], errors='ignore')
    with open(PRECOMPUTED_DF_PATH, 'wb') as f:
        pickle.dump(df, f)
    print("DataFrame saved.")

    print(f"Saving FAISS index to {PRECOMPUTED_FAISS_INDEX_PATH}...")
    faiss.write_index(faiss_index, PRECOMPUTED_FAISS_INDEX_PATH)
    print("FAISS index saved.")

    print(f"--- Data Preparation Complete ({pd.Timestamp.now()}) ---")

if __name__ == '__main__':
    prepare_data()
