import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss
from scipy.special import expit # Import the sigmoid function
import os 

# --- Configuration ---
# Paths for loading pre-computed data
PRECOMPUTED_DF_PATH = 'data/precomputed_df_with_embeddings.parquet' 
PRECOMPUTED_FAISS_INDEX_PATH = 'data/precomputed_faiss_index.bin'
RERANK_CANDIDATES = 10 # Number of top candidates to send to the Re-ranker

# --- Load Models ---
@st.cache_resource # Cache the initial search model
def load_initial_search_model():
    """Loads the model for initial quick search (finds many possible matches)."""
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

@st.cache_resource # Cache the re-ranker model
def load_reranker_model():
    """Loads the specialized re-ranker model (refines the initial matches)."""
    return CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1')

# --- Load Pre-computed Data and FAISS Index ---
@st.cache_data # Cache the loaded data and index
def load_precomputed_data_and_faiss_index(df_path, faiss_index_path):
    """
    Loads pre-computed DataFrame with embeddings and a pre-built FAISS index.
    """
    try:
        if not os.path.exists(df_path) or not os.path.exists(faiss_index_path):
            st.error(f"Pre-computed data files not found. Please ensure the 'data/' folder with '{os.path.basename(df_path)}' and '{os.path.basename(faiss_index_path)}' exists in your repository.")
            st.warning("If these files are missing, you need to run 'prepare_data.py' locally first to generate them and commit them to your repository.")
            st.stop()

        df = pd.read_parquet(df_path) # Load from Parquet
        # Parquet typically handles NumPy arrays in columns correctly, so explicit conversion might not be needed
        # df['embedding'] = df['embedding'].apply(lambda x: np.array(x) if isinstance(x, list) else x) # Keep if needed for older pyarrow

        faiss_index = faiss.read_index(faiss_index_path)
        return df, faiss_index

    except Exception as e:
        st.error(f"An error occurred while loading pre-computed data: {e}. Please ensure 'prepare_data.py' was run successfully and the generated files are correct.")
        st.stop()

# --- Task Matching and Analysis Function ---
def analyze_single_task(query_task_text, df, initial_search_model, reranker_model, faiss_index, initial_similarity_threshold, rerank_candidates=RERANK_CANDIDATES):
    """
    Analyzes a single job task to find the closest match and its characteristics.
    Uses a two-stage process: fast initial search with FAISS, then refined re-ranking.
    """
    # Add the recommended instruction to the query before encoding for BGE models.
    query_with_instruction = "Represent this sentence for searching relevant passages: " + query_task_text
    query_embedding = initial_search_model.encode(query_with_instruction).astype('float32').reshape(1, -1)

    # Normalize query embedding for IP index (cosine similarity)
    faiss.normalize_L2(query_embedding)

    k_candidates_for_faiss = max(RERANK_CANDIDATES, 20)
    # Search FAISS index. distances will be IP scores, faiss_indices are the row indices in the original matrix.
    initial_scores, faiss_indices = faiss_index.search(query_embedding, k_candidates_for_faiss)
    initial_scores = initial_scores[0] # Get scores for the first (and only) query

    valid_candidate_mask = initial_scores >= initial_similarity_threshold

    filtered_faiss_indices = faiss_indices[0][valid_candidate_mask]

    if len(filtered_faiss_indices) == 0:
        return {
            'original_task_input': query_task_text,
            'matched_task_name': 'No close match found',
            'initial_similarity': 0.0,
            'final_similarity': 0.0,
            'feedback_loop': 0.0, 'directive': 0.0, 'task_iteration': 0.0,
            'validation': 0.0, 'learning': 0.0, 'filtered': 0.0,
            'classification_type': 'Unmatched Task'
        }

    filtered_initial_scores = initial_scores[valid_candidate_mask]
    sorted_candidate_indices_in_filtered = np.argsort(filtered_initial_scores)[::-1][:rerank_candidates]
    sorted_df_indices_for_reranking = filtered_faiss_indices[sorted_candidate_indices_in_filtered]

    candidate_texts = df.iloc[sorted_df_indices_for_reranking]['task_name'].tolist()
    sentence_pairs = [[query_task_text, candidate_text] for candidate_text in candidate_texts]

    raw_reranked_scores = reranker_model.predict(sentence_pairs)
    normalized_reranked_scores = expit(raw_reranked_scores)

    best_rerank_idx_in_candidates = np.argmax(normalized_reranked_scores)
    final_best_match_df_idx = sorted_df_indices_for_reranking[best_rerank_idx_in_candidates]
    matched_task_row = df.iloc[final_best_match_df_idx]

    return {
        'original_task_input': query_task_text,
        'matched_task_name': matched_task_row['task_name'],
        'initial_similarity': filtered_initial_scores[sorted_candidate_indices_in_filtered[best_rerank_idx_in_candidates]],
        'final_similarity': normalized_reranked_scores[best_rerank_idx_in_candidates],
        'feedback_loop': matched_task_row['feedback_loop'],
        'directive': matched_task_row['directive'],
        'task_iteration': matched_task_row['task_iteration'],
        'validation': matched_task_row['validation'],
        'learning': matched_task_row['learning'],
        'filtered': matched_task_row['filtered'],
        'classification_type': 'Matched Task'
    }

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("Job Task Automation & Augmentation Estimator")

st.markdown("""
This app helps estimate how much of a job can be **automated** (done by AI) or **augmented** (AI helping humans).

Just enter a list of job tasks, and the app will try to find similar tasks from our database and calculate their potential for automation and augmentation.
""")

# Load models
initial_search_model = load_initial_search_model()
reranker_model = load_reranker_model()

# Load pre-computed data and FAISS index
with st.spinner("Setting up the brain of the app..."):
    df_tasks, faiss_index = load_precomputed_data_and_faiss_index(PRECOMPUTED_DF_PATH, PRECOMPUTED_FAISS_INDEX_PATH)

# User input for similarity threshold
st.sidebar.header("Controls")
selected_similarity_threshold = st.sidebar.slider(
    "Minimum Match Strength",
    min_value=0.0,
    max_value=1.0,
    value=0.65, # Default value
    step=0.01,
    help="Controls filtering. Tasks with a raw score below this value will be marked 'Unmatched'."
)

st.sidebar.info("""
**About the Dataset**

The data in this app is based on research from the Anthropic Economic Index, which analyzed 1 million anonymized conversations from Claude to understand real-world AI usage. The analysis involves breaking down how users interact with the AI, distinguishing between "augmentable" tasks (where humans collaborate with the model) and "automatable" tasks where the model completes a task with minimal human involvement).

""")

default_job_tasks = """...
"""

job_tasks_input = st.text_area("Job Tasks (one per line):", height=200,
                                value=default_job_tasks)

if st.button("Analyze My Job Tasks"):
    if job_tasks_input:
        input_tasks = [task.strip() for task in job_tasks_input.split('\n') if task.strip()]

        if not input_tasks:
            st.warning("Please enter some job tasks to analyze.")
            st.stop()

        all_analysis_results = []
        with st.spinner(f"Analyzing {len(input_tasks)} tasks..."):
            for task in input_tasks:
                result = analyze_single_task(
                    task, df_tasks, initial_search_model, reranker_model, faiss_index,
                    selected_similarity_threshold, RERANK_CANDIDATES
                )
                all_analysis_results.append(result)

        if all_analysis_results:
            results_df = pd.DataFrame(all_analysis_results)

            st.subheader("Individual Task Analysis:")
            st.markdown("""
            **Automatable Tasks** (done by AI)
            - **Directive:** Complete task delegation with minimal interaction.
            - **Feedback Loop:** Task completion guided by environmental feedback.

            **Augmentable Tasks** (AI helping humans)
            - **Task Iteration:** Collaborative refinement process.
            - **Learning:** Knowledge acquisition and understanding.
            - **Validation:** Work verification and improvement.

            **Other**
            - **Potentially Sensitive** (Filtered): Requires more human judgment or is potentially too sensitive for third-party models.
            """)

            st.dataframe(results_df[[
                'classification_type','original_task_input','matched_task_name',
                'initial_similarity', 'final_similarity',
                'feedback_loop', 'directive', 'task_iteration', 'validation', 'learning', 'filtered'
            ]].style.format({'initial_similarity': "{:.3f}",'final_similarity': "{:.3f}"}))

            st.subheader("Overall Job Summary:")

            matched_tasks_df = results_df[results_df['classification_type'] == 'Matched Task']
            total_input_tasks = len(input_tasks)
            num_matched_tasks = len(matched_tasks_df)
            num_unmatched_tasks = total_input_tasks - num_matched_tasks

            st.write(f"**Total Tasks Entered:** {total_input_tasks}")
            st.write(f"**Tasks with a Close Match Found:** {num_matched_tasks}")
            st.write(f"**Unmatched Tasks:** {num_unmatched_tasks}")

            if num_matched_tasks > 0:
                total_automatable_sum_matched = matched_tasks_df['feedback_loop'].sum() + matched_tasks_df['directive'].sum()
                total_augmentable_sum_matched = matched_tasks_df['task_iteration'].sum() + matched_tasks_df['validation'].sum() + matched_tasks_df['learning'].sum()
                total_potentially_sensitive_sum_matched = matched_tasks_df['filtered'].sum()
                overall_matched_work_total = total_automatable_sum_matched + total_augmentable_sum_matched + total_potentially_sensitive_sum_matched

                if overall_matched_work_total > 0:
                    percent_automatable_matched = (total_automatable_sum_matched / overall_matched_work_total) * 100
                    percent_augmentable_matched = (total_augmentable_sum_matched / overall_matched_work_total) * 100
                    percent_potentially_sensitive_matched = (total_potentially_sensitive_sum_matched / overall_matched_work_total) * 100
                else:
                    percent_automatable_matched = percent_augmentable_matched = percent_potentially_sensitive_matched = 0

                st.markdown("### Breakdown of *Matched* Tasks:")
                st.markdown(f"**Potentially Automatable** (Feedback Loop, Directive): **{percent_automatable_matched:.1f}%**")
                st.markdown(f"**Potentially Augmentable** (Task Iteration, Validation, Learning): **{percent_augmentable_matched:.1f}%**")
                st.markdown(f"**Potentially Sensitive** (Filtered): **{percent_potentially_sensitive_matched:.1f}%**")

            st.markdown("### Job Coverage & Unmatched Tasks:")
            percent_unmatched_of_total = (num_unmatched_tasks / total_input_tasks) * 100 if total_input_tasks > 0 else 0
            st.markdown(f"**Percentage of Tasks Unmatched:** **{percent_unmatched_of_total:.1f}%**")
            st.info("Unmatched tasks mean there isn't a similar task in our database based on the 'Minimum Match Strength'.")

        else:
            st.warning("No tasks were entered or processed. Please try again.")
    else:
        st.warning("Please enter job tasks to analyze.")
