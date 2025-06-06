import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import io
from scipy.special import expit # Import the sigmoid function
import faiss # Import FAISS

# --- Configuration ---
DATA_FILE_PATH = 'job_tasks_data.csv' # Path to your dataset
RERANK_CANDIDATES = 10 # Number of top candidates to send to the Re-ranker

# --- Load Models ---
@st.cache_resource # Cache the initial search model
def load_initial_search_model():
    """Loads the model for initial quick search (finds many possible matches)."""
    # Using embeddings model for initial retrieval
    return SentenceTransformer('paraphrase-mpnet-base-v2')

@st.cache_resource # Cache the re-ranker model
def load_reranker_model():
    """Loads the specialized re-ranker model (refines the initial matches)."""
    # Using an embeddings model for re-ranking
    return CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1')

# --- Load Data and Process ---
@st.cache_data # Cache data loading, embedding computation, and FAISS index creation
def load_and_process_task_data(filepath, _initial_search_model):
    """
    Loads the task dataset, computes embeddings for quick search, and builds a FAISS index.
    """
    try:
        df = pd.read_csv(filepath)

        df.columns = df.columns.str.strip()

        metric_cols = ['feedback_loop', 'directive', 'task_iteration', 'validation', 'learning', 'filtered']
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        st.write("First-time setup: Generating task descriptions for quick search and organizing data... (This happens once)")
        df['embedding'] = df['task_name'].apply(lambda x: _initial_search_model.encode(x) if pd.notna(x) else None)
        df = df.dropna(subset=['embedding']) # Remove rows where embedding failed

        # Prepare embeddings for FAISS
        embeddings_matrix = np.vstack(df['embedding'].values).astype('float32')
        dimension = embeddings_matrix.shape[1]

        # Create and build FAISS index for fast initial search
        # IndexFlatL2 performs a brute-force L2 distance search (exact nearest neighbor)
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings_matrix)
        st.write(f"Data organized for fast search with {faiss_index.ntotal} tasks.")

        return df, faiss_index

    except FileNotFoundError:
        st.error(f"Error: Data file not found at {filepath}. Please ensure 'job_tasks_data.csv' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
        st.stop()

# --- Task Matching and Analysis Function ---
def analyze_single_task(query_task_text, df, initial_search_model, reranker_model, faiss_index, initial_similarity_threshold, rerank_candidates=RERANK_CANDIDATES):
    """
    Analyzes a single job task to find the closest match and its characteristics.
    Uses a two-stage process: fast initial search with FAISS, then refined re-ranking.
    """
    query_embedding = initial_search_model.encode(query_task_text).astype('float32').reshape(1, -1)

    # 1. Initial Quick Search (using FAISS)
    # Search FAISS index for top_k candidates to send to the more powerful re-ranker
    k_candidates_for_faiss = max(RERANK_CANDIDATES, 20) # Retrieve at least 20 for robustness
    distances, faiss_indices = faiss_index.search(query_embedding, k_candidates_for_faiss) # L2 distances, indices

    # Convert L2 distances to cosine similarity for 'initial_similarity' (assuming embeddings are normalized)
    initial_cosine_similarities = 1 - (distances[0]**2) / 2

    # Filter candidates by the user-defined initial similarity threshold
    valid_candidate_mask = initial_cosine_similarities >= initial_similarity_threshold

    filtered_faiss_indices = faiss_indices[0][valid_candidate_mask]
    filtered_cosine_similarities = initial_cosine_similarities[valid_candidate_mask]

    if len(filtered_faiss_indices) == 0:
        # If no initial matches above threshold, categorize as 'Unmatched'
        return {
            'original_task_input': query_task_text,
            'matched_task_name': 'No close match found',
            'initial_similarity': 0.0,
            'similarity': 0.0, # Re-ranked similarity for unmatched tasks is 0
            'feedback_loop': 0.0,
            'directive': 0.0,
            'task_iteration': 0.0,
            'validation': 0.0,
            'learning': 0.0,
            'filtered': 0.0, # 'Filtered' probability from dataset is 0 for unmatched
            'classification_type': 'Unmatched Task' # New classification for summary
        }

    # Sort filtered candidates by cosine similarity and take top RERANK_CANDIDATES for re-ranking
    sorted_candidate_indices_in_filtered = np.argsort(filtered_cosine_similarities)[::-1][:RERANK_CANDIDATES]
    sorted_df_indices_for_reranking = filtered_faiss_indices[sorted_candidate_indices_in_filtered]

    # Get the actual task names for re-ranking
    candidate_texts = df.iloc[sorted_df_indices_for_reranking]['task_name'].tolist()
    sentence_pairs = [[query_task_text, candidate_text] for candidate_text in candidate_texts]

    # 2. Refined Re-ranking (using specialized Re-ranker)
    raw_reranked_scores = reranker_model.predict(sentence_pairs)
    # Apply sigmoid to normalize scores to 0-1 for easier understanding
    normalized_reranked_scores = expit(raw_reranked_scores)

    # Find the best match among the re-ranked candidates based on normalized scores
    best_rerank_idx_in_candidates = np.argmax(normalized_reranked_scores)
    final_best_match_df_idx = sorted_df_indices_for_reranking[best_rerank_idx_in_candidates]

    # Get the row of the best re-ranked match from the original dataset
    matched_task_row = df.iloc[final_best_match_df_idx]

    return {
        'original_task_input': query_task_text,
        'matched_task_name': matched_task_row['task_name'],
        'initial_similarity': initial_cosine_similarities[valid_candidate_mask][sorted_candidate_indices_in_filtered][best_rerank_idx_in_candidates],
        'similarity': normalized_reranked_scores[best_rerank_idx_in_candidates],
        'feedback_loop': matched_task_row['feedback_loop'],
        'directive': matched_task_row['directive'],
        'task_iteration': matched_task_row['task_iteration'],
        'validation': matched_task_row['validation'],
        'learning': matched_task_row['learning'],
        'filtered': matched_task_row['filtered'],
        'classification_type': 'Matched Task' # New classification for summary
    }

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("Job Task Automation & Augmentation Estimator")

st.markdown("""
This app helps estimate how much of a job can be **automated** (done by machines) or **augmented** (machines helping humans).
Just enter a list of job tasks, and the app will try to find similar tasks from our database and calculate their potential for automation and augmentation.
""")

# Load models and data
initial_search_model = load_initial_search_model()
reranker_model = load_reranker_model()

with st.spinner("Setting up the brain of the app..."):
    df_tasks, faiss_index = load_and_process_task_data(DATA_FILE_PATH, initial_search_model)

# User input for similarity threshold
st.sidebar.header("How the App Matches Tasks")
selected_similarity_threshold = st.sidebar.slider(
    "Minimum Match Strength (for initial search)",
    min_value=0.0,
    max_value=1.0,
    value=0.75, # Default value
    step=0.01,
    help="Adjust how 'similar' a task needs to be to find a match in our database. Tasks below this strength will be categorized as 'Unmatched'."
)
st.sidebar.info(f"Current minimum match strength: {selected_similarity_threshold:.2f}")

default_job_tasks = """Develop narratives and materials that clearly communicate the value of our unified management solution and Digital Experience Management capabilities, translating technical features into compelling messages that resonate with customers

Create impactful marketing content, including datasheets, whitepapers, blogs, and presentations

Define and implement GTM strategies for new launches and campaigns in collaboration with cross-functional marketing teams

Contribute to our vision for how the AI-powered Strata Network Security Platform is essential in securing customers' networks, both at internal and external events

Work closely with Product Management and Product Marketing peers to build a unified market and product vision, developing value propositions and messaging that resonate with our target buyer personas and form the foundation for all product marketing activities

Serve as the primary thought leader for the products you support, engaging in speaking opportunities and content development

Simplify product capabilities into clear, compelling messages and tools that resonate with customers

Create powerful marketing content, including datasheets, whitepapers, blogs, and presentations

Educate and inspire our Sales, GTM, and field marketing teams on the customer and business value of our products
"""

job_tasks_input = st.text_area("Enter Job Tasks (one per line):", height=350,
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
            Here's a breakdown for each task entered:
            - **Similarity:** How closely the task matches a task in our database (0-1, 1 is a perfect match).
            - **Initial Match Score:** The first quick score before detailed analysis.
            - **Matched Task in Database:** The closest task we found in our dataset.
            - **Automatable Probabilities:**
                - **Feedback Loop:** Task completion guided by environmental feedback.
                - **Directive:** Complete task delegation with minimal interaction.
            - **Augmentable Probabilities:**
                - **Task Iteration:** Collaborative refinement process.
                - **Validation:** Work verification and improvement.
                - **Learning:** Knowledge acquisition and understanding.
            - **Potentially Sensitive Probability:** Task requires sensitive handling by humans (according to Anthropic).
            - **Classification:** Whether we found a close match or it's an "Unmatched Task".
            """)
            st.dataframe(results_df[[
                'similarity', # Primary re-ranked similarity, 0-1
                'initial_similarity', # Original bi-encoder cosine similarity
                'original_task_input',
                'matched_task_name',
                'feedback_loop',
                'directive',
                'task_iteration',
                'validation',
                'learning',
                'filtered', # Now 'Potentially Sensitive' in interpretation
                'classification_type' # Display the classification
            ]].style.format({'similarity': "{:.4f}", 'initial_similarity': "{:.2f}"}))

            st.subheader("Overall Job Summary:")
            st.markdown("""
            This summary provides an estimate for the **entire job** based on the tasks entered.
            """)

            # Separate matched and unmatched tasks
            matched_tasks_df = results_df[results_df['classification_type'] == 'Matched Task']
            unmatched_tasks_df = results_df[results_df['classification_type'] == 'Unmatched Task']

            total_input_tasks = len(input_tasks)
            num_matched_tasks = len(matched_tasks_df)
            num_unmatched_tasks = len(unmatched_tasks_df)

            st.write(f"**Total Tasks Entered:** {total_input_tasks}")
            st.write(f"**Tasks with a Close Match Found:** {num_matched_tasks}")
            st.write(f"**Unmatched Tasks:** {num_unmatched_tasks}")

            if num_matched_tasks > 0:
                # Calculate the overall categories based ONLY on the matched tasks' probabilities
                total_automatable_sum_matched = matched_tasks_df['feedback_loop'].sum() + matched_tasks_df['directive'].sum()
                total_augmentable_sum_matched = matched_tasks_df['task_iteration'].sum() + matched_tasks_df['validation'].sum() + matched_tasks_df['learning'].sum()
                total_potentially_sensitive_sum_matched = matched_tasks_df['filtered'].sum()

                # Calculate percentages relative to the sum of these three categories for matched tasks
                # This represents the breakdown *of the work that was matched*.
                overall_matched_work_total = total_automatable_sum_matched + total_augmentable_sum_matched + total_potentially_sensitive_sum_matched

                if overall_matched_work_total > 0:
                    percent_automatable_matched = (total_automatable_sum_matched / overall_matched_work_total) * 100
                    percent_augmentable_matched = (total_augmentable_sum_matched / overall_matched_work_total) * 100
                    percent_potentially_sensitive_matched = (total_potentially_sensitive_sum_matched / overall_matched_work_total) * 100
                else:
                    percent_automatable_matched = percent_augmentable_matched = percent_potentially_sensitive_matched = 0

                st.markdown("### Breakdown of *Matched* Tasks:")
                st.markdown(f"**Potentially Automatable** (tasks machines can do, following rules/instructions): **{percent_automatable_matched:.2f}%**")
                st.markdown(f"**Potentially Augmentable** (tasks where machines can help humans be more effective): **{percent_augmentable_matched:.2f}%**")
                st.markdown(f"**Potentially Sensitive** (tasks requiring more sensitive handling by a human): **{percent_potentially_sensitive_matched:.2f}%**")

            st.markdown("### Job Coverage & Unmatched Tasks:")
            # Percentages for unmatched/sensitive relative to total input tasks
            percent_unmatched_of_total = (num_unmatched_tasks / total_input_tasks) * 100

            st.markdown(f"**Percentage of Tasks Unmatched** (no close match found in our database): **{percent_unmatched_of_total:.2f}%**")
            st.info("Unmatched tasks mean there isn't a similar task in our database based on the 'Minimum Match Strength'.")

            st.markdown("---")

        else:
            st.warning("No tasks were entered or processed. Please try again.")
    else:
        st.warning("Please enter some job tasks to analyze.")
