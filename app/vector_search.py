import os

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from google.cloud import bigquery
from datetime import datetime
import uuid
import logging

# --- Configuration Constants ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
BQ_DATASET_ID = "video_cache_dataset"
BQ_TABLE_ID = "semantic_cache"
SIMILARITY_THRESHOLD = 0.15 # For COSINE distance, lower is more similar. Tune this value.


def _search_for_similar_video(query_prompt: str) -> tuple[str, float] | None:
    """
    Generates a query embedding and searches BigQuery for the most similar vector.

    Returns:
        A tuple of (gcs_uri, distance) if a match is found, otherwise None.
    """
    # 1. Generate the query-specific embedding
    query_embedding = get_text_embedding(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        text_content=query_prompt,
        task_type="RETRIEVAL_QUERY",
    )

    if not query_embedding:
        logging.error("Failed to generate query embedding. Cannot perform search.")
        return None

    # 2. Execute the parameterized VECTOR_SEARCH query
    try:
        client = bigquery.Client(project=GCP_PROJECT_ID)

        # The object type to search for
        object_type_to_search = 'VIDEO'

        # The SQL query now uses a named parameter @object_type
        # Without an index, it simply performs a brute-force search.
        sql_query = f"""
        SELECT
          base.gcs_uri,
          distance
        FROM
          VECTOR_SEARCH(
            TABLE `report_cache_dataset.semantic_object_cache`,
            'prompt_embedding',
            (SELECT @query_embedding AS prompt_embedding),
            top_k => 1,
            distance_type => 'COSINE'
          )
        WHERE
          base.object_type = @object_type
        """

        # Define the query parameter for the embedding vector
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
                bigquery.ScalarQueryParameter("object_type", "STRING", object_type_to_search)
            ]
        )

        query_job = client.query(sql_query, job_config=job_config)
        results = list(query_job.result()) # Materialize the results into a list

        # Correctly access the first row of the results
        if results:
            top_result = results[0]
            return top_result.gcs_uri, top_result.distance
        else:
            return None

    except Exception as e:
        logging.error(f"An error occurred during BigQuery vector search: {e}")
        return None


def get_cached(prompt: str) -> str | None:
    """
    Orchestrates the semantic cache workflow.

    1. Searches for a semantically similar video in BigQuery.
    2. If a close match is found (cache hit), returns its GCS URI.
    3. If no close match is found (cache miss), generates a new video,
       caches it, and returns the new GCS URI.
    """
    logging.info(f"Processing prompt: '{prompt}'")

    # --- Retrieval Path ---
    search_result = _search_for_similar_video(prompt)

    if search_result:
        gcs_uri, distance = search_result
        logging.info(f"Found a potential match with distance: {distance:.4f}")

        if distance <= SIMILARITY_THRESHOLD:
            logging.info(f"CACHE HIT. Returning existing video: {gcs_uri}")
            # Optionally, update the 'last_accessed_at' timestamp here
            return gcs_uri
        else:
            logging.info(
                f"CACHE MISS. Match found but distance ({distance:.4f}) exceeds threshold ({SIMILARITY_THRESHOLD}).")
    else:
        logging.info("CACHE MISS. No similar video found in the cache.")

    # --- Ingestion Path (triggered on cache miss) ---
    logging.info("Generating a new video...")

def insert(prompt: str, new_gcs_uri: str) -> None:
    logging.info(f"New video generated at: {new_gcs_uri}. Caching result...")

    # Generate embedding for the new document
    doc_embedding = get_text_embedding(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        text_content=prompt,
        task_type="RETRIEVAL_DOCUMENT"
    )

    if doc_embedding:
        # Insert the new record into BigQuery
        insert_video_record(
            project_id=GCP_PROJECT_ID,
            dataset_id=BQ_DATASET_ID,
            table_id=BQ_TABLE_ID,
            prompt=prompt,
            gcs_uri=new_gcs_uri,
            embedding=doc_embedding
        )
    else:
        logging.error("Failed to generate embedding for the new video. Result not cached.")


def insert_video_record(
    project_id: str,
    dataset_id: str,
    table_id: str,
    prompt: str,
    gcs_uri: str,
    embedding: list[float],
) -> bool:
    """
    Inserts a new video record into the BigQuery semantic cache table.

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.
        prompt: The original user prompt.
        gcs_uri: The GCS URI of the generated video.
        embedding: The text embedding of the prompt.

    Returns:
        True if insertion was successful, False otherwise.
    """
    try:
        client = bigquery.Client(project=project_id)
        table_full_id = f"{project_id}.{dataset_id}.{table_id}"

        rows_to_insert = [
            {
                "video_id": str(uuid.uuid4()),
                "prompt_text": prompt,
                "gcs_uri": gcs_uri,
                "prompt_embedding": embedding,
                "created_at": datetime.utcnow(), # Pass datetime object directly
                "last_accessed_at": None,
            }
        ]

        errors = client.insert_rows_json(table_full_id, rows_to_insert)

        # insert_rows_json returns an empty list for success.
        if not errors:
            logging.info(f"Successfully inserted 1 row for prompt: '{prompt[:50]}...'")
            return True
        else:
            logging.error(f"Encountered errors while inserting rows: {errors}")
            return False

    except Exception as e:
        logging.error(f"An error occurred during BigQuery insertion: {e}")
        return False


def get_text_embedding(
    project_id: str,
    location: str,
    text_content: str,
    task_type: str,
    output_dimensionality: int | None = None,
) -> list[float] | None:
    """
    Generates a text embedding using the gemini-embedding-004 model.

    Args:
        project_id: The Google Cloud project ID.
        location: The Google Cloud region (e.g., 'us-central1').
        text_content: The text to embed.
        task_type: The task type for the embedding ('RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', etc.).
        output_dimensionality: The desired size of the output embedding vector.

    Returns:
        A list of floats representing the embedding, or None if an error occurs.
    """
    try:
        vertexai.init(project=project_id, location=location)
        model = TextEmbeddingModel.from_pretrained("text-embedding-004") # Note: gemini-embedding-001 is a legacy model name. text-embedding-004 is current.

        # Ensure task_type is valid
        valid_tasks = [
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
        ]
        if task_type not in valid_tasks:
            raise ValueError(
                f"Invalid task_type: {task_type}. Must be one of {valid_tasks}"
            )

        embedding_input = TextEmbeddingInput(text=text_content, task_type=task_type)

        params = {}
        if output_dimensionality:
            params["output_dimensionality"] = output_dimensionality

        # The get_embeddings method returns a list of TextEmbedding objects
        embeddings = model.get_embeddings([embedding_input], **params)

        # Each TextEmbedding object has a 'values' attribute with the embedding vector.
        # Since we are sending one text, we access the first element of the list.
        if embeddings:
            return embeddings[0].values
        else:
            logging.error("Failed to retrieve embedding values from the API response.")
            return None

    except Exception as e:
        logging.error(f"An error occurred during embedding generation: {e}")
        return None
