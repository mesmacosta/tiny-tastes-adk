import io
import json
import os
from functools import lru_cache

from google import genai

from google.genai.types import EmbedContentConfig


from google.cloud import bigquery
from datetime import datetime, UTC
import uuid
import logging

# --- Configuration Constants ---
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
BQ_DATASET_ID = "report_cache_dataset"
BQ_TABLE_ID = "semantic_object_cache"
SIMILARITY_THRESHOLD = 0.15 # For COSINE distance, lower is more similar. Tune this value.


def _search_for_similar_object(query_prompt: str, object_type_to_search: str) -> tuple[str, float] | None:
    """
    Generates a query embedding and searches BigQuery for the most similar vector.

    Returns:
        A tuple of (gcs_uri, distance) if a match is found, otherwise None.
    """
    # 1. Generate the query-specific embedding
    query_embedding = get_text_embedding(
        text_content=query_prompt,
    )

    if not query_embedding:
        logging.error("Failed to generate query embedding. Cannot perform search.")
        return None

    # 2. Execute the parameterized VECTOR_SEARCH query
    try:
        client = bigquery.Client(project=GCP_PROJECT_ID)

        # The SQL query now uses a named parameter @object_type
        # Without an index, it simply performs a brute-force search.
        sql_query = f"""
        SELECT
          base.gcs_uri,
          distance
        FROM
          VECTOR_SEARCH(
            (SELECT * FROM `report_cache_dataset.semantic_object_cache` WHERE object_type = @object_type),
            'prompt_embedding',
            (SELECT @query_embedding AS prompt_embedding),
            top_k => 1,
            distance_type => 'COSINE'
          )
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


def get_cached(prompt: str, object_type_to_search: str) -> str | None:
    """
    Orchestrates the semantic cache workflow.

    1. Searches for a semantically similar object in BigQuery.
    2. If a close match is found (cache hit), returns its GCS URI.
    3. If no close match is found (cache miss), generates a new object,
       caches it, and returns the new GCS URI.
    """
    logging.info(f"Processing prompt: '{prompt}'")

    # --- Retrieval Path ---
    search_result = _search_for_similar_object(prompt, object_type_to_search)

    if search_result:
        gcs_uri, distance = search_result
        logging.info(f"Found a potential match with distance: {distance:.4f}")

        if distance <= SIMILARITY_THRESHOLD:
            logging.info(f"CACHE HIT. Returning existing object: {gcs_uri}")
            # Optionally, update the 'last_accessed_at' timestamp here
            return gcs_uri
        else:
            logging.info(
                f"CACHE MISS. Match found but distance ({distance:.4f}) exceeds threshold ({SIMILARITY_THRESHOLD}).")
    else:
        logging.info("CACHE MISS. No similar object found in the cache.")

def insert(object_type: str, prompt: str, new_gcs_uri: str) -> None:
    logging.info(f"New object generated at: {new_gcs_uri}. Caching result...")

    # Generate embedding for the new document
    doc_embedding = get_text_embedding(
        text_content=prompt,
    )

    if doc_embedding:
        # Insert the new record into BigQuery
        insert_object_record(
            project_id=GCP_PROJECT_ID,
            dataset_id=BQ_DATASET_ID,
            table_id=BQ_TABLE_ID,
            prompt=prompt,
            gcs_uri=new_gcs_uri,
            embedding=doc_embedding,
            object_type=object_type
        )
    else:
        logging.error("Failed to generate embedding for the new object. Result not cached.")

def insert_object_record(
    project_id: str,
    dataset_id: str,
    table_id: str,
    prompt: str,
    gcs_uri: str,
    embedding: list[float],
    object_type: str,
    load_type: str = 'streaming',
) -> bool:
    """
    Inserts a new object record into the BigQuery semantic cache table.

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.
        prompt: The original user prompt.
        gcs_uri: The GCS URI of the generated object.
        embedding: The text embedding of the prompt.
        object_type: Object type (e.g., VIDEO/IMAGE).
        load_type: The insertion method to use ('streaming' or 'load_job').

    Returns:
        True if insertion was successful, False otherwise.
    """
    try:
        client = bigquery.Client(project=project_id)
        table_full_id = f"{project_id}.{dataset_id}.{table_id}"

        # Prepare the row to be inserted
        row_to_insert = {
            "object_id": str(uuid.uuid4()),
            "object_type": object_type,
            "prompt_text": prompt,
            "gcs_uri": gcs_uri,
            "prompt_embedding": embedding,
            "created_at": datetime.now(UTC).isoformat(),
            "last_accessed_at": None,
        }

        if load_type == 'load_job':
            # Use a load job to insert the data
            json_data = json.dumps(row_to_insert).encode("utf-8")
            binary_buffer = io.BytesIO(json_data)
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            )
            load_job = client.load_table_from_file(
                binary_buffer, table_full_id, job_config=job_config
            )
            load_job.result()  # Wait for the job to complete

            if load_job.errors:
                logging.error(f"Encountered errors while inserting rows: {load_job.errors}")
                return False
        else:
            # Use the streaming API to insert the data
            errors = client.insert_rows_json(table_full_id, [row_to_insert])
            if errors:
                logging.error(f"Encountered errors while inserting rows: {errors}")
                return False

        logging.info(f"Successfully inserted 1 row for prompt: '{prompt[:50]}...'")
        return True

    except Exception as e:
        logging.error(f"An error occurred during BigQuery insertion: {e}")
        return False


@lru_cache(maxsize=128)
def get_text_embedding(
    text_content: str,
    output_dimensionality: int | None = None,
) -> list[float] | None:
    """
    Generates a text embedding using the gemini-embedding-004 model.

    Args:
        text_content: The text to embed.
        output_dimensionality: The desired size of the output embedding vector.

    Returns:
        A list of floats representing the embedding, or None if an error occurs.
    """
    try:
        client = genai.Client(location=GCP_LOCATION, project=GCP_PROJECT_ID)
        embeddings_resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text_content,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=output_dimensionality,
            ),
        )

        # Each TextEmbedding object has a 'values' attribute with the embedding vector.
        # Since we are sending one text, we access the first element of the list.
        if embeddings_resp:
            return embeddings_resp.embeddings[0].values
        else:
            logging.error("Failed to retrieve embedding values from the API response.")
            return None

    except Exception as e:
        logging.error(f"An error occurred during embedding generation: {e}")
        return None
