CREATE SCHEMA IF NOT EXISTS report_cache_dataset
OPTIONS(
  location = 'US'
);

CREATE TABLE IF NOT EXISTS `report_cache_dataset.semantic_object_cache` (
  object_id STRING NOT NULL OPTIONS(description="Unique identifier for the record, e.g., a UUID"),
  object_type STRING NOT NULL OPTIONS(description="Object type, e.g., VIDEO/IMAGE"),
  prompt_text STRING OPTIONS(description="The original user prompt used for generation"),
  gcs_uri STRING NOT NULL OPTIONS(description="Cloud Storage URI of the generated"),
  prompt_embedding ARRAY<FLOAT64> OPTIONS(description="Embedding vector from gemini-embedding-001"),
  created_at TIMESTAMP NOT NULL OPTIONS(description="Timestamp of when the record was created"),
  last_accessed_at TIMESTAMP OPTIONS(description="Timestamp of the last cache hit for lifecycle management")
);