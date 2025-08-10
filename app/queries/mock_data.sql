INSERT INTO `report_cache_dataset.semantic_object_cache` (object_id, object_type, prompt_text, gcs_uri, prompt_embedding, created_at, last_accessed_at)
VALUES
  (
    'e1f2a3b4-c5d6-7890-1234-567890abcdef',
    'VIDEO',
    'A time-lapse of a city skyline from dusk to dawn',
    'gs://your-media-bucket/timelapse_city.mp4',
    ARRAY(SELECT x * 0.0001 FROM UNNEST(GENERATE_ARRAY(1, 3072)) AS x),
    '2025-07-22 20:10:00 UTC',
    '2025-07-22 20:15:30 UTC'
  ),
  (
    'f2a3b4c5-d6e7-8901-2345-67890abcdef1',
    'IMAGE',
    'Impressionist painting of a robot playing chess in a park',
    'gs://your-media-bucket/robot_chess.png',
    ARRAY(SELECT x * 0.0002 FROM UNNEST(GENERATE_ARRAY(1, 3072)) AS x),
    '2025-07-22 11:00:00 UTC',
    NULL
  ),
  (
    'a3b4c5d6-e7f8-9012-3456-7890abcdef12',
    'VIDEO',
    'Close-up shot of a bee collecting pollen from a sunflower, slow motion',
    'gs://your-media-bucket/bee_pollen_slowmo.mp4',
    ARRAY(SELECT x * -0.0001 FROM UNNEST(GENERATE_ARRAY(1, 3072)) AS x),
    '2025-07-18 14:30:00 UTC',
    NULL
  ),
  (
    'b4c5d6e7-f8a9-0123-4567-890abcdef123',
    'IMAGE',
    'A logo for a coffee shop named "The Daily Grind", minimalist style',
    'gs://your-media-bucket/daily_grind_logo.svg',
    ARRAY(SELECT x * 0.0003 FROM UNNEST(GENERATE_ARRAY(1, 3072)) AS x),
    '2025-07-15 09:00:00 UTC',
    '2025-07-20 16:45:00 UTC'
  );


-- test query
SELECT
  base.gcs_uri,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `report_cache_dataset.semantic_object_cache`,
    'prompt_embedding',
    (SELECT [0.011, -0.035, 0.023, 0.048, -0.019, 0.002, 0.021, -0.031] AS prompt_embedding),
    top_k => 1,
    distance_type => 'COSINE'
  )
WHERE
  base.object_type = 'VIDEO'

--- test queries
-- Check the dimension of your STORED data
SELECT ARRAY_LENGTH(prompt_embedding) AS stored_dimension
FROM `report_cache_dataset.semantic_object_cache`
LIMIT 1;

-- Check the dimension of your QUERY vector
-- (You'll need to set the @query_embedding parameter first)
SELECT ARRAY_LENGTH(@query_embedding) AS query_dimension;

For a vector search to work, all vectors must have the exact same number of dimensions.