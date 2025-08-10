-- index_type = 'IVF': Specifies the Inverted File index, a proven and effective ANN algorithm.
--
-- distance_type = 'COSINE': Cosine similarity is the standard and most appropriate distance metric for comparing text embeddings. It measures the angle between vectors, capturing semantic similarity regardless of vector magnitude.
--
-- ivf_options = '{"num_lists": 100}': This is a key tuning parameter. num_lists determines the number of clusters the data is partitioned into. A higher number allows for more granular (and potentially faster/cheaper) queries by searching a smaller fraction of the total lists. A starting value of 100 is reasonable for a dataset expected to grow into the tens of thousands or hundreds of thousands of records.

-- Total rows 4 is smaller than min allowed 5000 for CREATE VECTOR INDEX query with the IVF index type. Please use VECTOR_SEARCH table-valued function directly to perform the similarity search.
CREATE OR REPLACE VECTOR INDEX object_embedding_ivf_index
ON `report_cache_dataset.semantic_object_cache`(prompt_embedding)
OPTIONS(
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 100}'
);