# Re-ranking and Context Compression

First-stage retrieval is optimised for speed and recall, so its top results contain noise. A re-ranker is a cross-encoder model that reads the query and each candidate together and produces a relevance score. It is slower than vector search, so it is applied only to a short candidate list.

A common setup retrieves the top 20 candidates from the hybrid search and passes them to Cohere ReRank. The re-ranker returns the candidates ordered by relevance score, and only the top 5 are kept. Sending 5 chunks instead of 20 reduces the number of prompt tokens, which lowers cost and latency and reduces the chance that irrelevant text distracts the model and causes hallucinations.

A minimum relevance score can also be applied. Chunks scoring below the threshold are dropped even if they are within the top 5, so a query with no good matches does not fill the prompt with unrelated text.
