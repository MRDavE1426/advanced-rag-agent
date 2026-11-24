# Hybrid Search

Hybrid search combines dense semantic retrieval with sparse lexical retrieval. Dense retrieval embeds the query and the documents into vectors and finds the nearest neighbours, which works well when the query and the document use different words for the same idea. Sparse retrieval, such as BM25, scores documents by exact keyword overlap, which works well for rare terms, product names, error codes and identifiers that embeddings often blur together.

Neither approach is best on its own. A dense index may miss an exact identifier, while BM25 cannot match a paraphrase. Running both and merging the ranked lists gives more robust recall than either list alone.

Reciprocal Rank Fusion (RRF) is a simple way to merge the lists. Each document receives a score of 1 / (k + rank) from every list it appears in, where rank starts at 1 and k is a smoothing constant, commonly 60. The scores are summed and the documents are sorted by the total. RRF only uses ranks, so it does not need the dense and sparse scores to be on the same scale.
