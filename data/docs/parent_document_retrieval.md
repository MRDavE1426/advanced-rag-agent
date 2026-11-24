# Parent-Document Retrieval

Small chunks are good for matching because each one is about a single idea, so its embedding is focused. Small chunks are bad for answering because they often lack the surrounding context a language model needs.

The parent-document pattern solves this by indexing small child chunks, for example 200 characters, while keeping a link to a larger parent chunk, for example 1000 characters. The child chunks are what the vector index and the keyword index search over. When a child chunk matches a query, the retriever looks up its parent in a document store and returns the full parent text to the language model.

Several children can point to the same parent. The retriever should therefore de-duplicate parents so the same passage is not sent to the model twice. The parents themselves live in a simple key-value document store, because they never need to be searched directly.
