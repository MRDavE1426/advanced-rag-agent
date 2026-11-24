# Long-Term Semantic Memory

A chat model has no memory between sessions. Semantic memory systems such as Mem0 and Zep add persistence. After each interaction, an extraction step uses a language model to pull durable facts about the user out of the conversation, such as the programming languages they use, the level of detail they want, and the style of answer they prefer. These facts are embedded and stored in a vector database keyed by user id.

When the same user starts a new session, the stored facts are loaded and inserted into the system prompt. The assistant can then adapt its behaviour, for example answering with code only when the user has said they prefer brief, code-only responses, without the user repeating the preference.
