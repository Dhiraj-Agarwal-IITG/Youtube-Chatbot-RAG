## Project Highlights

- Built an end-to-end **RAG (Retrieval-Augmented Generation)** chatbot for YouTube videos using **LangChain + FAISS + OpenAI**.
- Converts a user-provided YouTube URL into a queryable knowledge base by:
  1. fetching transcript,
  2. chunking text,
  3. creating embeddings,
  4. storing vectors in FAISS,
  5. retrieving relevant context per question.
- Implemented a clean Streamlit UI for non-technical users to process videos and ask contextual questions in real time.
- Added guardrails to reduce hallucinations by prompting the model to answer from retrieved transcript context only.
- Improved reliability with session-state handling (clearing stale context on failed processing) and user-friendly error handling.

## Why This Project Matters

This project demonstrates practical LLM application skills beyond prompting:
- RAG architecture design
- vector indexing and retrieval strategy
- prompt grounding and response control
- web app integration for real user interaction
- debugging state-related issues in interactive apps

## Tech Stack

`Python` `Streamlit` `LangChain` `OpenAI` `FAISS` `YouTube Transcript API`
