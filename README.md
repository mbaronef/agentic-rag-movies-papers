🇬🇧 English | 🇪🇸 [Español](README.es.md)

# Agentic RAG - Movies & Academic Papers

An **agentic RAG** system built as a group academic project, able to answer questions about a movie dataset (Letterboxd) and a set of academic papers, combining information retrieval, reasoning, and conversational memory.

---

## Overview

The agent integrates three main capabilities:

- **RAG over two knowledge sources**: a structured movie dataset and six academic papers in PDF.
- **Tool-calling with LangGraph**: the LLM dynamically decides whether to query the movie dataset, look up theoretical concepts in the PDFs, access recent memory, or answer directly (small talk).
- **Short-term memory**: the last interactions are stored in the agent's state, enabling references like "that", "the previous one", or "what did I tell you", to keep the conversation coherent.

The design aims to prevent hallucinations: the agent never generates factual content without evidence previously retrieved from the vector stores.

## Architecture

The pipeline is organized into four layers:

1. **Data loading and preprocessing** - normalization of the movie dataset (Letterboxd, 28 columns) and of the 6 papers (loaded per page with `PyPDFLoader`, lemmatization, abbreviation expansion, stopword removal).
2. **Chunking and embeddings** - `RecursiveCharacterTextSplitter` (chunk size 500/overlap 50 for movies, 700/100 for papers) and embeddings with `sentence-transformers/all-MiniLM-L6-v2`, indexed into two independent **Pinecone** indexes (`letterboxd` and `rag-pdfs-obligatorio`).
3. **Specialized semantic retrieval via tools** - three dedicated tools:
   - `buscar_peliculas`: normalizes the query, builds dynamic metadata filters (country, decade, genre, language) and runs a hybrid search (`top_k=10`).
   - `buscar_papers`: expands technical terminology (RAG, embeddings, attention, GPT) and searches the papers index (`top_k=3`).
   - `consultar_memoria`: resolves references to previous interactions using an auxiliary array of the last 5 interactions.
4. **Agentic reasoning and controlled generation** - graph built with **LangGraph**, with a decision node (`chatbot_node`) that detects small talk manually and delegates the rest of intent classification to the LLM via `bind_tools`. The LLM used is `Qwen/Qwen3-4B-Instruct` via `HuggingFaceEndpoint` (`temperature=0.3`, `repetition_penalty=1.1`, `max_new_tokens=512`).

![graph diagram](image.png)

## Prompt engineering

Three types of prompts are combined:

- **Main system prompt**: defines the agent's role, describes when to use each tool, explicitly forbids inventing information, and explains how to resolve contextual references and short replies ("yes", "ok", "sure").
- **Reduced prompt for small talk**: avoids triggering tool-calling on trivial messages.
- **Enriched prompt for automatic tool-calling**: includes the conversation history and tool descriptions so the LLM can autonomously decide what to invoke.

Key instructions applied: *"Don't make up information"*, *"Use EXCLUSIVELY the data returned by the tool"*, and a fixed instruction to always answer in Spanish (while the indexed data stays in English, untranslated).

## Challenges and design decisions

| Challenge | Solution |
|---|---|
| Mixed languages in responses | Kept the data in English and instructed the model to always answer in Spanish, instead of translating the dataset |
| Inconsistent memory in `AgentState` | Auxiliary global array as a backup of the last interactions |
| Complexity of a dedicated small talk node | Detection integrated into `chatbot_node` via `es_smalltalk()` |
| `top_k=3` insufficient in movie search | Raised to `top_k=10` to improve coverage without generating excessive noise |
| Memory resolved without an explicit tool | Turned into a tool (`consultar_memoria`) for better traceability and control |

## Tests performed

The agent was evaluated with a fixed set of representative queries, using two modes: direct execution (final answer only) and step-by-step streaming execution (nodes, tool calls, and intermediate results). Cases covered:

1. Small talk (greeting)
2. Adding data to short-term memory
3. Simple retrieval over the dataset
4. Synthesis of paper content
5. Search by attribute/metadata (country)
6. Combination of RAG + reasoning + memory
7. Multi-criteria query (country + genre)
8. Immediate conversational memory (repeating the last response)
9. Source citation (anti-hallucination)
10. Conversational closing

Full detail of each test, with the prompts and responses obtained, is documented in the notebook and in the report.

## Interface

An interactive chat interface was built with **Gradio** that preserves the `AgentState` across turns, allowing the full flow (reasoning, tool-calling, and memory) to be tested interactively.

## Possible future improvements

- **Deep agents**: break down complex questions into sub-tasks and plan multiple tool queries before answering.
- **Re-ranking**: add a stage that re-evaluates the retrieved documents and filters only the most relevant ones before generating the response.
- **Long-term memory**: persist user preferences and information across sessions, not just within a single conversation.

## How to run it

The notebook was built and designed to run on **Google Colab** (it uses `google.colab.drive` and `userdata` for secrets).

1. Open `Obligatorio_Taller_IA.ipynb` in Google Colab.
2. Configure the `HF_TOKEN` (Hugging Face) and `PINECONE_API_KEY` (Pinecone) secrets in Colab.
3. Run the cells in order. If the Pinecone indexes were already created previously, the indexing step can be skipped (see the note in the notebook itself).
4. The last section launches a Gradio interface to chat with the agent.

## Tech stack

`LangChain` · `LangGraph` · `Pinecone` · `sentence-transformers` · `Hugging Face` (Qwen3-4B-Instruct) · `Gradio` · `spaCy` · `pandas`

## Data sources

- Movie dataset: [Letterboxd Movies Dataset (Kaggle)](https://www.kaggle.com/)
- 6 academic papers provided by the course