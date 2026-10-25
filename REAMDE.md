# RAG vs CAG on a Growing Knowledge Base

Code and data for the bachelor's thesis Retrieval-Augmented Generation (RAG) vs Cache-Augmented Generation (CAG) by Aroush Irfan, Metropolia University of Applied Sciences, 2026.

The thesis compares two ways of giving a language model access to a knowledge base:

- **RAG (Retrieval-Augmented Generation):** retrieve the 5 most relevant articles for each question and give only those to the model.
- **CAG (Cache-Augmented Generation):** put the whole knowledge base into the model's context and skip retrieval.

Both use the same model (Qwen3-4B) and are tested on 670 questions over a news corpus that grows from 30 to 75 to 369 articles.

## Main result

| Corpus size | RAG correct | CAG correct |
|---|---|---|
| 30 articles | 88.9% | 92.6% |
| 75 articles | 91.9% | 55.6% |
| 369 articles | 92.4% | 12.5% |

CAG is as accurate as RAG while the whole corpus fits in its context, and drops sharply once articles have to be left out. See `figures/` and the thesis for details.

## Repository structure

```
notebooks/
  01_data_collection.ipynb       collect, clean and batch news articles (NewsAPI)
  02_qa_generation.ipynb         generate and verify question-answer pairs (Gemini)
  03_rag_cag_pipelines.ipynb     run RAG and CAG on every batch (Qwen3-4B, GPU)
  04_evaluation.ipynb            grade answers with an LLM judge, retrieval metrics
  05_results_and_figures.ipynb   tables, significance tests, figures, manual check
data/
  articles_metadata.json         title, source, date and URL of the 369 articles
  qa_dataset.json                670 verified question-answer pairs
  pipeline_results.json          1,718 generated answers (RAG + CAG)
  evaluation_results.json        answers with judge verdicts
  spot_check_filled.csv          manual grading of 30 answers (judge validation)
tables/                          result tables (CSV)
figures/                         result figures (PNG)
```

The prompts used for question generation, answering and judging are defined in the config cells of notebooks 02, 03 and 04.

## How to run

1. Open the notebooks in Google Colab and run them in order (01 → 05).
2. Add two Colab secrets (key icon in the left sidebar):
   - `NEWSAPI_KEY`: a key from https://newsapi.org
   - `GEMINI_API_KEY`: a key from https://aistudio.google.com
3. Each notebook mounts Google Drive and works in a project folder (`MyDrive/thesis-rag-cag`). Change the path in the first cell if needed.
4. Notebook 03 needs a GPU runtime (an NVIDIA T4 was used). The other notebooks run on CPU.

Notebooks 02, 03 and 04 save their progress after every item. If a run stops (disconnection or API quota limit), re-run the notebook and it continues where it left off.

## Notes

- **Article texts are not included.** The news articles belong to their publishers, so `data/` only contains their metadata. Running notebook 01 collects articles again, but because NewsAPI only returns recent articles, a new run will produce a different corpus.
- **Hardware limit:** on the free T4 GPU, CAG's context was capped at 5,000 tokens to avoid running out of GPU memory. This cap is part of the results and is discussed in the thesis.
- No API keys are stored in this repository.
