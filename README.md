# Installation

## Prerequisite
- Poetry 1.8.3
- `poetry install`

# Run crawler
```bash
poetry run scrapy crawl llama_blog -o data/blogs.json
```

# Run RAG pipeline
- Create and populate `.env` file from `.env.example`
- Main notebook: `notebooks/002-rag-pipeline.ipynb`
