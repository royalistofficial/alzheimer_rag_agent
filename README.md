# RAG-агент: поиск терапевтических мишеней болезни Альцгеймера

Retrieval-Augmented Generation система для автоматического анализа научной литературы и поиска терапевтических мишеней болезни Альцгеймера с цитированием источников.

---

## Архитектура

```
PubMed API → XML Parsing → Text Cleaning → Parent-Child Chunking
                                                    │
                                     Sentence-Transformers (all-MiniLM-L6-v2)
                                                    │
                                              ChromaDB (dense)
                                                    +
                                              BM25 (sparse)
                                                    │
                                  Hybrid Search (RRF Fusion, k=60)
                                                    │
                                Cross-Encoder Re-ranking (ms-marco-MiniLM)
                                                    │
                                  Local LLM (Qwen2-1.5B-Instruct)
                                                    │
                                        Answer + [PMID:X] Citations
```

---

## Структура ноутбука

| Часть | Описание |
|-------|----------|
| **1. Сбор данных** | 15 поисковых запросов → PubMed ESearch/EFetch API → PMC full-text → очистка и фильтрация |
| **2. EDA** | Распределение по годам, длины текстов, TF-IDF, word cloud, co-occurrence мишеней |
| **3. Векторная БД** | Parent-child чанкинг → эмбеддинги children → ChromaDB + BM25 индекс |
| **4. RAG Pipeline** | Hybrid retrieval → Cross-encoder → XML-structured prompt → LLM → Proxy-метрики |
| **5. Интерфейс** | ipywidgets: ввод запроса, настройка параметров, ответ с источниками и метриками |

---

## Быстрый старт

### Требования

- Python 3.10+
- RAM: ≥8 GB (CPU-режим с Qwen2-1.5B)
- GPU (опционально): CUDA-совместимая, ≥4 GB VRAM для 4-bit квантизации

### Установка

```bash
pip install requests matplotlib seaborn wordcloud scikit-learn pandas numpy \
    chromadb sentence-transformers langchain-text-splitters \
    transformers torch accelerate ipywidgets rank_bm25 tiktoken
```

### Запуск

```bash
jupyter notebook alzheimer_rag_agent.ipynb
```

Выполнять ячейки последовательно. Сбор данных с PubMed занимает 5–15 минут (зависит от сети). Загрузка LLM — 1–3 минуты на CPU.

---

## Компоненты

### Сбор данных (Часть 1)

- **Источник**: PubMed + PMC через NCBI E-utilities API
- **Запросы**: 15 тематических запросов (amyloid, tau, BACE1, TREM2, neuroinflammation и др.)
- **Обработка**: дедупликация по PMID, очистка inline-ссылок и спецсимволов, фильтрация по длине (≥100 символов) и релевантности (≥2 ключевых слова)
- **PMC full-text**: извлечение Introduction и Conclusion из открытых полнотекстовых статей
- **Retry-логика**: экспоненциальный backoff при rate-limit (HTTP 429) и сетевых ошибках

### Retrieval (Часть 3–4)

| Компонент | Модель / Метод | Назначение |
|-----------|---------------|------------|
| Embedding | `all-MiniLM-L6-v2` (384d) | Bi-encoder для dense search по children-чанкам |
| Sparse | BM25 Okapi | Лексический поиск точных терминов (BACE1, TREM2) |
| Fusion | Reciprocal Rank Fusion (k=60) | Объединение dense + sparse без калибровки скоров |
| Re-ranking | `ms-marco-MiniLM-L-6-v2` | Cross-encoder переранжирование top-N кандидатов |
| Хранилище | ChromaDB (persistent, cosine) | Векторная БД для children-чанков |

### Чанкинг: Parent-Child

- **Parent** — полная секция статьи (abstract / introduction / conclusion)
- **Child** — мелкий чанк внутри parent (300 символов, overlap 50)
- Индексируются children, при retrieval возвращаются parents (полный контекст)
- Дедупликация по `parent_id` — одна секция не повторяется в результатах

### Генерация (Часть 4)

| Параметр | Значение |
|----------|----------|
| LLM по умолчанию | `Qwen/Qwen2-1.5B-Instruct` |
| Формат промпта | XML-токены: `<CONTEXT>`, `<SOURCE>`, `<QUESTION>`, `<INSTRUCTION>` |
| Декодирование | Greedy (temperature=0, do_sample=False) |
| Repetition penalty | 1.12 |
| Постобработка | Удаление зацикливания, XML-тегов, системных маркеров |

### Метрики качества (Часть 4)

| Метрика | Формула |
|---------|---------|
| Faithfulness | cited_sentences / total_sentences |
| Source Coverage | cited_pmids ∩ source_pmids / source_pmids |
| Citation Accuracy | cited_pmids ∩ source_pmids / cited_pmids |
| Context Similarity | mean(chunk_similarities) |
| Answer Relevance | cosine_sim(embed(query), embed(answer)) |

---

## Каталог LLM

Ноутбук поддерживает выбор модели через интерактивный виджет:

| Модель | Размер | CPU RAM | GPU VRAM (4-bit) | Лицензия |
|--------|--------|---------|-----------------|----------|
| TinyLlama-1.1B-Chat | 1.1B | ~5 GB | ~1.5 GB | Apache 2.0 |
| **Qwen2-1.5B-Instruct** | 1.5B | ~7 GB | ~2 GB | Apache 2.0 |
| Phi-2 | 2.7B | ~11 GB | ~3 GB | MIT |
| Gemma-2-2B-it | 2.6B | ~11 GB | ~3 GB | Gemma License* |
| Mistral-7B-Instruct-v0.3 | 7.2B | ~30 GB | ~5 GB | Apache 2.0 |
| Qwen2-7B-Instruct | 7.6B | ~32 GB | ~5 GB | Apache 2.0 |
| Llama-3.1-8B-Instruct | 8.0B | ~34 GB | ~6 GB | Llama 3.1* |

*Gated-модели требуют HuggingFace token.

---

## Структура файлов

```
.
├── alzheimer_rag_agent.ipynb    # Основной ноутбук
├── README.md
├── data/
│   ├── raw_articles.json        # Сырые статьи с PubMed (генерируется)
│   └── clean_articles.json      # Очищенные и отфильтрованные статьи (генерируется)
└── vectordb/                    # ChromaDB persistent storage (генерируется)
```

Директории `data/` и `vectordb/` создаются автоматически при первом запуске и пересоздаются при повторном.

---

## Ограничения

- **Корпус** — ограничен ~450–700 статьями из PubMed (2010–2026), не покрывает все публикации
- **LLM** — малые модели (1–3B) могут пропускать цитаты или галлюцинировать;
- **Нет full-text для закрытых статей** — PMC full-text доступен только для Open Access
- **Proxy-метрики** — не заменяют human evaluation, но дают ориентир качества