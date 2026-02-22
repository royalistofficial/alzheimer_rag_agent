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
                                        или MMR (с вычислением эмбеддингов)
                                                    │
                                  Local LLM (Qwen2-1.5B-Instruct)
                                                    │
                                        Answer + [PMID:X] Citations
```

---

## Структура ноутбука

| Часть | Описание |
|-------|----------|
| **1. Сбор данных** | 15 поисковых запросов, `MAX_PER_QUERY=50` → PubMed ESearch/EFetch API → PMC full-text → очистка и фильтрация по длине (≥100 символов) |
| **2. EDA** | Распределение по годам, длины текстов, TF-IDF, word cloud, co-occurrence матрица мишеней |
| **3. Векторная БД** | Parent-child чанкинг → эмбеддинги children → ChromaDB + BM25 индекс |
| **4. RAG Pipeline** | Hybrid retrieval → Cross-encoder / MMR → XML-structured prompt → LLM → Proxy-метрики |
| **5. Интерфейс** | ipywidgets: ввод запроса, настройка параметров, ответ с источниками и метриками |
| **6. Теор. вопросы** | Модальности данных, мультимодальный RAG, обоснование выбора моделей |

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
    transformers torch accelerate ipywidgets rank-bm25 tiktoken
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
- **Запросы**: 15 тематических запросов (amyloid, tau, BACE1, TREM2, neuroinflammation и др.), до 50 статей на запрос
- **Период**: 2010–2026
- **Обработка**: дедупликация по PMID, очистка inline-ссылок (`[1,2]`, `(Smith et al.)`), URL, email и спецсимволов через `unicodedata.normalize`, фильтрация по длине абстракта (≥100 символов)
- **PMC full-text**: извлечение Introduction и Conclusion из открытых полнотекстовых статей
- **Retry-логика**: экспоненциальный backoff при rate-limit (HTTP 429) и сетевых ошибках

### Retrieval (Части 3–4)

| Компонент | Модель / Метод | Назначение |
|-----------|---------------|------------|
| Embedding | `all-MiniLM-L6-v2` (384d, `normalize_embeddings=True`) | Bi-encoder для dense search по children-чанкам |
| Sparse | BM25 Okapi (токены ≥3 символов, включая греческие буквы) | Лексический поиск точных терминов (BACE1, TREM2) |
| Fusion | Reciprocal Rank Fusion (`k=60`) | Объединение dense + sparse без калибровки скоров |
| Re-ranking | `ms-marco-MiniLM-L-6-v2` (`max_length=512`) | Cross-encoder переранжирование top-N кандидатов |
| MMR | Maximal Marginal Relevance (`diversity=0.5`) | Fallback при отключённом re-ranking; эмбеддинги кандидатов вычисляются через `encoder.encode()` |
| Хранилище | ChromaDB (`PersistentClient`, HNSW, cosine space) | Векторная БД для children-чанков |
| Фильтрация | `SIMILARITY_THRESHOLD=0.3` + min-max нормализация | Отсечение нерелевантных кандидатов |
| Fallback | Промоция top-3 children с извлечением метаданных из `meta` | Если после фильтрации не осталось кандидатов |

### Чанкинг: Parent-Child

- **Parent** — полная секция статьи (abstract / introduction / conclusion)
- **Child** — мелкий чанк внутри parent (`CHILD_CHUNK_SIZE=300`, `CHILD_CHUNK_OVERLAP=50`, `MIN_CHUNK_LENGTH=40`)
- **Splitter**: `RecursiveCharacterTextSplitter` с разделителями `['. ', '; ', ', ', ' ', '']`
- Индексируются только children, при retrieval возвращаются parents (полный контекст)
- Дедупликация по `parent_id` — одна секция не повторяется в результатах

### Генерация (Часть 4)

| Параметр | Значение |
|----------|----------|
| LLM по умолчанию | `Qwen/Qwen2-1.5B-Instruct` |
| Формат промпта | XML-токены: `<CONTEXT>`, `<SOURCE pmid="..." section="..." year="...">`, `<QUESTION>`, `<INSTRUCTION>` |
| Декодирование | Условное: `TEMPERATURE > 0` → sampling (`temperature=0.3`, `top_p=0.9`); `TEMPERATURE = 0` → greedy (`do_sample=False`) |
| Repetition penalty | 1.12 |
| Chat template | `apply_chat_template` для моделей с поддержкой (Qwen, Llama, Mistral); plain prompt `### System / ### User / ### Answer` для остальных (Phi-2) |
| Подсчёт токенов | Токенизатор загруженной модели; `tiktoken cl100k_base` как fallback |
| Постобработка | Удаление зацикливания (>2 одинаковых строки), XML-тегов, системных маркеров |

### System Prompt

7 обязательных правил: SOURCE-ONLY (только из контекста), CITATION FORMAT (`[PMID:X]`), INSUFFICIENT DATA (отказ при нехватке данных), LANGUAGE (язык ответа = язык вопроса), CONCISENESS (≤250 слов), CONFLICT RESOLUTION (оба источника при противоречии), NO SPECULATION. Включает OUTPUT VALIDATION checklist для самопроверки.

### Метрики качества (Часть 4)

| Метрика | Формула | Что измеряет |
|---------|---------|-------------|
| Faithfulness | cited_sentences / total_sentences | Доля предложений с цитатой `[PMID:X]` |
| Source Coverage | cited_pmids ∩ source_pmids / source_pmids | Какую долю найденных источников модель процитировала |
| Citation Accuracy | cited_pmids ∩ source_pmids / cited_pmids | Все ли цитаты ссылаются на реальные источники |
| Context Similarity | mean(retrieval_scores) | Средний retrieval score (cosine sim / RRF) чанков |
| Answer Relevance | cosine_sim(embed(query), embed(answer)) | Семантическая близость вопроса и ответа |

---

## Каталог LLM

Ноутбук поддерживает выбор модели через интерактивный виджет:

| Модель | Размер | CPU RAM | GPU VRAM (4-bit) |
|--------|--------|---------|------------------|
| TinyLlama-1.1B-Chat | 1.1B | ~5 GB | ~1.5 GB |
| **Qwen2-1.5B-Instruct** | 1.5B | ~7 GB | ~2 GB |
| Phi-2 | 2.7B | ~11 GB | ~3 GB | 
| Gemma-2-2B-it | 2.6B | ~11 GB | ~3 GB | 
| Mistral-7B-Instruct-v0.3 | 7.2B | ~30 GB | ~5 GB | 
| Qwen2-7B-Instruct | 7.6B | ~32 GB | ~5 GB |
| Llama-3.1-8B-Instruct | 8.0B | ~34 GB | ~6 GB | |

Поддерживаемые режимы квантизации: FP16, 4-bit NF4 (bitsandbytes), 8-bit, FP32 (CPU).

---

## Тестовые вопросы

Три обязательных вопроса по заданию:

1. *"What are potential targets for Alzheimer's disease treatment?"*
2. *"Are the targets druggable with small molecules, biologics, or other modalities?"*
3. *"What additional studies are needed to advance these targets?"*

Для каждого вопроса замеряется время retrieval и generation, вычисляются proxy-метрики.

---

## Интерфейс (Часть 5)

Интерактивный интерфейс на `ipywidgets`:

- Ввод произвольного вопроса или выбор из примеров
- Настройка: Top-K (1–15), max tokens (64–2048), hybrid search вкл/выкл, re-ranking вкл/выкл
- Ответ с кликабельными ссылками `[PMID:XXXXX]` → PubMed
- Список источников с оценками релевантности
- Proxy-метрики качества для каждого ответа

---

## Структура файлов

```
.
├── alzheimer_rag_agent.ipynb    # Основной ноутбук (105 ячеек)
├── testovoye zadaniye.md        # Теоретические вопросы
├── README.md
├── data/                        # Генерируется при запуске
│   ├── raw_articles.json        # Сырые статьи с PubMed
│   └── clean_articles.json      # Очищенные и отфильтрованные статьи
└── vectordb/                    # ChromaDB persistent storage (генерируется)
```

Директории `data/` и `vectordb/` создаются автоматически при первом запуске и пересоздаются при повторном.

---

## Ограничения

- **Корпус** — ограничен статьями из PubMed (2010–2026, до 750 кандидатов до дедупликации), не покрывает все публикации
- **LLM** — малые модели (1–3B) могут пропускать цитаты или галлюцинировать; 7B+ дают значительно лучшее качество
- **Нет full-text для закрытых статей** — PMC full-text доступен только для Open Access
- **Proxy-метрики** — не заменяют human evaluation, но дают ориентир качества
- **Embedding модель** — `all-MiniLM-L6-v2` — general-purpose; доменная модель (PubMedBERT, BioSentVec) могла бы улучшить retrieval
