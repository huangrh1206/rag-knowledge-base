# Word RAG Knowledge Base

一个面向工程化学习的 Word 文档问答系统。项目读取 `.docx` 文件，按段落感知方式切分文本，调用 embedding 模型建立向量索引，检索相关证据，再让生成模型基于证据回答并返回可验证引用。

## 项目能力

- 读取 `.docx` 文档，并保留源文件和 Word 段落范围。
- 按段落边界优先的策略切分 chunk，降低跨主题拼接。
- 使用余弦相似度进行 Top-K 检索，并支持相似度阈值。
- 使用 evidence policy 判断证据是否足够，不足时拒答。
- 校验回答中的引用编号，拒绝越界引用和无引用回答。
- 支持 NumPy 本地索引和 Qdrant 持久化向量库。
- 使用 index manifest 校验模型、向量维度、chunk 参数和索引数量。
- 记录检索、生成、拒答和校验失败日志，并带有 request ID。
- 提供 CLI：索引、单次问答和 Agent 对话。

## 快速开始

项目要求 Python 3.11 或更高版本。PowerShell 示例：

```powershell
python -m venv ragenv
.\ragenv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

编辑 `.env`，至少设置：

```env
RAG_API_KEY=your-api-key
RAG_BASE_URL=https://api.openai.com/v1
RAG_CHAT_MODEL=gpt-4.1-mini
RAG_EMBEDDING_MODEL=text-embedding-3-small
```

将 Word 文件放入 `documents/`，建立索引并运行问答：

```powershell
python -m src.cli index .\documents
python -m src.cli ask "如何配置定时任务？"
python -m src.cli chat
```

## 配置说明

完整示例见 `.env.example`。

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `RAG_CHUNK_SIZE` | `700` | chunk 的近似字符上限 |
| `RAG_CHUNK_OVERLAP` | `100` | 相邻 chunk 的重叠字符数 |
| `RAG_TOP_K` | `5` | 向量检索返回的最大结果数 |
| `RAG_SIMILARITY_THRESHOLD` | `0.30` | 单条结果过滤阈值 |
| `RAG_EVIDENCE_MIN_SCORE` | `0.50` | 允许生成的最高结果最低分数 |
| `RAG_EVIDENCE_MIN_RESULTS` | `1` | 允许回答所需的最少结果数 |
| `RAG_VECTOR_STORE_BACKEND` | `numpy` | `numpy` 或 `qdrant` |
| `RAG_REQUEST_TIMEOUT` | `30` | API 单次请求超时秒数 |
| `RAG_MAX_RETRIES` | `2` | SDK 可重试错误的最大重试次数 |

Alibaba Cloud 等服务可能限制 embedding 批量大小，建议保持 `RAG_EMBEDDING_BATCH_SIZE=20`。`.env`、文档、索引、日志和评估报告属于本地数据，已通过 `.gitignore` 排除，不应上传到 GitHub。

## 架构

```text
.docx -> paragraphs -> chunks -> embeddings -> NumPy/Qdrant Top-K
      -> evidence policy -> grounded prompt -> cited answer
```

`RAGService.ask()` 是核心应用边界。CLI、未来的 HTTP API 或其他调用方都应通过该服务使用检索、证据判断、生成和引用校验。

## 索引与存储

NumPy 后端默认写入 `storage/index/`：

- `chunks.json`：chunk 文本、来源文件、段落起止位置。
- `embeddings.npy`：与 chunk 顺序一致的向量矩阵。
- `index-manifest.json`：后端、模型、维度、切分参数和数量信息。

Qdrant 后端写入 `storage/qdrant/`，collection 名称由 `RAG_QDRANT_COLLECTION` 指定，同时保存 manifest。加载索引时会校验配置与 manifest 是否兼容；文档、embedding 模型或切分参数变化后，应重新构建索引。

切换 Qdrant：

```env
RAG_VECTOR_STORE_BACKEND=qdrant
RAG_QDRANT_PATH=storage/qdrant
RAG_QDRANT_COLLECTION=rag_chunks
```

Qdrant 混合检索与精排配置：

```env
RAG_VECTOR_STORE_BACKEND=qdrant
RAG_HYBRID_ENABLED=true
RAG_RERANK_ENABLED=true
RAG_RETRIEVAL_CANDIDATE_K=20
RAG_TOP_K=5
RAG_DENSE_WEIGHT=1.0
RAG_BM25_WEIGHT=1.0
RAG_RRF_K=60
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

该模式执行 `Qdrant 向量粗排 + BM25 粗排 -> RRF 融合 -> CrossEncoder 精排 -> Top-K`。
BM25 启动时通过 Qdrant `scroll` 分页读取 collection 中的 chunk payload 并在本地建索引，
因此无需额外维护 `chunks.json` 或进行双写。当前实现适合中小规模知识库；数据量较大或需要多实例服务时，
再将关键词检索替换为 Elasticsearch 等独立倒排索引服务。

然后重新建立索引：

```powershell
python -m src.cli index .\documents
```

## 评估与测试

```powershell
python -m pytest -q
python -m compileall -q src tests evaluation
git diff --check
python -m evaluation.run_retrieval > evaluation\local-report.json
```

评估包括 Recall@1、Recall@5、MRR、answerable acceptance、refusal accuracy 和 keyword coverage。当前小规模 paraphrase 数据集的一次结果为 Recall@1 `0.875`、Recall@5 `1.0`、MRR `0.9167`、keyword coverage `0.7083`、refusal accuracy `0.75`。这些数值用于当前数据集上的回归比较，不代表所有领域数据的效果。测试使用 fake API，不消耗真实 API 额度。

## 日志

CLI 启动时会在 `log/` 下按日期写入 `rag-YYYY-MM-DD.log`。日志包含 request ID、结果数量、检索耗时、生成耗时、引用数量、拒答原因和异常阶段，不记录 API Key、完整问题、答案或 chunk 正文。

## 目录结构

```text
src/                    应用和领域逻辑
tests/                  单元测试和集成测试
evaluation/             评估集、指标和评估脚本
PROJECT_FRAMEWORK.md    面试复盘和项目框架书
```

关键模块包括：`rag_service.py`、`text_splitter.py`、`retriever.py`、`vector_store.py`、`qdrant_vector_store.py`、`store_factory.py`、`evidence_policy.py`、`citation_validator.py`、`generator.py` 和 `cli.py`。

## 当前限制

- 目前只读取 `.docx` 段落，暂不处理 `.doc`、OCR、图片和复杂表格。
- NumPy 是全量扫描，适合学习和小规模知识库；更大规模应使用向量数据库。
- 当前 Agent 主要支持 `search_knowledge_base` 一个工具。
- 相似度分数只能反映语义相近程度，不能单独证明答案正确，因此系统同时使用 evidence policy、引用校验和离线评估。
- Qdrant 检索支持 payload 中的 chunk 元数据读取，可与 BM25、RRF 和 CrossEncoder 精排组成完整混合检索链路。

## Final Retrieval Modes

```text
NumPy + hybrid disabled       dense retrieval
NumPy + hybrid enabled        dense + BM25 -> RRF -> rerank
Qdrant + hybrid disabled      dense retrieval
Qdrant + hybrid enabled       dense + BM25 -> RRF -> rerank
```

Qdrant 混合模式会通过分页 `scroll` 读取 payload 中的全部 chunk，并在进程内建立 BM25 索引，因此当前不需要维护第二份 `chunks.json` 或接入 Elasticsearch。

索引构建目前采用全量重建策略：不会向已有 NumPy 或 Qdrant 索引增量追加，也不会自动去重。文档内容或分块参数发生变化后，请重新执行索引命令。
