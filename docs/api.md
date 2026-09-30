# API 启动

安装 API 可选依赖：

```powershell
python -m pip install -e ".[api]"
```

使用 Uvicorn 启动：

```powershell
python -m uvicorn src.api.main:create_application --factory --host 0.0.0.0 --port 8000
```

也可以使用项目命令：

```powershell
rag-api
```

接口：

- `GET /health`
- `POST /ask`，请求体为 `{ "question": "..." }`
