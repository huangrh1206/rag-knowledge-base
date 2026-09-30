# 部署

本地构建并启动 API：

```powershell
docker build -t rag-agent .
docker run --rm -p 8000:8000 --env-file .env rag-agent
```

启动后可以检查：

```powershell
curl http://localhost:8000/health
```

容器不会复制 `.env`、文档、索引或测试文件；配置通过运行时环境变量注入，持久化索引和记忆目录需要按部署环境挂载到容器。CI 会在 push 和 pull request 时安装依赖、运行 pytest 和编译检查。
