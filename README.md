# VLM Demo

基于 FastAPI 的 VLM（视觉大模型）智能路由与降级演示服务。

核心能力：
- **图片复杂度路由** — 通过 CV 分析自动将简单/复杂图片路由到不同成本的模型
- **API 降级兜底** — 当模型调用超时/限流/5xx 时，自动降级到备用模型
- **质量升档** — 当模型返回 JSON 解析失败时，自动升档到更强模型重试

支持 OpenAI GPT-4o 系列和阿里云百炼 Qwen-VL 系列模型。

---

## 项目结构

```
vlm_demo/
├── api/                        # API 层（请求入口 + Pydantic Schema）
│   ├── main.py                 # FastAPI 入口
│   └── schemas.py              # Pydantic 数据模型
│
├── services/                   # 业务编排层（路由决策 + 降级兜底）
│   ├── routing.py              # VLM 图片复杂度路由
│   ├── execution.py            # Fallback 降级会话 + 质量升档封装
│   └── errors.py               # 错误分类
│
├── core/                       # 基础设施层
│   ├── settings.py             # 路由与降级链配置
│   └── llm/                    # LLM 基础设施
│       ├── config.py           # YAML + ${ENV} 配置加载器
│       ├── model_type.py       # ModelType 枚举
│       ├── factory.py          # LLMFactory 模型工厂
│       └── providers/
│           ├── openai.py       # OpenAI Provider
│           └── bailian.py      # 百炼 Provider
│
├── complexity/                 # CV 分析领域逻辑
│   └── analyzer.py             # 图片复杂度分析
│
├── mocks/                      # 测试/演示辅助
│   └── mock_llm.py             # Mock 模型（单元测试用）
│
├── utils/                      # 通用工具
│   └── image_utils.py          # 图片 URL 转换工具
│
├── tests/
│   └── test_demo.py            # 单元测试
│
├── config.yaml                 # LLM Provider 配置（API Key / Base URL）
├── Dockerfile                  # Docker 构建文件
├── docker-compose.yml          # Docker Compose 编排
├── requirements.txt            # Python 依赖
└── README.md                   # 本文档
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目后进入目录
cd vlm_demo

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑环境变量，填入你的真实 API Key：

```env
# OpenAI
OPENAI_API_KEY=sk-your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1

# Bailian (阿里云百炼)
BAILIAN_API_KEY=sk-your-bailian-key
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> `config.yaml` 通过 `${ENV_VAR:-default}` 语法引用环境变量，本身不含敏感信息，可以提交到仓库。

### 3. 启动服务

**本地开发：**

```bash
python api/main.py
# 或
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Docker 部署：**

```bash
docker compose up --build -d
```

### 4. 验证

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## 核心流程

```
POST /evaluate
  → VlmAgentRouter.resolve()
       下载图片 → CV 分析（边缘密度/色彩熵/背景均匀度）
       → 按 composite_score 分档（low / mid / high）
       → 输出 RoutingDecision（含目标模型）
  → run_with_quality_escalate()
       → FallbackSession.ainvoke(prompt, image_urls)
            → llm_factory.get_model(model_type) 获取 ChatOpenAI 实例
            → 构建 vision 消息（text + image_url）
            → ChatOpenAI.ainvoke() → 调用真实 API
            → (失败) classify_invoke_error()
                 → timeout/429/5xx/404 → fallback 降级链
                 → parse_error → 抛出 → quality_escalate 升档
  → 组装 EvaluateResponse 返回
```

---

## API 接口

### `POST /evaluate` — 评测图片

**请求体：**

```json
{
  "agent_id": "generic",
  "image_urls": [
    "https://example.com/image1.jpg",
    "file:///app/uploads/image2.jpg"
  ],
  "role": "generic"
}
```

> `image_urls` 支持 HTTP URL、`file://` 本地路径、绝对路径三种形式。`file://` 和绝对路径会在服务端自动转换为 base64 data URI 传入 Vision API。

**响应：**

```json
{
  "agent_id": "generic",
  "routing": {
    "agent_id": "generic",
    "model_type": "bailian-qwen-vl-plus",
    "tier": "mid",
    "composite_score": 0.52,
    "shadow_mode": false,
    "profiles": { ... }
  },
  "complexity": { ... },
  "result": { "model": "bailian-qwen-vl-plus", "score": 0.95, ... },
  "meta": {
    "requested_model": "bailian-qwen-vl-plus",
    "actual_model": "bailian-qwen-vl-plus",
    "fallback_hops": [],
    "used_fallback": false,
    "escalated": false
  }
}
```

### `GET /health` — 健康检查

### `GET /config` — 查看当前路由与降级配置

---

## 配置说明

### 路由配置（`core/settings.py`）

```python
@dataclass
class AgentRoutingProfile:
    routing_enabled: bool = True
    tier_low: str = "bailian-qwen-vl-turbo"
    tier_mid: str = "bailian-qwen-vl-plus"
    tier_high: str = "bailian-qwen-vl-max"
    min_tier: str = "low"          # 最低档位限制（如 audit 不能低于 mid）
    default_model: str = "bailian-qwen-vl-plus"
    quality_escalate: bool = True  # 是否开启质量升档
```

### 降级链配置（`core/settings.py`）

```python
DEFAULT_FALLBACK_CHAINS = {
    "openai-gpt-4o": ["bailian-qwen-vl-max", "bailian-qwen-vl-plus"],
    "bailian-qwen-vl-max": ["bailian-qwen-vl-plus", "bailian-qwen-vl-turbo"],
    "bailian-qwen-vl-plus": ["bailian-qwen-vl-turbo"],
}
```

### Provider 配置（`config.yaml`）

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
  bailian:
    api_key: ${BAILIAN_API_KEY}
    base_url: ${BAILIAN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
```

---

## 测试

```bash
# 运行全部单元测试
pytest tests/test_demo.py -v

# 运行特定测试
pytest tests/test_demo.py::test_fallback_timeout -v

# 验证工厂加载
python -c "from core.llm.factory import llm_factory; print(llm_factory.list_models())"
```

---

## 部署

### Docker Compose（推荐）

```bash
# 构建并启动
docker compose up --build -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

### 裸机部署

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量（.env 或系统环境变量）

# 3. 启动（生产环境建议用 gunicorn + uvicorn worker）
gunicorn api.main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 -w 2
```

---

## 支持的模型

| 模型类型 | 说明 |
|---|---|
| `openai-gpt-4o` | OpenAI GPT-4o |
| `openai-gpt-4o-mini` | OpenAI GPT-4o Mini |
| `bailian-qwen-vl-max` | 阿里云百炼 Qwen-VL-Max |
| `bailian-qwen-vl-plus` | 阿里云百炼 Qwen-VL-Plus |
| `bailian-qwen-vl-turbo` | 阿里云百炼 Qwen-VL-Turbo |

---

## License

MIT
