# StructBench：面向 Agent 工具调用与结构化输出的轻量级压力评测基准

> **项目定位**：大模型与端侧小模型的“结构化输出与工具调用（Tool-Call）压力测试仪”  
> **核心理念**：聚焦边界故障（Edge Cases）、零成本确定性验证（AST/Pydantic）、自愈能力评测、一键生成可视化雷达图报告  
> **开发预算**：每周 10~15 小时，3 周完成工业级开源交付  
> **核心技术栈**：Python + Asyncio + httpx + Pydantic v2 + Typer (CLI) + Rich (TUI) + Jinja2 (HTML 报告)

---

## 一、 背景与战略转向：为什么选择“垂直微评测基准（Micro-Benchmark）”？

### 1. 求职与升学（保研/申研）的差异化破局点
* **避开同质化重灾区**：简历上写“基于 LangChain 的知识库问答”或“通用 Agent 编排”已严重审美疲劳。
* **切中工业界与学术界真痛点**：无论是大模型团队训后微调（SFT），还是业务团队将大模型替换为低成本小模型（如 Qwen-7B、DeepSeek-Distill、Llama-8B），最头疼的问题都是：**“这个小模型在复杂的 JSON 输出和嵌套工具调用（Function Calling）中到底稳不稳定？遇到死钻牛角尖的边界情况会不会崩？”**
* **极致的“免费额度”友好度**：相比于需要持续在线抗并发的 SaaS 服务，评测基准是**离线批处理运行**的。利用各大平台的免费 API 额度或本地 Ollama，仅需几千 Token 就能跑完一轮 100 道题的标准测试集，产出极具说服力的横向评测报告。

### 2. 为什么不做通用评测（如 MMLU），而做“结构化微基准”？
* **无需昂贵的“裁判模型（LLM-as-a-Judge）”**：评测“写诗好不好”需要调用昂贵的 GPT-4 打分；而评测“JSON 括号有没闭合、类型对不对、工具参数有没幻觉”，直接用 **Python AST（抽象语法树）+ Pydantic Schema 校验**即可实现 **100% 准确率、0 成本、毫秒级判定**。

---

## 二、 项目命名与品牌设计

| 维度 | 内容 |
| :--- | :--- |
| **正式名称** | **StructBench**（主推，直观体现 Structured Output + Benchmark） |
| **备选名称** | **ToolTorture**（工具调用极限拷问） / **SchemaProbe**（Schema 探针） |
| **英文 Slogan** | *Stress-testing LLMs on extreme JSON schemas, nested tool calls, and self-healing recovery.* |
| **中文 Slogan** | 专治大模型“格式崩坏”与“参数幻觉”：面向 Agent 工具调用与极端 Schema 的轻量级评测基准。 |

---

## 三、 核心评测维度与测试集设计（Dataset Design）

项目内置 **80~100 个精心构造的“魔鬼级”边界测试用例（Hard Edge Cases）**，分为四大核心维度：

### 维度 1：极端嵌套与复杂类型约束（Deep Nesting & Strict Typing）
* **测试点**：多层嵌套对象（深度 $\ge 4$）、联合类型（`Union`）、枚举值严格匹配（`Enum`）、正则表达式格式约束（如时间戳、特定编号）。
* **典型报错**：模型漏写右大括号 `}`、枚举值自己瞎编、把数字输出为字符串 `"123"`。

### 维度 2：特殊字符与转义地狱（Escape Hell & Edge Strings）
* **测试点**：要求输出的 JSON 字段中包含**多行代码片段、SQL 语句、正则表达式、双引号套单引号、LaTeX 公式、Markdown 表格**。
* **典型报错**：未正确转义内部双引号 `\"` 或换行符 `\n`，导致 `json.loads()` 直接崩溃。

### 维度 3：抗干扰与防参数幻觉（Anti-Hallucination & Missing Info）
* **测试点**：提供 5 个极其相似的候选工具（Tool Definitions），或者故意在用户 Prompt 中缺失某个必填参数。
* **期望行为**：模型应当拒绝调用并追问缺失参数，或者准确置为 `null`，而不是**自作聪明编造一个假参数（Hallucination）**。

### 维度 4：多轮报错自愈能力（Multi-Turn Self-Healing）✨ *核心亮点*
* **测试点**：不仅仅是测一次通过率（Pass@1）。当模型第一次输出格式错误时，评测框架自动将 Pydantic 的报错信息（`ValidationError`）回传给模型，测试模型能否在 **第 2 轮（Retry@1）修正错误**。
* **工程价值**：这直接反映了该模型在真实 Agent 框架（如 `dsh`、AutoGen）中的容错恢复能力。

---

## 四、 系统架构与评测流水线

```text
[YAML / JSONL 标准测试集 (100 个边界 Case)]
       │
       ▼
┌───────────────────────────────────────────────────────────────────┐
│                    StructBench 核心评测引擎                        │
│                                                                   │
│  ┌──────────────┐    ┌───────────────────┐    ┌────────────────┐  │
│  │ 1. 异步并发池 │───>│ 2. 确定性验证沙箱 │───>│ 3. 自愈重试环  │  │
│  │ (Async Runner)│    │ (JSON AST+Pydantic)│    │ (Error Replay) │  │
│  └──────────────┘    └─────────┬─────────┘    └────────┬───────┘  │
│                                │ 通过/失败归因         │ 修正成功 │
│                                ▼                       ▼          │
│                      [细粒度错误分类器 (Taxonomy)]                │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
         [终端 Rich 实时排行榜]  +  [交互式 HTML 雷达图对比报告]
```

---

## 五、 核心功能模块拆解（MVP 范围）

### 模块 1：声明式测试集加载器（Declarative Task Loader）
* 采用人类易读的 `YAML` 或 `JSONL` 定义测试题，每道题包含：
  * `id` & `category`（所属难度维度）
  * `prompt`（包含干扰项的用户输入）
  * `target_schema`（标准 JSON Schema 或对应的 Pydantic 模型名称）
  * `ground_truth_rules`（值级别校验规则，如某个字段必须等于特定值）

### 模块 2：高并发多模型运行器（Async Model Runner）
* 基于 `httpx` + `asyncio.Semaphore` 实现异步并发请求，支持控制并发度（如 `--concurrency 5`），完美适配免费 API 的速率限制（Rate Limit）。
* 支持同时传入多个模型进行横向 PK（如 `structbench run --models deepseek-chat,qwen-plus,ollama/qwen2.5:7b`）。
* 精确采集系统级指标：**首字延迟（TTFT）**、**每秒生成 Token 数（TPS）**、**總 Token 消耗**。

### 模块 3：多级验证与错误归因引擎（Validator & Error Taxonomy）
不笼统地给出“成功/失败”，而是将错误精确归类为 5 种病灶（面试极佳谈资）：
1. `Markdown_Wrapper_Leak`：要求纯 JSON 却包裹了 ````json ... ```` 或多余的寒暄废话。
2. `Syntax_Truncated`：括号未闭合、转义符错误导致语法解析失败。
3. `Schema_Type_Mismatch`：JSON 语法合法，但字段缺失或数据类型不匹配（如 `int` 写成 `str`）。
4. `Enum_Or_Value_Hallucination`：未遵循枚举约束或编造了原文不存在的参数。
5. `Valid_Pass`：完全合规通过。

### 模块 4：多维评分与可视化报告生成器（Reporter）
* **核心指标计算**：
  * **Pass@1**：首次一次性通过率。
  * **Heal@1**：首次失败后，喂入报错信息二次修正后的总通过率。
  * **Latency P50 / P95**：响应速度分位数。
* **双端输出**：
  1. **终端对比表格**：利用 `Rich` 在命令行直接打印各模型在不同维度的得分排名。
  2. **静态 HTML 评测报告**：基于 `Jinja2` + `ECharts` 生成单文件 `report.html`，包含**多模型能力五维雷达图**、**错误类型分布堆叠柱状图**，以及点击查看具体失败 Case 的 Diff 对比。

---

## 六、 标准工业级工程结构

遵循现代 Python 开源标准（基于 `uv` 构建）：

```text
structbench/
├── pyproject.toml              # 声明依赖与 CLI 入口 `structbench`
├── README.md                   # 含最新主流模型评测天梯图 (Leaderboard)、架构图、快速上手
├── LICENSE                     # MIT 协议
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions 自动运行 ruff 代码规范与 pytest
├── datasets/                   # 核心资产：分类整理的硬核边界测试集
│   ├── 01_deep_nesting.yaml
│   ├── 02_escape_hell.yaml
│   ├── 03_tool_hallucination.yaml
│   └── 04_edge_types.yaml
├── src/
│   └── structbench/
│       ├── __init__.py
│       ├── cli.py              # Typer 命令行 (structbench run / list / report)
│       ├── runner.py           # 异步并发调度、API 适配与重试控制
│       ├── evaluator.py        # AST 语法解析、Pydantic 动态校验与错误归因分类
│       ├── healer.py           # 错误回传与多轮自愈测试逻辑
│       ├── metrics.py          # Pass@1, Heal@1, TTFT, TPS 统计计算
│       └── reporter/
│           ├── console.py      # Rich 终端彩色排行榜输出
│           └── template.html   # 内嵌 ECharts 的可视化雷达图报告模板
└── tests/
    ├── test_evaluator.py       # 单元测试：验证各类奇葩破损 JSON 的归因准确性
    └── test_runner.py          # 单元测试：Mock API 测试异步并发与自愈循环
```

---

## 七、 三周开发与开源落地排期（10~15 小时/周）

| 周期 | 核心目标 | 关键任务清单 | 阶段里程碑 |
| :--- | :--- | :--- | :--- |
| **第 1 周** | **评测内核与首批题库** | 1. 编写 `evaluator.py`：实现 JSON 提取、语法容错解析、动态 Schema 校验与 5 类错误归因<br>2. 手工打磨前 30 道高价值边界测试用例（重点覆盖转义字符与嵌套工具调用）<br>3. 编写单元测试确保验证器本身无 Bug | 输入任意大模型乱码或残缺输出，验证器能毫秒级精准报出错误分类 |
| **第 2 周** | **异步调度与自愈闭环** | 1. 编写 `runner.py`：基于 `httpx` 实现支持并发限流的多模型异步调用器<br>2. 实现 `healer.py`：当首轮校验失败时，自动拼接 `ValidationError` 进行二次请求<br>3. 使用 `Typer` + `Rich` 封装 CLI 与终端进度条、得分表格 | 一行命令 `structbench run -m deepseek-chat` 跑通 30 道题并在终端打印 Pass@1 与 Heal@1 |
| **第 3 周** | **扩充题库与开源宣发** | 1. 利用 AI 辅助+人工校验，将题库扩充至 80~100 道；完成 `template.html` 雷达图报告生成<br>2. **跑出真实天梯榜**：用免费额度实测 4~5 个热门模型（如 DeepSeek-V3, Qwen-2.5-72B/7B, GLM-4-Flash），把评测对比图贴在 README 首页<br>3. 发布至 PyPI 并推送到 GitHub / 知乎 / V2EX | 仓库具备自带流量的“真实模型横评榜单”，支持 `pip install structbench` |

---

## 八、 简历与面试包装指南（成果转化）

这个项目不仅代码本身是亮点，**你用它测出来的结论和数据**更是面试时和面试官深度交流的王牌。

### 简历描述模板
> **StructBench — 面向大模型结构化输出与工具调用的轻量级评测基准（独立开源作者）**  
> *技术栈：Python, Asyncio, httpx, Pydantic v2, AST, Typer, Rich, ECharts, Pytest*
> * **高挑战性微基准构建**：针对 Agent 落地高频崩溃场景，构建包含 **100+ 极端边界用例**（深层嵌套、多语言转义地狱、近似工具防幻觉）的声明式评测集，填补轻量级端侧模型结构化可靠性定量评估空白。
> * **确定性验证与多轮自愈评估**：设计基于 JSON AST 与 Pydantic 动态 Schema 的零成本验证沙箱，构建 5 维细粒度错误归因体系；首创 **Heal@1（错误回传自愈率）** 评测指标，定量刻画模型在 Agent 框架中的容错恢复能力。
> * **高并发引擎与开源影响力**：基于 `Asyncio` 信号量机制实现支持多模型横向对比的异步评测流水线，自动采集 TTFT/TPS 性能分位数并生成交互式雷达图报告；实测对比 6 款主流开源模型并产出评测分析报告，打包发布至 PyPI。

### 面试高频追问与满分回答准备
1. **问：你通过自己做的这个 Benchmark，测出了哪些有趣的结论？（必考题！）**
   * *答*：“我用这套基准横评了 DeepSeek-V3 和几个 7B/14B 尺寸的端侧模型，发现了两个反直觉的现象：第一，小模型在常规 JSON 上表现不错（Pass@1 达 85%），但在包含代码块或双引号转义的 `Escape Hell` 维度会断崖式跌到 40% 以下；第二，有些模型虽然首次输出容易漏字段，但把 Pydantic 报错喂给它后，**自愈率（Heal@1）能飙升到 95% 以上**。这说明在工程落地时，与其花大力气微调小模型，不如在 Agent 运行时加一层轻量级的 Validation Error 回传重试环，性价比最高。”
2. **问：为什么不用大模型来给输出打分（LLM-as-a-Judge），而是自己写规则校验？**
   * *答*：“因为结构化输出和工具调用属于**强确定性任务（Deterministic Tasks）**。如果用 LLM 当裁判，不仅浪费 API 额度、评测延迟高，而且裁判模型自身也会对复杂转义字符产生幻觉。通过 AST 语法解析配合 JSON Schema 校验，不仅能做到零成本、毫秒级判定，还能确保评测结果的 100% 可复现性。”