[根目录](/Users/berton/Github/transformers/CLAUDE.md) > [src](/Users/berton/Github/transformers/src/CLAUDE.md) > [transformers](/Users/berton/Github/transformers/src/transformers/CLAUDE.md) > **data**

# Data 模块文档

> 模块路径: `src/transformers/data/`
> 最后更新: 2025-01-20
> 覆盖率: 90%

## 模块职责

Data模块负责Transformers库的数据处理功能，包括：

1. **数据收集器**: 将原始数据转换为模型输入格式
2. **数据处理器**: 标准数据集的处理逻辑
3. **指标计算**: 模型评估指标实现
4. **特征提取**: 数据预处理和特征工程

## 核心组件

### 1. 数据收集器 (`data_collator.py`)
```python
# 基础数据收集器
DataCollator                    # 基础数据收集器类
DefaultDataCollator           # 默认数据收集器
default_data_collator()        # 默认收集器函数

# 特定任务数据收集器
DataCollatorWithPadding       # 带填充的数据收集器
DataCollatorForLanguageModeling  # 语言建模数据收集器
DataCollatorForTokenClassification  # 标记分类数据收集器
DataCollatorForSeq2Seq        # 序列到序列数据收集器
DataCollatorForMultipleChoice  # 多选数据收集器
```

### 2. 数据处理器 (`processors/`)
```python
# 基础处理器
DataProcessor                  # 数据处理器基类
InputExample                   # 输入样例类
InputFeatures                  # 输入特征类

# GLUE任务处理器
glue_processors               # GLUE任务处理器字典
glue_convert_examples_to_features()  # GLUE样例转换
glue_output_modes             # GLUE输出模式
glue_tasks_num_labels         # GLUE任务标签数

# SQuAD任务处理器
SquadV1Processor              # SQuAD v1.0处理器
SquadV2Processor              # SQuAD v2.0处理器
SquadExample                  # SQuAD样例类
SquadFeatures                 # SQuAD特征类
```

### 3. 指标计算 (`metrics/`)
```python
# GLUE指标
glue_compute_metrics()        # GLUE任务指标计算
xnli_compute_metrics()        # XNLI任务指标计算
squad_metrics.py              # SQuAD任务指标
```

## 子模块结构

### data_collator.py
数据收集器的核心实现，负责：

- **动态填充**: 根据批次中最大序列长度进行填充
- **任务特定处理**: 针对不同NLP任务的专门数据处理
- **张量格式转换**: 将数据转换为PyTorch张量
- **批处理优化**: 高效的批量数据处理

#### 关键数据收集器类

1. **DataCollatorForLanguageModeling**
```python
# 掩码语言建模数据收集器
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,           # 启用掩码语言建模
    mlm_probability=0.15  # 掩码概率
)
```

2. **DataCollatorForTokenClassification**
```python
# 标记分类数据收集器
collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100
)
```

### processors/
数据处理器模块，包含标准NLP任务的处理逻辑：

#### GLUE任务支持
- **CoLA**: 语言学可接受性判断
- **SST-2**: 情感分析
- **MRPC**: 语义等价判断
- **STS-B**: 语义相似度
- **QQP**: 问题等价判断
- **MNLI**: 多体裁自然语言推理
- **QNLI**: 问答自然语言推理
- **RTE**: 识别文本蕴含
- **WNLI**: Winograd模式挑战

#### SQuAD任务支持
- **SQuAD v1.0**: 阅读理解数据集
- **SQuAD v2.0**: 包含无答案情况的阅读理解

### datasets/
预定义数据集处理：
- **language_modeling.py**: 语言建模数据集
- **glue.py**: GLUE数据集处理
- **squad.py**: SQuAD数据集处理

## 使用示例

### 1. 基础数据收集器使用
```python
from transformers import DefaultDataCollator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
collator = DefaultDataCollator()

# 准备数据
data = [
    {"text": "Hello world"},
    {"text": "Transformers are great"}
]

# 编码和收集
encoded = tokenizer([d["text"] for d in data], padding=True, return_tensors="pt")
batch = collator(encoded)
```

### 2. 语言建模数据收集器
```python
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
    return_tensors="pt"
)

# 掩码语言建模的批次处理
batch = collator(texts)
```

### 3. GLUE任务处理
```python
from transformers import glue_processors, glue_convert_examples_to_features

# 获取MRPC任务处理器
processor = glue_processors["mrpc"]()
examples = processor.get_train_examples("glue_data/MRPC")

# 转换为特征
features = glue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=128,
    label_list=processor.get_labels(),
    output_mode="classification"
)
```

### 4. 自定义数据收集器
```python
from transformers import DataCollatorWithPadding
from typing import List, Dict, Any

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 自定义处理逻辑
        labels = [feature.pop("labels") for feature in features]

        # 调用父类方法处理其他特征
        batch = super().__call__(features)
        batch["labels"] = torch.stack(labels)

        return batch
```

## 数据处理流程

### 1. 原始数据 → InputExample
```python
example = InputExample(
    guid="train-0",
    text_a="First sentence",
    text_b="Second sentence",  # 可选
    label="1"
)
```

### 2. InputExample → InputFeatures
```python
features = InputFeatures(
    input_ids=[101, 102],
    attention_mask=[1, 1],
    token_type_ids=[0, 0],
    label=1
)
```

### 3. InputFeatures → 批次张量
```python
# 通过DataCollator转换为批次
batch = {
    "input_ids": tensor([[101, 102], [101, 103]]),
    "attention_mask": tensor([[1, 1], [1, 1]]),
    "labels": tensor([1, 0])
}
```

## 性能优化

### 1. 动态填充策略
```python
# 使用动态填充减少内存使用
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest"  # 填充到批次中最长序列
)
```

### 2. 批处理优化
```python
# 预分批处理提高效率
def batch_process(examples, batch_size=32):
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        yield collator(batch)
```

## 测试策略

### 1. 单元测试
- 数据收集器功能测试
- 处理器转换逻辑测试
- 指标计算准确性测试

### 2. 集成测试
- 与模型训练的集成测试
- 不同任务的数据流程测试

### 3. 性能测试
- 大规模数据处理性能
- 内存使用效率测试

## 常见问题 (FAQ)

### Q: 如何选择合适的数据收集器？
A: 根据任务类型选择：
- 语言建模：`DataCollatorForLanguageModeling`
- 标记分类：`DataCollatorForTokenClassification`
- 序列到序列：`DataCollatorForSeq2Seq`
- 通用任务：`DefaultDataCollator`

### Q: 如何处理长度差异很大的序列？
A: 使用动态填充：
```python
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest"
)
```

### Q: 如何添加自定义处理器？
A: 继承DataProcessor基类：
```python
class CustomProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        # 自定义训练数据读取逻辑
        pass
```

## 相关文件清单

### 核心文件
- `__init__.py` - 模块导出定义
- `data_collator.py` - 数据收集器实现
- `metrics/squad_metrics.py` - SQuAD指标计算

### 处理器模块
- `processors/__init__.py` - 处理器导出
- `processors/glue.py` - GLUE任务处理器
- `processors/squad.py` - SQuAD任务处理器
- `processors/utils.py` - 处理器工具函数
- `processors/xnli.py` - XNLI任务处理器

### 数据集模块
- `datasets/__init__.py` - 数据集导出
- `datasets/glue.py` - GLUE数据集处理
- `datasets/language_modeling.py` - 语言建模数据集
- `datasets/squad.py` - SQuAD数据集处理

## 扩展指南

### 1. 添加新数据收集器
```python
class NewTaskDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # 实现新任务的数据收集逻辑
        return super().__call__(processed_features)
```

### 2. 添加新数据集处理器
```python
class NewDatasetProcessor(DataProcessor):
    def get_examples(self, data_dir, split):
        # 实现新数据集的读取逻辑
        return examples
```

## 变更记录 (Changelog)

### 2025-01-20 - 初始分析
- ✨ 创建data模块详细文档
- 🔍 分析数据收集器架构
- 📊 记录处理器使用模式
- 🎯 识别性能优化机会

---

**📊 当前覆盖率**: 90%
**🎯 目标覆盖率**: 98%+
**⏱️ 分析时间**: 2025-01-20