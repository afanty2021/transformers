# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Transformers数据收集器模块

该模块实现了各种NLP任务的数据收集器(DataCollator)，负责将原始数据样本转换为模型可处理的批次格式。
数据收集器是训练和推理过程中的关键组件，确保数据能够以正确的格式传递给模型。

核心功能：
- 批次数据处理：将多个样本组合成批次
- 动态填充：根据实际需要动态填充序列
- 任务特定处理：针对不同NLP任务的专门数据处理
- 框架兼容：支持PyTorch和NumPy格式转换
- 内存优化：高效的内存使用策略

主要组件：
- DefaultDataCollator: 默认数据收集器
- DataCollatorWithPadding: 带填充功能的数据收集器
- DataCollatorForLanguageModeling: 语言建模数据收集器
- DataCollatorForTokenClassification: 标记分类数据收集器
- DataCollatorForSeq2Seq: 序列到序列数据收集器

使用场景：
- 训练时的批次数据准备
- 推理时的输入格式化
- 动态填充优化内存使用
- 多任务学习的数据处理

设计原则：
- 灵活性：支持多种NLP任务和数据处理策略
- 效率性：优化内存使用和处理速度
- 兼容性：与不同深度学习框架的兼容
- 易用性：提供简单的API和合理的默认配置
"""

import multiprocessing as mp
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Optional, Union

import numpy as np

from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy


# 输入数据类型的类型别名
# 用于表示任意的输入数据格式，可以是字典、张量或自定义数据结构
InputDataClass = Any

"""
数据收集器类型定义

数据收集器是一个函数，接收来自数据集的样本列表，将它们整理成一个批次。
输出是一个包含PyTorch张量或NumPy数组的字典。

功能描述：
1. 接收样本列表：从数据集中获取的一组输入样本
2. 批次整理：将样本组合成模型可处理的批次格式
3. 张量转换：将数据转换为深度学习框架的张量格式
4. 返回字典：返回模型需要的所有输入张量

输入：
- list[InputDataClass]: 样本列表，每个样本可以是任意格式

输出：
- dict[str, Any]: 批次字典，包含模型所需的所有张量

使用示例：
    ```python
    def custom_collator(examples):
        # 自定义数据收集逻辑
        return {
            'input_ids': torch.stack([ex['input_ids'] for ex in examples]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in examples])
        }
    ```
"""
DataCollator = Callable[[list[InputDataClass]], dict[str, Any]]


class DataCollatorMixin:
    """
    数据收集器混入类

    为数据收集器提供框架兼容性和统一接口的基础混入类。
    支持多种深度学习框架的张量格式转换，包括PyTorch和NumPy。

    主要功能：
    - 框架自动检测：根据return_tensors参数自动选择处理方法
    - 统一接口：提供一致的__call__接口
    - 格式转换：支持pt(PyTorch)和np(NumPy)格式
    - 错误处理：对不支持的框架提供明确的错误信息

    使用方法：
        继承此混入类并实现相应的torch_call和numpy_call方法：

        ```python
        class MyDataCollator(DataCollatorMixin):
            def torch_call(self, features):
                # PyTorch格式的处理逻辑
                return torch_batch

            def numpy_call(self, features):
                # NumPy格式的处理逻辑
                return numpy_batch
        ```

    参数说明：
        features: 输入特征列表，每个元素是一个样本的特征
        return_tensors: 返回张量的格式，可选值：
            - "pt": PyTorch张量格式
            - "np": NumPy数组格式
            - None: 使用默认格式(self.return_tensors)
    """

    def __call__(self, features, return_tensors: Optional[str] = None):
        """
        数据收集器的主调用方法

        根据指定的张量格式选择相应的处理方法，并返回处理后的批次数据。

        Args:
            features: 特征列表，包含待处理的多个样本
            return_tensors (Optional[str]): 返回张量的格式
                - "pt": PyTorch张量
                - "np": NumPy数组
                - None: 使用实例的默认格式

        Returns:
            dict[str, Any]: 处理后的批次数据，包含模型所需的所有张量

        Raises:
            ValueError: 当指定的框架格式不被支持时抛出异常

        执行流程：
        1. 确定返回张量格式
        2. 根据格式调用相应的处理方法
        3. 返回处理后的批次数据
        """
        # 如果未指定返回格式，使用实例的默认格式
        if return_tensors is None:
            return_tensors = self.return_tensors

        # 根据返回格式选择相应的处理方法
        if return_tensors == "pt":
            # PyTorch格式处理
            return self.torch_call(features)
        elif return_tensors == "np":
            # NumPy格式处理
            return self.numpy_call(features)
        else:
            # 不支持的格式，抛出错误
            raise ValueError(f"Framework '{return_tensors}' not recognized! "
                           f"Supported formats are: 'pt' (PyTorch), 'np' (NumPy)")


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


def default_data_collator(features: list[InputDataClass], return_tensors="pt") -> dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    if return_tensors == "pt":
        return torch_default_data_collator(features)
    elif return_tensors == "np":
        return numpy_default_data_collator(features)


@dataclass
class DefaultDataCollator(DataCollatorMixin):
    """
    默认数据收集器

    这是一个简单的数据收集器，用于将字典样式的对象整理成批次。
    主要处理具有以下特殊键的数据：

        - `label`: 处理每个对象的单个值（int或float）
        - `label_ids`: 处理每个对象的值列表

    特性说明：
    - 不会进行额外的预处理操作
    - 输入对象的属性名将直接用作模型的对应输入
    - 特别适用于已经预处理好的数据
    - 提供对象接口而非纯函数接口，便于在初始化时设置return_tensors

    使用场景：
    - GLUE任务（General Language Understanding Evaluation）
    - NER任务（Named Entity Recognition）
    - 已经完成预处理的数据
    - 需要简单批次整理的情况

    示例使用：
        ```python
        collator = DefaultDataCollator(return_tensors="pt")

        # 准备数据
        features = [
            {"input_ids": [101, 102], "label": 1},
            {"input_ids": [101, 103], "label": 0}
        ]

        # 批次处理
        batch = collator(features)
        # 结果：{"input_ids": tensor([[101, 102], [101, 103]]), "label": tensor([1, 0])}
        ```

    Args:
        return_tensors (str, optional, defaults to "pt"):
            返回的张量类型。可选值：
            - "pt": PyTorch张量（默认）
            - "np": NumPy数组
    """

    return_tensors: str = "pt"  # 默认返回PyTorch张量格式

    def __call__(self, features: list[dict[str, Any]], return_tensors=None) -> dict[str, Any]:
        """
        调用默认数据收集器处理特征列表

        Args:
            features (list[dict[str, Any]]): 特征列表，每个元素是字典格式的样本
            return_tensors (str, optional): 返回张量的格式，如果为None则使用实例的默认值

        Returns:
            dict[str, Any]: 处理后的批次数据，包含所有输入张量

        处理逻辑：
        1. 如果未指定return_tensors，使用实例的默认值
        2. 调用default_data_collator函数进行实际的批次处理
        3. 返回处理后的批次字典

        特殊处理：
        - label键：转换为张量并适当堆叠
        - label_ids键：处理多标签情况
        - 其他键：直接堆叠成张量
        """
        if return_tensors is None:
            return_tensors = self.return_tensors
        return default_data_collator(features, return_tensors)


def torch_default_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def numpy_default_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], np.ndarray) else first["label"]
        dtype = np.int64 if isinstance(label, int) else np.float32
        batch["labels"] = np.array([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], np.ndarray):
            batch["labels"] = np.stack([f["label_ids"] for f in features])
        else:
            dtype = np.int64 if isinstance(first["label_ids"][0], int) else np.float32
            batch["labels"] = np.array([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, np.ndarray):
                batch[k] = np.stack([f[k] for f in features])
            else:
                batch[k] = np.array([f[k] for f in features])

    return batch


@dataclass
class DataCollatorWithPadding:
    """
    动态填充数据收集器

    这是一个能够动态填充输入序列的数据收集器，是实际应用中最常用的收集器之一。
    根据批次中序列的实际长度进行动态填充，避免不必要的内存浪费。

    🎯 主要功能：
    - 动态填充：根据批次中实际最长序列进行填充
    - 内存优化：避免固定长度的过度填充
    - 多策略支持：支持多种填充策略
    - 硬件优化：支持特定硬件的张量核心优化

    📋 填充策略：

    - `True` 或 `'longest'`（默认）：填充到批次中最长序列的长度
      - 如果只有一个序列，则不进行填充
      - 最常用的策略，内存效率最高

    - `'max_length'`：填充到指定的最大长度
      - 如果未提供max_length参数，则使用模型的最大输入长度
      - 适用于需要固定长度输入的场景

    - `False` 或 `'do_not_pad'`：不进行填充
      - 输出批次中的序列长度可能不同
      - 适用于支持可变长度输入的模型

    🚀 硬件优化（pad_to_multiple_of）：
    - 填充到指定值的倍数
    - 特别适用于NVIDIA硬件的计算能力>=7.0（Volta架构）
    - 启用Tensor Cores的核心优化，提升计算效率
    - 常用值：8、16、32等（取决于模型架构）

    💡 使用示例：
        ```python
        # 基础使用：动态填充到最长序列
        collator = DataCollatorWithPadding(tokenizer)

        # 固定长度填充
        collator = DataCollatorWithPadding(
            tokenizer,
            padding="max_length",
            max_length=128
        )

        # 硬件优化填充
        collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=8  # 填充到8的倍数
        )
        ```

    Args:
        tokenizer (PreTrainedTokenizer or PreTrainedTokenizerFast):
            用于编码数据的分词器，必须是预训练的分词器实例

        padding (bool, str or PaddingStrategy, optional, defaults to True):
            填充策略，控制返回序列的填充方式：
            - True/longest: 填充到批次最长序列
            - max_length: 填充到指定最大长度
            - False/do_not_pad: 不填充

        max_length (int, optional):
            返回列表的最大长度，可选的填充长度
            仅在padding="max_length"时生效

        pad_to_multiple_of (int, optional):
            如果设置，将序列填充到指定值的倍数
            用于NVIDIA Tensor Core优化（计算能力>=7.0）

        return_tensors (str, optional, defaults to "pt"):
            返回的张量类型
            - "pt": PyTorch张量（默认）
            - "np": NumPy数组

    🎨 性能优化建议：
    1. 优先使用动态填充（padding="longest"）节省内存
    2. 在GPU训练时使用pad_to_multiple_of=8
    3. 批量推理时可考虑固定长度填充
    4. 大模型训练时优先使用fast tokenizer
    """

    # 核心属性定义
    tokenizer: PreTrainedTokenizerBase              # 分词器实例，用于文本编码和填充
    padding: Union[bool, str, PaddingStrategy] = True  # 填充策略，默认为动态填充
    max_length: Optional[int] = None               # 最大序列长度，用于固定长度填充
    pad_to_multiple_of: Optional[int] = None       # 填充倍数，用于硬件优化
    return_tensors: str = "pt"                     # 返回张量格式，默认PyTorch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        执行动态填充数据收集

        将输入的特征列表转换为填充后的批次张量，处理不同长度的序列。

        Args:
            features (list[dict[str, Any]]): 特征列表，每个元素包含待处理的样本数据
                典型的特征包括：
                - input_ids: 输入ID序列
                - attention_mask: 注意力掩码
                - token_type_ids: 段落ID（可选）
                - labels: 标签数据（可选）

        Returns:
            dict[str, Any]: 填充后的批次数据，包含：
            - input_ids: 填充后的输入ID张量
            - attention_mask: 填充后的注意力掩码张量
            - 其他字段：根据输入特征动态添加

        处理流程：
        1. 调用分词器的pad方法进行填充
        2. 应用指定的填充策略
        3. 处理硬件优化参数
        4. 转换为指定的张量格式
        5. 返回批次字典

        性能考虑：
        - 使用fast tokenizer提升填充速度
        - 合理设置pad_to_multiple_of优化GPU利用率
        - 避免不必要的固定长度填充
        """
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,                    # 分词器实例
            features,                         # 特征列表
            padding=self.padding,             # 填充策略
            max_length=self.max_length,       # 最大长度
            pad_to_multiple_of=self.pad_to_multiple_of,  # 填充倍数
            return_tensors=self.return_tensors,  # 张量格式
        )

        # 处理标签字段的标准化
        # 将不同格式的标签字段统一为"labels"字段，符合大多数模型的期望格式
        if "label" in batch:
            # 处理单个标签值的场景（如分类任务）
            batch["labels"] = batch["label"]
            del batch["label"]  # 删除原始字段，避免重复
        if "label_ids" in batch:
            # 处理标签序列的场景（如标记分类任务）
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]  # 删除原始字段，避免重复

        return batch


@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", or "pt".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch

    def numpy_call(self, features):
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="np" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = np.array(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]

        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        return batch


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        if not isinstance(examples, torch.Tensor):
            return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer.pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def _numpy_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [np.array(e, dtype=np.int64) for e in examples]

    # Check if padding is necessary.
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return np.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer.pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = np.full(shape=(len(examples), max_length), fill_value=tokenizer.pad_token_id, dtype=examples[0].dtype)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


@dataclass
class DataCollatorForMultipleChoice(DataCollatorMixin):
    """
    Data collator that dynamically pads a batch of nested examples for multiple choice, so that all choices
    of all examples have the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences according to the model's padding side and padding index
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            Pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", or "pt".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]):  # Refactored implementation from the docs.
        import torch

        # Take labels out of the examples beforehand, because they aren't nested.
        label_name = "label" if "label" in examples[0] else "labels"
        labels = [example.pop(label_name) for example in examples]

        batch_size = len(examples)
        num_choices = len(examples[0]["input_ids"])

        # Go from e.g. 2 examples of 2 choices [{input_ids: [[1], [2]]}, {input_ids: [[3], [4]]}]
        # to 4 examples [{input_ids: [1]}, {input_ids: [2]}] + [{input_ids: [3]}, {input_ids: [4]}]
        flat_examples = sum(
            ([{k: v[i] for k, v in example.items()} for i in range(num_choices)] for example in examples), start=[]
        )

        # Pad all choices of all examples as if you're padding any other batch of examples.
        batch = self.tokenizer.pad(
            flat_examples,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Reshape from B*C x L into B x C x L, and add the labels back in.
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.0 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", or "pt".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        whole_word_mask (`bool`, *optional*, defaults to `False`):
            Whether or not to mask whole words instead of individual tokens.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        mask_replace_prob (`float`, *optional*, defaults to 0.8):
            The probability with which masked tokens are replaced by the tokenizer's mask token (e.g., `[MASK]`).
            Defaults to 0.8, meaning 80% of the masked tokens will be replaced with `[MASK]`.
            Only works when `mlm` is set to `True`.
        random_replace_prob (`float`, *optional*, defaults to 0.1):
            The probability with which masked tokens are replaced by random tokens from the tokenizer's vocabulary.
            Defaults to 0.1, meaning 10% of the masked tokens will be replaced with random tokens. The remaining
            masked tokens (1 - mask_replace_prob - random_replace_prob) are left unchanged.
            Only works when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", or "pt".
        seed (`int`, *optional*):
            The seed to use for the random number generator for masking. If not provided, the global RNG will be used.

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    <Example Options and Expectations>

    1. Default Behavior:
        - `mask_replace_prob=0.8`, `random_replace_prob=0.1`.
        - Expect 80% of masked tokens replaced with `[MASK]`, 10% replaced with random tokens, and 10% left unchanged.

    2. All masked tokens replaced by `[MASK]`:
        - `mask_replace_prob=1.0`, `random_replace_prob=0.0`.
        - Expect all masked tokens to be replaced with `[MASK]`. No tokens are left unchanged or replaced with random tokens.

    3. No `[MASK]` replacement, only random tokens:
        - `mask_replace_prob=0.0`, `random_replace_prob=1.0`.
        - Expect all masked tokens to be replaced with random tokens. No `[MASK]` replacements or unchanged tokens.

    4. Balanced replacement:
        - `mask_replace_prob=0.5`, `random_replace_prob=0.4`.
        - Expect 50% of masked tokens replaced with `[MASK]`, 40% replaced with random tokens, and 10% left unchanged.

    Note:
        The sum of `mask_replace_prob` and `random_replace_prob` must not exceed 1. If their sum is less than 1, the
        remaining proportion will consist of masked tokens left unchanged.

    </Tip>
    """

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    whole_word_mask: bool = False
    mlm_probability: Optional[float] = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    seed: Optional[int] = None

    def __post_init__(self):
        if self.mlm:
            if self.tokenizer.mask_token is None:
                raise ValueError(
                    "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                    "You should pass `mlm=False` to train on causal language modeling instead."
                )
            if self.mlm_probability is None or self.mlm_probability < 0 or self.mlm_probability > 1:
                raise ValueError("mlm_probability should be between 0 and 1.")
            self.mlm_probability = float(self.mlm_probability)
        elif self.whole_word_mask:
            raise ValueError(
                "Whole word masking can only be used with mlm=True."
                "If you want to use whole word masking, please set mlm=True."
            )
        if self.mask_replace_prob + self.random_replace_prob > 1:
            raise ValueError("The sum of mask_replace_prob and random_replace_prob should not exceed 1")
        if self.mask_replace_prob < 0 or self.mask_replace_prob > 1:
            raise ValueError("mask_replace_prob should be between 0 and 1.")
        if self.random_replace_prob < 0 or self.random_replace_prob > 1:
            raise ValueError("random_replace_prob should be between 0 and 1.")

        self.mask_replace_prob = float(self.mask_replace_prob)
        self.random_replace_prob = float(self.random_replace_prob)

        if self.whole_word_mask:
            if not self.tokenizer.is_fast:
                warnings.warn(
                    "Whole word masking depends on offset mapping which is only natively available with fast tokenizers.",
                    UserWarning,
                )

            if self.mask_replace_prob < 1:
                warnings.warn(
                    "Random token replacement is not supported with whole word masking.",
                    "Setting mask_replace_prob to 1.",
                )
                self.mask_replace_prob = 1
                self.random_replace_prob = 0

        self.generator = None

    def get_generator(self, seed):
        if self.return_tensors == "pt":
            import torch

            return torch.Generator().manual_seed(seed)
        else:
            return np.random.default_rng(seed)

    def create_rng(self):
        if mp.current_process().name == "MainProcess":
            # If we are in the main process, we create a generator object with the seed
            self.generator = self.get_generator(self.seed)
        else:
            # If we are in a worker process (i.e using multiprocessing), we need to set a unique seed for each
            # worker's generator, generated as the main seed + the worker's ID.
            # (https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading)
            # Only PyTorch DataLoader allows us to access the worker ID, and so we check for this.
            import torch

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                error_string = (
                    "Worker process information is not available for seeding the generator. This may be because",
                    "you are using multiprocessing without using a PyTorch DataLoader. The `seed` parameter can",
                    "only be used when using multiprocessing with a PyTorch DataLoader. Please either use a",
                    "single process or use a PyTorch DataLoader with multiple workers.",
                )
                raise ValueError(error_string)

            self.generator = self.get_generator(self.seed + worker_info.id)

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.

        if self.seed and self.generator is None:
            # If we have a seed, we need to create a generator object. Subsequent calls to this function will use the same generator.
            # If no seed supplied, we will use the global RNG
            self.create_rng()

        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        offset_mapping = batch.pop("offset_mapping", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask, offset_mapping=offset_mapping
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None, offset_mapping: Optional[Any] = None
    ) -> tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]

        if self.whole_word_mask:
            word_ids, no_mask_mask = self._calc_word_ids_and_prob_mask(
                to_numpy(offset_mapping), to_numpy(special_tokens_mask)
            )
            no_mask_mask = torch.tensor(no_mask_mask, dtype=torch.bool)
        else:
            no_mask_mask = (
                special_tokens_mask.bool()
                if isinstance(special_tokens_mask, torch.Tensor)
                else torch.tensor(special_tokens_mask, dtype=torch.bool)
            )

        probability_matrix.masked_fill_(no_mask_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
        if self.whole_word_mask:
            masked_indices = torch.BoolTensor(self._whole_word_mask(word_ids, masked_indices))

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob), generator=self.generator).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        # scaling the random_replace_prob to the remaining probability for example if
        # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
        # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob

        # random_replace_prob% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled), generator=self.generator).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def numpy_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.

        if self.seed and self.generator is None:
            # If we have a seed, we need to create a generator object. Subsequent calls to this function will use the same generator.
            # If no seed supplied, we will use the global RNG
            self.create_rng()

        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="np", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _numpy_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        offset_mapping = batch.pop("offset_mapping", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.numpy_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask, offset_mapping=offset_mapping
            )
        else:
            labels = np.copy(batch["input_ids"])
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def numpy_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
        offset_mapping: Optional[Any] = None,
    ) -> tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]

        if self.whole_word_mask:
            word_ids, no_mask_mask = self._calc_word_ids_and_prob_mask(
                to_numpy(offset_mapping), to_numpy(special_tokens_mask)
            )
        else:
            no_mask_mask = (
                special_tokens_mask.astype(bool)
                if isinstance(special_tokens_mask, np.ndarray)
                else np.array(special_tokens_mask, dtype=bool)
            )

        probability_matrix[no_mask_mask] = 0
        # Numpy doesn't have bernoulli, so we use a binomial with 1 trial
        if self.generator:
            masked_indices = self.generator.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)
        else:
            masked_indices = np.random.binomial(1, probability_matrix, size=probability_matrix.shape).astype(bool)

        if self.whole_word_mask:
            masked_indices = self._whole_word_mask(word_ids, masked_indices)

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        if self.generator:
            indices_replaced = (
                self.generator.binomial(1, self.mask_replace_prob, size=labels.shape).astype(bool) & masked_indices
            )
        else:
            indices_replaced = (
                np.random.binomial(1, self.mask_replace_prob, size=labels.shape).astype(bool) & masked_indices
            )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        # scaling the random_replace_prob to the remaining probability for example if
        # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
        # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob
        if self.generator:
            indices_random = (
                self.generator.binomial(1, random_replace_prob_scaled, size=labels.shape).astype(bool)
                & masked_indices
                & ~indices_replaced
            )
            random_words = self.generator.integers(
                low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
            )
        else:
            indices_random = (
                np.random.binomial(1, random_replace_prob_scaled, size=labels.shape).astype(bool)
                & masked_indices
                & ~indices_replaced
            )
            random_words = np.random.randint(
                low=0, high=len(self.tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64
            )
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    @staticmethod
    def _calc_word_ids_and_prob_mask(
        offsets: np.ndarray[np.ndarray[tuple[int, int]]], special_tokens_mask: np.ndarray[np.ndarray[int]]
    ) -> tuple[np.ndarray[np.ndarray[int]], np.ndarray[np.ndarray[int]]]:
        """
        Map tokens to word ids and create mask of tokens to not mask.
        Tokens that are part of the same word will have the same word id and we will only
        set a mask probability for the first token of each word.
        """

        token_starts = offsets[:, :, 0]
        token_ends = offsets[:, :, 1]

        prev_token_ends = np.roll(token_ends, 1, axis=1)
        prev_token_ends[:, 0] = -1  # First token has no previous token

        prev_token_special = np.roll(special_tokens_mask, 1, axis=1)
        prev_token_special[:, 0] = 0

        # Not special token AND (gap from previous or previous token was special)
        special_tokens_mask = special_tokens_mask.astype(bool)
        is_new_word = (~special_tokens_mask) & ((token_starts != prev_token_ends) | (prev_token_special == 1))

        word_ids = np.cumsum(is_new_word, axis=1)
        word_ids[special_tokens_mask] = -1

        prob_mask = ~is_new_word

        return word_ids, prob_mask

    @staticmethod
    def _whole_word_mask(word_ids: np.ndarray[np.ndarray[int]], mask: Any) -> Any:
        """
        Mask whole words based on word ids and mask.
        """
        mask = to_numpy(mask)

        valid_ids = word_ids != -1

        # Create 3D mask where [batch, token_i, token_j] is True if token_i and token_j are the same word
        same_word = (word_ids[:, :, None] == word_ids[:, None, :]) & valid_ids[:, :, None] & valid_ids[:, None, :]

        # For each token, set True if any token in the same word is masked
        return np.any(same_word & mask[:, None, :], axis=2)


@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DataCollatorForWholeWordMask is deprecated and will be removed in a future version, you can now use "
            "DataCollatorForLanguageModeling with whole_word_mask=True instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)
        self.mlm = True  # Force masked language modeling
        self.whole_word_mask = True  # Force whole word masking


def tolist(x) -> list[Any]:
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return x.tolist()


def to_numpy(x) -> np.ndarray[Any]:
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)


@dataclass
class DataCollatorForSOP(DataCollatorForLanguageModeling):
    """
    Data collator used for sentence order prediction task.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and sentence order prediction
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DataCollatorForSOP is deprecated and will be removed in a future version, you can now use "
            "DataCollatorForLanguageModeling instead.",
            FutureWarning,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [example["input_ids"] for example in examples]
        input_ids = _torch_collate_batch(input_ids, self.tokenizer)
        input_ids, labels, attention_mask = self.mask_tokens(input_ids)

        token_type_ids = [example["token_type_ids"] for example in examples]
        # size of segment_ids varied because randomness, padding zero to the end as the original implementation
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        sop_label_list = [example["sentence_order_label"] for example in examples]
        sentence_order_label = torch.stack(sop_label_list)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sentence_order_label": sentence_order_label,
        }

    def mask_tokens(self, inputs: Any) -> tuple[Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels/attention_mask for masked language modeling: 80% MASK, 10% random, 10%
        original. N-gram not applied yet.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # probability be `1` (masked), however in albert model attention mask `0` means masked, revert the value
        attention_mask = (~masked_indices).float()
        if self.tokenizer.pad_token is not None:
            attention_padding_mask = labels.eq(self.tokenizer.pad_token_id)
            attention_mask.masked_fill_(attention_padding_mask, value=1.0)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens, -100 is default for CE compute

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, attention_mask


@dataclass
class DataCollatorForPermutationLanguageModeling(DataCollatorMixin):
    """
    Data collator used for permutation language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for permutation language modeling with procedures specific to XLNet
    """

    tokenizer: PreTrainedTokenizerBase
    plm_probability: float = 1 / 6
    max_span_length: int = 5  # maximum length of a span of masked tokens
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        batch = _torch_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.torch_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    def numpy_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        if isinstance(examples[0], Mapping):
            examples = [e["input_ids"] for e in examples]
        batch = _numpy_collate_batch(examples, self.tokenizer)
        inputs, perm_mask, target_mapping, labels = self.numpy_mask_tokens(batch)
        return {"input_ids": inputs, "perm_mask": perm_mask, "target_mapping": target_mapping, "labels": labels}

    def torch_mask_tokens(self, inputs: Any) -> tuple[Any, Any, Any, Any]:
        """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

            0. Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            1. Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
            2. Reserve a context of length `context_length = span_length / plm_probability` to surround span to be
               masked
            3. Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length -
               span_length]` and mask tokens `start_index:start_index + span_length`
            4. Set `cur_len = cur_len + context_length`. If `cur_len < max_len` (i.e. there are tokens remaining in the
               sequence to be processed), repeat from Step 1.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling."
                " Please add a mask token if you want to use this tokenizer."
            )

        if inputs.size(1) % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see"
                " relevant comments in source code for details."
            )

        labels = inputs.clone()
        # Creating the mask and target_mapping tensors
        masked_indices = torch.full(labels.shape, 0, dtype=torch.bool)
        target_mapping = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)

        for i in range(labels.size(0)):
            # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            cur_len = 0
            max_len = labels.size(1)

            while cur_len < max_len:
                # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
                span_length = torch.randint(1, self.max_span_length + 1, (1,)).item()
                # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
                context_length = int(span_length / self.plm_probability)
                # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
                start_index = cur_len + torch.randint(context_length - span_length + 1, (1,)).item()
                masked_indices[i, start_index : start_index + span_length] = 1
                # Set `cur_len = cur_len + context_length`
                cur_len += context_length

            # Since we're replacing non-masked tokens with -100 in the labels tensor instead of skipping them altogether,
            # the i-th predict corresponds to the i-th token.
            target_mapping[i] = torch.eye(labels.size(1))

        special_tokens_mask = torch.tensor(
            [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
            dtype=torch.bool,
        )
        masked_indices.masked_fill_(special_tokens_mask, value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            masked_indices.masked_fill_(padding_mask, value=0.0)

        # Mask indicating non-functional tokens, where functional tokens are [SEP], [CLS], padding, etc.
        non_func_mask = ~(padding_mask | special_tokens_mask)

        inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        perm_mask = torch.zeros((labels.size(0), labels.size(1), labels.size(1)), dtype=torch.float32)

        for i in range(labels.size(0)):
            # Generate permutation indices i.e. sample a random factorisation order for the sequence. This will
            # determine which tokens a given token can attend to (encoded in `perm_mask`).
            # Note: Length of token sequence being permuted has to be less than or equal to reused sequence length
            # (see documentation for `mems`), otherwise information may leak through due to reuse. In this implementation,
            # we assume that reused length is half of sequence length and permutation length is equal to reused length.
            # This requires that the sequence length be even.

            # Create a linear factorisation order
            perm_index = torch.arange(labels.size(1))
            # Split this into two halves, assuming that half the sequence is reused each time
            perm_index = perm_index.reshape((-1, labels.size(1) // 2)).transpose(0, 1)
            # Permute the two halves such that they do not cross over
            perm_index = perm_index[torch.randperm(labels.size(1) // 2)]
            # Flatten this out into the desired permuted factorisation order
            perm_index = torch.flatten(perm_index.transpose(0, 1))
            # Set the permutation indices of non-masked (non-functional) tokens to the
            # smallest index (-1) so that:
            # (1) They can be seen by all other positions
            # (2) They cannot see masked positions, so there won't be information leak
            perm_index.masked_fill_(~masked_indices[i] & non_func_mask[i], -1)
            # The logic for whether the i-th token can attend on the j-th token based on the factorisation order:
            # 0 (can attend): If perm_index[i] > perm_index[j] or j is neither masked nor a functional token
            # 1 (cannot attend): If perm_index[i] <= perm_index[j] and j is either masked or a functional token
            perm_mask[i] = (
                perm_index.reshape((labels.size(1), 1)) <= perm_index.reshape((1, labels.size(1)))
            ) & masked_indices[i]

        return inputs.long(), perm_mask, target_mapping, labels.long()

    def numpy_mask_tokens(self, inputs: Any) -> tuple[Any, Any, Any, Any]:
        """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

            0. Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            1. Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
            2. Reserve a context of length `context_length = span_length / plm_probability` to surround span to be
               masked
            3. Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length -
               span_length]` and mask tokens `start_index:start_index + span_length`
            4. Set `cur_len = cur_len + context_length`. If `cur_len < max_len` (i.e. there are tokens remaining in the
               sequence to be processed), repeat from Step 1.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for permutation language modeling."
                " Please add a mask token if you want to use this tokenizer."
            )

        if inputs.shape[1] % 2 != 0:
            raise ValueError(
                "This collator requires that sequence lengths be even to create a leakage-free perm_mask. Please see"
                " relevant comments in source code for details."
            )

        labels = np.copy(inputs)
        # Creating the mask and target_mapping tensors
        masked_indices = np.full(labels.shape, 0, dtype=bool)
        target_mapping = np.zeros((labels.shape[0], labels.shape[1], labels.shape[1]), dtype=np.float32)

        for i in range(labels.shape[0]):
            # Start from the beginning of the sequence by setting `cur_len = 0` (number of tokens processed so far).
            cur_len = 0
            max_len = labels.shape[1]

            while cur_len < max_len:
                # Sample a `span_length` from the interval `[1, max_span_length]` (length of span of tokens to be masked)
                span_length = randint(1, self.max_span_length + 1)
                # Reserve a context of length `context_length = span_length / plm_probability` to surround the span to be masked
                context_length = int(span_length / self.plm_probability)
                # Sample a starting point `start_index` from the interval `[cur_len, cur_len + context_length - span_length]` and mask tokens `start_index:start_index + span_length`
                start_index = cur_len + randint(0, context_length - span_length + 1)
                masked_indices[i, start_index : start_index + span_length] = 1
                # Set `cur_len = cur_len + context_length`
                cur_len += context_length

            # Since we're replacing non-masked tokens with -100 in the labels tensor instead of skipping them altogether,
            # the i-th predict corresponds to the i-th token.
            target_mapping[i] = np.eye(labels.shape[1])

        special_tokens_mask = np.array(
            [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()],
            dtype=bool,
        )
        masked_indices[special_tokens_mask] = 0
        if self.tokenizer.pad_token is not None:
            padding_mask = labels == self.tokenizer.pad_token_id
            masked_indices[padding_mask] = 0.0

        # Mask indicating non-functional tokens, where functional tokens are [SEP], [CLS], padding, etc.
        non_func_mask = ~(padding_mask | special_tokens_mask)

        inputs[masked_indices] = self.tokenizer.mask_token_id
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        perm_mask = np.zeros((labels.shape[0], labels.shape[1], labels.shape[1]), dtype=np.float32)

        for i in range(labels.shape[0]):
            # Generate permutation indices i.e. sample a random factorisation order for the sequence. This will
            # determine which tokens a given token can attend to (encoded in `perm_mask`).
            # Note: Length of token sequence being permuted has to be less than or equal to reused sequence length
            # (see documentation for `mems`), otherwise information may leak through due to reuse. In this implementation,
            # we assume that reused length is half of sequence length and permutation length is equal to reused length.
            # This requires that the sequence length be even.

            # Create a linear factorisation order
            perm_index = np.arange(labels.shape[1])
            # Split this into two halves, assuming that half the sequence is reused each time
            perm_index = perm_index.reshape((-1, labels.shape[1] // 2)).T
            # Permute the two halves such that they do not cross over
            np.random.shuffle(perm_index)
            # Flatten this out into the desired permuted factorisation order
            perm_index = perm_index.T.flatten()
            # Set the permutation indices of non-masked (non-functional) tokens to the
            # smallest index (-1) so that:
            # (1) They can be seen by all other positions
            # (2) They cannot see masked positions, so there won't be information leak
            perm_index[~masked_indices[i] & non_func_mask[i]] = -1
            # The logic for whether the i-th token can attend on the j-th token based on the factorisation order:
            # 0 (can attend): If perm_index[i] > perm_index[j] or j is neither masked nor a functional token
            # 1 (cannot attend): If perm_index[i] <= perm_index[j] and j is either masked or a functional token
            perm_mask[i] = (
                perm_index.reshape((labels.shape[1], 1)) <= perm_index.reshape((1, labels.shape[1]))
            ) & masked_indices[i]

        return inputs.astype(np.int64), perm_mask, target_mapping, labels.astype(np.int64)


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:

    - concatenates the entire mini batch into single long sequence of shape [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids` by default
    - optionally returns the kwargs contained in FlashAttentionKwargs
    - optionally returns seq_idx indicating which sequence each token belongs to

    <Tip warning={true}>

    Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence.
    Make sure your attention computation is able to handle it!

    </Tip>
    """

    def __init__(
        self,
        *args,
        return_position_ids=True,
        separator_id=-100,
        return_flash_attn_kwargs=False,
        return_seq_idx=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id
        self.return_flash_attn_kwargs = return_flash_attn_kwargs
        self.return_seq_idx = return_seq_idx
        self._int_64_keys = {"labels", "position_ids", "input_ids"}
        self._batch_dim_keys = {"labels", "position_ids", "input_ids", "seq_idx"}
        self._py_int_keys = {"max_length_q", "max_length_k"}

    def __call__(self, features, return_tensors=None, separator_id=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id
        is_labels_provided = "labels" in features[0]
        batch = {"input_ids": [], "labels": []}
        if self.return_position_ids:
            batch.update({"position_ids": []})
        if self.return_seq_idx:
            batch.update({"seq_idx": []})
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        for seq_idx, sample in enumerate(features):
            input_ids = sample["input_ids"]
            batch["input_ids"] += input_ids
            if is_labels_provided:
                batch["labels"] += [separator_id] + sample["labels"][1:]
            else:
                batch["labels"] += [separator_id] + input_ids[1:]
            if self.return_position_ids:
                batch["position_ids"] += list(range(len(input_ids)))
            if self.return_seq_idx:
                batch["seq_idx"] += [seq_idx for _ in range(len(input_ids))]
            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))

        if self.return_flash_attn_kwargs:
            batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
            batch["max_length_q"] = batch["max_length_k"] = max_length

        # FlashAttentionKwargs and seq_idx are expected to be int32s.
        if return_tensors == "pt":
            import torch

            data_cls = torch.tensor
            dtype_64 = torch.int64
            dtype_32 = torch.int32
        elif return_tensors == "np":
            data_cls = np.array
            dtype_64 = np.int64
            dtype_32 = np.int32
        else:
            raise ValueError(f'return_tensors must be one of ("pt", "np"), {return_tensors=} not supported')

        for k, v in batch.items():
            if k in self._batch_dim_keys:
                v = [v]
            # Flash attention max_len_{q,k} are python ints
            if k not in self._py_int_keys:
                batch[k] = data_cls(v, dtype=dtype_64 if k in self._int_64_keys else dtype_32)

        return batch
