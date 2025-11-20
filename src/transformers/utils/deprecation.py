# Copyright 2024 The HuggingFace Team. All rights reserved.
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
Transformers API弃用处理模块

该模块提供了一套完整的API弃用处理机制，用于优雅地管理函数参数的废弃和迁移。
支持参数重命名、版本控制、警告提示和错误抛出等多种弃用策略。

主要功能：
- 自动检测已弃用的关键字参数
- 支持参数重命名和自动替换
- 基于版本的弃用策略控制
- 与torch.compile兼容的警告系统
- 灵活的错误处理和通知机制

使用场景：
- API重构时的向后兼容性维护
- 函数参数名的标准化和统一
- 渐进式的API改进和迁移
- 版本发布时的废弃管理

设计原则：
- 保持向后兼容性
- 提供清晰的迁移指导
- 支持灵活的弃用策略
- 确保编译器兼容性
"""
import inspect
import warnings
from functools import wraps

import packaging.version

from .. import __version__
from . import ExplicitEnum, is_torch_available, is_torchdynamo_compiling


# 确保在torch.compile编译环境下能正常处理弃用逻辑
# 当函数可能被PyTorch编译器优化时，仍需要能处理弃用的关键字参数
if is_torch_available():
    import torch  # noqa: F401


class Action(ExplicitEnum):
    """
    弃用处理动作枚举类

    定义了弃用参数处理的不同动作类型，用于控制对已弃用参数的处理方式。
    这些动作决定了是忽略、警告还是抛出错误。

    枚举值：
    - NONE: 不进行任何处理，静默忽略
    - NOTIFY: 发出警告通知用户
    - NOTIFY_ALWAYS: 总是发出警告，即使在弃用版本之后
    - RAISE: 抛出异常，阻止函数继续执行
    """
    NONE = "none"            # 不处理，静默忽略
    NOTIFY = "notify"        # 发出弃用警告
    NOTIFY_ALWAYS = "notify_always"  # 总是发出警告（跨版本）
    RAISE = "raise"          # 抛出异常，强制用户更新代码


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: str | None = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: str | None = None,
):
    """
    关键字参数弃用装饰器

    这是一个用于处理函数关键字参数弃用的装饰器，可以自动检测、通知和处理已弃用的参数。
    该装饰器与torch.compile兼容，不会导致图编译中断（但在编译时不会显示警告）。

    主要功能：
    - 自动检测已弃用的关键字参数使用
    - 支持参数重命名的自动替换
    - 基于版本的灵活弃用策略
    - 提供清晰的迁移指导信息

    弃用策略说明：
    - 默认情况下，仅在当前版本小于弃用版本时显示警告
    - 可以通过参数控制在弃用版本之后的行为（继续警告或抛出错误）
    - 支持同时设置新旧参数时的处理策略

    Args:
        old_name (str): 已弃用的关键字参数名称
        version (str): 弃用生效的版本号（格式如"4.20.0"）
        new_name (Optional[str], optional): 新的参数名称。如果提供，会自动替换旧参数。默认为None。
        warn_if_greater_or_equal_version (bool, optional): 是否在当前版本>=弃用版本时仍显示警告。默认为False。
        raise_if_greater_or_equal_version (bool, optional): 是否在当前版本>=弃用版本时抛出异常。默认为False。
        raise_if_both_names (bool, optional): 是否在新旧参数同时提供时抛出异常。默认为False。
        additional_message (Optional[str], optional): 附加的提示信息，会追加到默认警告信息后。默认为None。

    Raises:
        ValueError: 当设置了raise_if_greater_or_equal_version且版本已过期时，或设置了raise_if_both_names且同时提供新旧参数时

    Returns:
        Callable: 装饰后的函数，会根据配置处理弃用参数

    使用示例：

    1. 参数重命名示例：
    ```python
    @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="6.0.0")
    def my_function(do_reduce_labels):
        print(do_reduce_labels)

    my_function(reduce_labels=True)  # 显示警告并使用do_reduce_labels=True
    ```

    2. 直接移除参数示例：
    ```python
    @deprecate_kwarg("max_size", version="6.0.0")
    def my_function(max_size):
        print(max_size)

    my_function(max_size=1333)  # 显示弃用警告
    ```

    3. 强制错误示例：
    ```python
    @deprecate_kwarg("old_param", version="5.0.0", raise_if_greater_or_equal_version=True)
    def my_function():
        pass

    # 在5.0.0版本后调用会抛出异常
    my_function(old_param=True)  # 抛出ValueError
    ```
    """

    # 版本解析和比较
    # 解析弃用版本和当前版本，用于后续的版本控制逻辑
    deprecated_version = packaging.version.parse(version)  # 解析指定的弃用版本
    current_version = packaging.version.parse(__version__)  # 解析当前transformers版本

    # 判断当前版本是否已经达到或超过弃用版本
    is_greater_or_equal_version = current_version >= deprecated_version

    # 根据版本状态构建不同的版本提示信息
    # 已过期版本：使用"从版本X开始移除"的语气
    # 未过期版本：使用"将在版本X移除"的语气
    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        """
        装饰器内层包装函数

        该函数接收被装饰的函数，并返回一个包装后的函数。
        包装函数会检查和处理弃用的关键字参数。

        主要逻辑：
        1. 分析函数签名，识别实例方法和类方法
        2. 构建完整的函数名（包含类名）用于警告信息
        3. 根据参数使用情况决定采取的弃用处理动作
        4. 执行相应的警告或错误处理
        """
        # 分析函数签名以获取更好的警告信息
        sig = inspect.signature(func)  # 获取函数签名
        function_named_args = set(sig.parameters.keys())  # 获取所有命名参数
        is_instance_method = "self" in function_named_args  # 判断是否为实例方法
        is_class_method = "cls" in function_named_args      # 判断是否为类方法

        @wraps(func)  # 保持原函数的元数据
        def wrapped_func(*args, **kwargs):
            """
            实际的包装函数，处理弃用参数检查和替换逻辑

            执行流程：
            1. 构建完整的函数名用于警告信息
            2. 检查弃用参数的使用情况
            3. 根据配置决定处理策略（警告/错误/忽略）
            4. 执行参数替换并调用原函数
            """
            # 构建包含类信息的完整函数名，用于更清晰的警告信息
            func_name = func.__name__
            if is_instance_method:
                # 实例方法：获取实例的类名
                func_name = f"{args[0].__class__.__name__}.{func_name}"
            elif is_class_method:
                # 类方法：获取类的名称
                func_name = f"{args[0].__name__}.{func_name}"

            # 初始化处理动作和消息
            minimum_action = Action.NONE  # 默认不采取任何动作
            message = None                # 警告消息

            # 情况1：同时使用了弃用参数和新参数
            # 处理策略：根据配置决定是抛出错误还是只警告，并移除弃用参数
            if old_name in kwargs and new_name in kwargs:
                minimum_action = Action.RAISE if raise_if_both_names else Action.NOTIFY_ALWAYS
                message = f"Both `{old_name}` and `{new_name}` are set for `{func_name}`. Using `{new_name}={kwargs[new_name]}` and ignoring deprecated `{old_name}={kwargs[old_name]}`."
                kwargs.pop(old_name)  # 移除弃用的参数，保留新参数

            # 情况2：只使用了弃用参数，且有新的替代参数
            # 处理策略：显示警告，自动替换参数名
            elif old_name in kwargs and new_name is not None and new_name not in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`. Use `{new_name}` instead."
                kwargs[new_name] = kwargs.pop(old_name)  # 参数名替换：移除旧参数，添加新参数

            # 情况3：只使用了弃用参数，且没有指定新的替代参数
            # 处理策略：只显示弃用警告，不进行参数替换
            elif old_name in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`."

            # 如果存在附加消息，则将其追加到主消息后面
            if message is not None and additional_message is not None:
                message = f"{message} {additional_message}"

            # 版本控制策略：根据当前版本和弃用版本的关系调整处理动作
            if is_greater_or_equal_version:
                # 子情况A：如果要求对已过期参数抛出错误，且当前需要处理
                if raise_if_greater_or_equal_version and minimum_action != Action.NONE:
                    minimum_action = Action.RAISE  # 将警告升级为错误

                # 子情况B：如果要求忽略已过期参数的通知，且当前是普通警告
                elif not warn_if_greater_or_equal_version and minimum_action == Action.NOTIFY:
                    minimum_action = Action.NONE  # 将警告降级为忽略

            # 执行最终的处理动作
            if minimum_action == Action.RAISE:
                # 抛出ValueError异常，强制用户更新代码
                raise ValueError(message)
            # 显示警告（但避免在torch.compile编译时中断）
            elif minimum_action in (Action.NOTIFY, Action.NOTIFY_ALWAYS) and not is_torchdynamo_compiling():
                # 使用FutureWarning而不是DeprecationWarning，因为前者默认不会被忽略
                warnings.warn(message, FutureWarning, stacklevel=2)

            # 调用原函数并返回结果
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper
