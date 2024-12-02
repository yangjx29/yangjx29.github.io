---
title: peft学习笔记
abbrlink: 110ad625
date: 2024-12-02 11:36:12
updated: 2024-12-02 11:36:14
categories: ML
tags:
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>

# peft学习笔记

第一次使用lora微调，踩的坑已经多到心力憔悴。所以写一篇博客，总结梳理一下我混乱的逻辑。



## 什么是lora

LoRA 的全称是 **LoRA: Low-Rank Adaptation of Large Language Models**，是一种以极低资源[微调](https://so.csdn.net/so/search?q=微调&spm=1001.2101.3001.7020)[大模型](https://edu.csdn.net/cloud/pm_summit?utm_source=blogglc&spm=1001.2101.3001.7020)的方法，其来自于论文 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (还没读~)

Lora是PEFT微调方式的一种，google在论文[《Parameter-Efficient Transfer Learning for NLP》](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1902.00751.pdf)指出，在面对特定的下游任务时，如果进行 Full-Fintuning（即预训练模型中的所有参数都进行微调），太过低效；而如果采用固定预训练模型的某些层，只微调接近下游任务的那几层参数，又难以达到较好的效果。

在 LoRA 方法提出之前，也有很多方法尝试解决大模型微调困境的问题。其中有两个主要的方向：

- 添加 adapter 层；
- 使用 prefix-tuning。

但是这两种方法都有局限性：

* Adapter 层会引入推理时延。简单来说，它的主要思想是在预训练模型的每一层 Transformer 中插入一个小的可训练的模块，称为 adapter。这样可以保持预训练模型的权重不变，只更新 adapter 的参数，从而实现参数高效和灵活的迁移学习。
* Prefix-tuning 难以优化。prefix-tuning 方法是受语言模型 in-context learning 能力的启发，只要有合适的上下文则语言模型可以很好地解决自然语言任务。但是，针对特定的任务找到离散 token 的前缀需要花费很长时间，prefix-tuning 提出使用连续的 virtual token embedding 来替换离散 token。这些 virtual token embedding 需要作为可训练参数进行优化，而且会减少下游任务的序列长度

LoRA 的核心思想是冻结[预训练](https://so.csdn.net/so/search?q=预训练&spm=1001.2101.3001.7020)的模型权重，并将可训练的秩分解矩阵注入 Transformer 架构的每一层，从而大大减少了下游任务的可训练参数数量。相比于完全微调，LoRA 可以节省显存、提高训练速度、减少推理延迟，并且保持或提升模型质量

lora可以应用于自回归模型 如 GPT 系列和 Encoder-Decoder 模型（如 T5），并且可以与不同规模的预训练模型（如 RoBERTa, DeBERTa, GPT-2, GPT-3）兼容。

## LoRA使用

在使用前，首先需要配置环境，用到peft库，源码在https://github.com/huggingface/peft

然后可以开始使用。

**前期准备**

* 加载预训练模型+数据集准备
  * 这里使用huggingface api即可，示例Model选择codeT5

```python
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
}
from peft import LoraConfig, get_peft_model,PeftModel, TaskType,PeftConfig

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
config.num_labels=1
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
```

1. **定义用到的类和方法**

**LoraConfig**

LoraConfig是 `peft` 库中配置 LoRA 层的核心工具，用于配置低秩适应（LoRA）相关的参数，以便在训练过程中调整模型的部分参数。允许你自定义如何在模型中应用 LoRA 技术。

**`r` (低秩矩阵的秩)**

- **作用**：这是 LoRA 层的核心参数之一，它控制低秩矩阵的秩（rank）。在 LoRA 方法中，我们将权重矩阵分解为低秩矩阵，通过调整秩的大小来控制训练中需要更新的参数数量

**`lora_alpha` (LoRA的缩放因子)**

- **作用**：`lora_alpha` 是 LoRA 的缩放因子，用来调整低秩矩阵对模型输出的影响程度。LoRA 会将原始权重矩阵与低秩矩阵相加，而 `lora_alpha` 用于缩放低秩矩阵的贡献

**`target_modules` (应用 LoRA 的模块)**

- **作用**：LoRA 只会应用于某些特定的模块，通常是在模型的某些层中，例如全连接层或自注意力层。`target_modules` 允许你指定哪些模块应用 LoRA。

**`lora_dropout` (Dropout 比率)**

- **作用**：这是 LoRA 层的 dropout 参数，用于控制在低秩矩阵中的 dropout 率。dropout 是一种正则化技术，能有效防止过拟合。在 LoRA 中，dropout 会随机丢弃低秩矩阵中的部分元素

**`bias` (对偏置项的处理方式)**

- **作用**：LoRA 层允许你选择是否对偏置项进行处理。偏置项是模型中的常数项，通常不会像权重矩阵那样进行学习。

`task_type` ()

* 选择期望的下游任务类型

  > Overview of the supported task types:
  >
  > - SEQ_CLS: Text classification.
  > - SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
  > - CAUSAL_LM: Causal language modeling.
  > - TOKEN_CLS: Token classification.
  > - QUESTION_ANS: Question answering.
  > - FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
  >   for downstream tasks.

其他参数：参考源代码/peft/tuners/lora/config.py

eg:

```python
lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=8,  # 低秩矩阵的秩，通常选择 4-16 之间
        lora_alpha=16,  # LoRA 系数
        lora_dropout=0.1,  # dropout 比例
        target_modules=[
        "q",  # SelfAttention 中的查询
        "k",  # SelfAttention 中的键
        "v",  # SelfAttention 中的值
        "o",  # SelfAttention 中的输出
        ],  # 目标模块（这里是自注意力层的 query、key 和 value）
        bias="none",  # 是否调整偏置项
        inference_mode=False
    )
```

**get_peft_model()**

* `get_peft_model` 是 `peft` 库中的一个函数，它的主要作用是 **将一个已有的预训练模型转换为带有高效微调模块（如 LoRA）的模型**，从而可以在不改变原始模型大部分参数的情况下，通过少量的可训练参数进行高效的微调

eg：

```python
model = get_peft_model(model, lora_config)
```

2. **训练**

**TrainingArguments**

* 通过设置 `TrainingArguments`，我们可以轻松地定义训练时的各种参数，例如学习率、批量大小、训练轮次、日志记录频率等。这个类是 `Trainer` 类的一个重要输入

* 参数包括 

  * **`output_dir`**: 指定训练结果（模型、检查点、日志等）保存的目录。

    **`evaluation_strategy`**: 定义何时进行评估，可以设置为 `no`, `epoch` 或 `steps`。如果设置为 `epoch`，则在每个训练周期结束时进行评估；如果是 `steps`，则在每个 `save_steps` 设置的步数后进行评估。

    **`save_steps`**: 定义每多少步保存一次模型检查点。

    **`save_total_limit`**: 如果指定了该值，则会保存最多指定数量的检查点，并删除最旧的检查点。

    **`logging_dir`**: 日志文件的保存路径。日志通常用于记录训练过程中的详细信息，如损失值、学习率等。

    **`logging_steps`**: 设置每隔多少步记录一次日志。

    **`learning_rate`**: 设置学习率。

    **`per_device_train_batch_size`**: 每个设备上的训练批次大小（例如，GPU或TPU）。

    **`per_device_eval_batch_size`**: 每个设备上的评估批次大小。

    **`num_train_epochs`**: 设置训练的总轮次。

    **`weight_decay`**: 权重衰减系数，通常用于防止过拟合。

    **`adam_epsilon`**: Adam 优化器的 epsilon 参数。

    **`logging_first_step`**: 如果为 `True`，则在训练的第一步就记录日志。

    **`load_best_model_at_end`**: 如果设置为 `True`，则在训练结束后加载验证集上表现最好的模型。

    **`metric_for_best_model`**: 指定用于选择最好的模型的评估指标（例如 `accuracy`, `loss` 等）。

**Trainer**

`Trainer` 是 `transformers` 库提供的一个高层接口，用于简化模型训练过程。`Trainer` 处理了训练、评估、预测等任务，并与 `TrainingArguments` 配合使用，自动进行模型训练、验证、保存和日志记录等操作。

**主要功能**

- **训练**：自动管理训练过程，包括前向传播、反向传播、损失计算和优化。
- **评估**：在训练过程中按设定的策略（如每隔一定步数或每个周期结束时）评估模型。
- **日志记录**：在训练过程中，`Trainer` 会自动记录训练日志，例如损失、学习率、时间等。
- **保存模型**：在训练过程中自动保存模型检查点。
- **加载最佳模型**：如果启用了 `load_best_model_at_end`，则 `Trainer` 会在训练结束后加载表现最好的模型。

**主要参数**

- **`model`**: 要训练的模型（例如，BertForSequenceClassification）。
- **`args`**: `TrainingArguments` 对象，定义了训练过程的配置。
- **`train_dataset`**: 训练数据集。
- **`eval_dataset`**: 验证数据集。
- **`compute_metrics`**: 一个函数，用于计算评估指标（例如准确率、F1分数）。
- **`tokenizer`**: 用于对文本数据进行处理和编码的分词器。
- **`callbacks`**: 可选的回调函数，可以在训练过程中添加自定义操作。
- **`data_collator`**: 用于打包输入数据的函数，通常用于处理批量数据

```python
def train(args, train_dataset, eval_dataset,model, tokenizer):
    """
    Train the model 
    """ 
    # 定义训练参数
    """
    args.output_dir: 用来保存训练过程中的一些中间文件，如模型检查点、日志文件等。具体保存内容包括：
    checkpoint-* 文件夹：这些文件夹包含了每个检查点（checkpoint）的模型权重和配置文件。
    trainer_state.json：记录训练过程中的状态信息（如训练的步数、学习率等）。
    eval_results 等：保存评估过程中的结果（例如损失值等
    """
    training_args = TrainingArguments(
        output_dir= args.output_dir,  # 保存训练结果的目录
        eval_strategy="steps",  # 每个 steps 评估一次,同保存策略一致
        num_train_epochs=3,  # 训练轮次
        per_device_train_batch_size=8,  # 每个设备上的批量大小
        per_device_eval_batch_size=32,  # 每个设备上的评估批量大小
        logging_dir="./logs",  # 日志目录
        logging_steps=1000,  # 每1000步记录一次日志
        save_steps=1000,  # 每1000步保存一次模型
        # load_best_model_at_end=True,  # 加载最佳模型
        # metric_for_best_model="accuracy",  # 你可以选择自己想要的评估标准（例如准确率）
        # lr_scheduler_type='linear',  # 使用线性衰减的学习率调度
        learning_rate=5e-5,  # 调整学习率
    )
    


    # 定义 Trainer 类进行训练
    trainer = Trainer(
        model=model,  # 需要微调的模型
        args=training_args,  # 训练参数
        tokenizer=tokenizer,  # 分词器
        train_dataset=train_dataset,  # 训练数据集
        eval_dataset= eval_dataset,  # 验证数据集
        # compute_metrics= compute_metrics,
    )

    # 开始训练
    trainer.train()

    # 保存微调后的模型
    trainer.save_model("./finetuned_codet5_lora")
    print("Model saved")
    model.save_pretrained("./saved_models/acc_model") 


    # 评估模型
    eval_results = trainer.evaluate()
    print(f"eval_results: {eval_results}")
```

**trainer.save_model和model.save_pretrained区别：**

| 功能             | trainer.save_model                                   | model.save_pretrained                                      |
| ---------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| 保存内容         | 保存模型的权重、配置文件、训练状态（优化器、学习率等 | 仅保存模型的权重和配置文件（不包括训练状态）。             |
| 适用场景         | 适用于使用 `Trainer` 训练时，保存训练后的完整模型    | 适用于只需要保存模型的权重和配置文件，通常用于微调后的模型 |
| 是否包括训练状态 | 包括训练状态（如优化器、学习率调度器等）             | 不包括训练状态，只保存模型本身                             |
|                  |                                                      |                                                            |

如果你使用 `Trainer` 进行训练并且希望保存训练过程中的所有信息（如训练状态、模型参数等），推荐使用 `trainer.save_model`。

如果你仅关心保存训练后的模型（例如，在微调后进行推理或部署），可以使用 `model.save_pretrained`

3. 推理

我用分类任务做示例。

推理的基本过程包括以下几个步骤：

1. **加载模型和分词器**：加载训练后保存的模型（或预训练模型）和对应的分词器（tokenizer）。

这个过程没有什么差别，同样使用from_pretrain加载微调后的模型。如果在训练时修改了分词器（例如，添加了新的词汇），也应该加载分词器

2. **准备输入数据**：根据需要对输入数据进行预处理和编码，使其适合输入到模型中。

3. **进行推理**：将预处理后的数据输入模型进行推理，获取模型的输出

## LoRA 的原理和实现

https://blog.csdn.net/Sbtgmz/article/details/131175878

LoRA 为了更加参数高效，使用相对非常小的参数$\Theta$来表示任务相关的参数增量$\Delta\Phi=\Delta\Phi(\Theta)$ ,其中$|\Theta|\ll|\Phi_0|$。寻找$\Delta\Phi$的任务就变成对$\Theta$的优化：

$$\max_\Theta\sum_{(x,y)\in\mathcal{Z}}\sum_{t=1}^{|y|}\log(p_{\Phi_0+\Delta\Phi(\Theta)}\left(y_t\left|x,y_{<t}\right.\right))$$ 

LoRA 将会使用低秩表示来编码$\Delta\Phi$ ,同时实现计算高效和存储高效。当预训练模型是 175B GPT-3, 可训练参数$|\Theta|$可以小至$|\Phi_0|$的$0.01\%$。对于预训练权重矩阵$W_0\in\mathbb{R}^{d\times k}$,可以通过低秩分解来表示其更新$W_0+\Delta W=W_0+BA,B\in\mathbb{R}^{d\times r},A\in\mathbb{R}^{r\times k}$ 且秩 $r\ll$ $\min(d,k)$ 。在训练过程中**，$W_0$被冻结且不接受梯度更新，$A$和$B$则是可训练参数**。注意，$W_{0}$和$\Delta W=BA$都会乘以相同的输入。对于$h=W_{0}x$ ,前向传播变为：

$$h=W_0x+\Delta Wx=W_0x+BAx$$

对矩阵$A$使用随机高斯初始化，对矩阵$B$使用 0 进行初始化，因此$\Delta W=BA$在训练的开始为 0。使用$\frac\alpha r$来缩放$\Delta Wx$ 。当使用Adam 优化时，经过适当的缩放初始化，调优$\alpha$与调优学习率大致相同。当进行部署时，以显式的计算和存储$W=W_0+BA$,并正常执行推理。$W_{0}$和$BA$都是$\mathbb{R}^{d\times k}$ 。当需要转换至另一个下游任务，可以通过减去$BA$来恢复$W_0$ ,然后添加不同的$B^{\prime}A^{\prime}$ 。

LoRA 模型作为一种以极低资源微调大模型的方法，可以应用于以下一些场景：

* 需要使用大规模预训练语言模型来解决特定的自然语言处理任务，例如机器阅读理解、文本摘要、文本分类等，但是又受限于硬件资源或者成本预算的场景。
* 需要根据不同的用户或者领域来定制化大规模预训练语言模型的生成风格或者内容，例如对话系统、文本生成、文本风格转换等，但是又不想为每个用户或者领域保存一份完整微调的模型的场景。
* 需要在不同的下游任务之间快速切换大规模预训练语言模型的能力，例如多任务学习、元学习、迁移学习等，但是又不想重新训练或者加载完整微调的模型的场景。




















