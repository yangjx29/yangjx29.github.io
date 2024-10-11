---
title: ML-based-knowledge
categories: paper reading
abbrlink: '96138224'
date: 2024-10-11 16:58:41
updated: 2024-10-11 16:59:41
tags:
mathjax: true
toc: true
---
<meta name="referrer" content="no-referrer"/>

## \[Large Language Models for Code: Security Hardening and Adversarial Testing]

### Paper Summary (What)

*   **Main takeaway**: This paper focuses on enhancing the security of large language models (LLMs) used for code generation by addressing two key areas:

    *   Security hardening: improving the model's ability to generate secure code.
    *   Adversarial testing: evaluating the model's security by degrading its performance from an adversarial perspective.

*   **Hypothesis**: LLMs can be guided to generate secure or insecure code without modifying the model's core weights through a task called *controlled code generation*.

*   **Algorithmic structure**: The paper introduces *SVEN*, a novel learning-based approach that uses continuous vectors (prefixes) to guide the model in generating secure or insecure code. This is achieved without changing the model's internal weights.

### Issues or Targets Addressed by the Paper (Why)

*   **Problem**: LLMs, though successful in generating functional code, often produce insecure code with vulnerabilities. This poses a risk when using LLMs in practical, security-sensitive scenarios, such as software development.
*   **Motivation**: The paper aims to address this security gap by enhancing LLMs' ability to produce secure code while still retaining their capacity to generate functionally correct code.

### Detailed Information (How)

#### Problem Setting

*   **Context**: The problem involves improving LLMs' ability to generate secure code in scenarios where code security is crucial.
*   **Task**: The paper introduces a new task called *controlled code generation* which provides the model with a binary property (secure or insecure) to guide its code generation.

#### Methodology

*   **Approach**: The authors propose *SVEN* (Security Vulnerability Enhancer), a novel technique that uses continuous prompts to guide the LLM without modifying its weights.

    *   SVEN trains two property-specific sequences (prefixes) of continuous vectors to steer the LLM to generate secure or insecure code.
    *   SVEN maintains functional correctness through specialized loss terms applied during training, particularly focusing on different code regions (changed and unchanged).

*   **Data**: The authors curated a high-quality dataset of security fixes from GitHub commits to train SVEN efficiently on a small dataset.

#### Assumptions

*   The LLM retains its original ability to generate functional code after being guided towards secure or insecure code generation.
*   The security of code generation can be effectively controlled using small continuous vector sequences (prefixes).

#### Prominent Formulas or Structure

*   Loss functions are introduced to control security in changed code regions while preserving functional correctness in unchanged regions.

    *   Conditional language modeling loss for security-sensitive regions.
    *   Contrastive loss between secure and insecure code.
    *   KL divergence loss to preserve functional correctness.

#### Results

*   SVEN demonstrated a significant improvement in the security of generated code. For instance, using SVEN on a CodeGen model (2.7B parameters) increased the generation of secure code from 59.1% to 92.3% during security hardening.
*   Conversely, the adversarial testing setting decreased secure code generation to 36.8%.

#### Limitations

*   The curated dataset is relatively small, but the authors argue that SVEN's efficiency compensates for this.
*   SVEN's focus is primarily on binary properties (secure/insecure), and generalizing beyond this binary control might be a challenge.

#### Confusing Aspects

*   The relationship between functional correctness and security control is complex, and balancing both without sacrificing one aspect for the other requires careful tuning.

### Conclusions

#### The Author's Conclusions

*   The authors conclude that SVEN is an effective, modular approach to controlling LLMs for secure code generation. It can be seamlessly applied to existing LLM-based code completion systems to improve their security capabilities.

### Possible Future Work / Improvements

*   Extending the controlled code generation task to multi-property settings beyond secure/insecure could be explored.
*   Enhancing the size and quality of the training dataset, possibly through automated data curation techniques.

***

### Part 1. 标题 & 作者

*   **标题**: Large Language Models for Code: Security Hardening and Adversarial Testing
*   **作者**: Jingxuan He, Martin Vechev

### Part 2. 摘要

*   该论文探讨了在代码生成的背景下，LLMs如何进行安全加固和对抗性测试。作者提出了一个新的安全任务——受控代码生成，通过使用连续向量前缀来指导模型生成安全或不安全的代码，同时保留其生成功能性代码的能力。

### Part 3. 导言

*   LLMs在代码生成方面取得了显著的功能性成就，但安全性仍是一个重大问题。文中提到，诸如Copilot等系统在生成代码时，常常产生带有漏洞的代码。该论文旨在通过受控代码生成来解决这一问题，提出一种不改变模型权重的学习方法SVEN，以改进安全性。

*   论文的贡献

    *   提出一种新的安全任务controlled code generation，可用于对基于LM的代码生成器进行安全加固和对抗测试
    *   SVEN，上述任务的解决方法，包括模块推理以及平衡安全控制和功能正确性的专门培训程序
    *   高质量的数据集
    *   对于SVEN广泛的评估

### Part 4. 结论

*   作者认为，SVEN是一种有效的模块化方法，可以增强LLMs生成安全代码的能力，同时保持其功能正确性。该方法可以应用于现有的代码补全系统，为开发人员提供更安全的代码生成环境。
*   SVEN学习连续的前缀来引导程序生成朝向给定的属性，而不改变LM的权重

### Part 5. 相关工作

*   论文回顾了与代码生成、代码安全、漏洞检测相关的文献，并指出现有的研究在自动检测和修复漏洞方面的不足。作者对比了SVEN与其他研究在安全控制和功能正确性上的不同之处。

### Part 6. 相关模型

**Controlled Code Generation**

*   提供了一个性质c来指导LM生成满足性质c的代码
*   需要关注的是一个二元安全属性：c = { sec，vul }。如果c = sec，输出程序应该是安全的，允许LM的安全加固。若c = vul表示一个对抗测试场景，我们通过试图降低LM的安全等级来评估LM的安全等级

$P(\mathbf{x}|c)=\prod\limits_{t=1}^{|\mathbf{x}|}P(x_t|\mathbf{h}_{<t},c)$

*   与传统的漏洞检测、修复和注入不同，ccg针对一个代码完成设置，并对用户即将编写的代码生效![\<img alt="" data-attachment-key="B7A2V4DU" width="1211" height="203" src="attachments/B7A2V4DU.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/2ce64f9940214ffc9b0367af99be1944.png)

**Sven**

主要包括推理、训练和数据构造

*   推理

    *   使用连续的promots，特别是**前prefix-tuning方法**；连续的提示可以方便地用梯度下降法进行优化且更具表达能力

    *   由于注意力机制，$SVEN_{sec}$极大地提升了生成安全程序的概率；同样，$SVEN_{vul}$可以驱动LM以更高的概率生成不安全代码

    *   sven是轻量化和模块化的，前缀参数的个数由前缀长度N可调，前缀作为一个独立的模块，可以方便地连接或脱离LM。

**前缀调优**：以通过插入前缀向量（continuous vectors）来引导模型生成具有特定属性的代码，而不需要对模型的核心参数进行重新训练

每个前缀是一组连续向量，它们作为模型推理时的初始隐藏状态，用于引导生成过程。前缀的大小与LLM中隐藏状态的维度一致。SVEN根据安全属性（安全或不安全）分别训练了两组前缀：SVENsec（安全代码前缀）和SVENvul（不安全代码前缀）

调优过程：

前缀在推理开始时被加入模型的隐藏状态，通过Transformer的注意力机制影响模型的生成行为。SVENsec通过引导模型在生成代码时偏向于安全代码生成，而SVENvul则用于引导模型生成不安全的代码，从而实现对抗性测试或模拟潜在的攻击

* 训练

  * 从GitHub中提取安全修复来构造这样的数据集，其中为修复之前的版本是不安全的，修复之后的版本是安全的

  * 修改后的代码决定了整个程序的安全性，而未修改的代码是中性的。所以，在安全敏感区域，我们训练SVEN以强制实现代码安全属性，而在中立区域，我们约束SVEN遵守原始LM以保持功能正确性。从程序维度、行维度和字符维度构建了三种token的掩码

  * SVEN模型使用了三类损失函数来实现对安全性和功能正确性的控制

    *   条件语言建模损失：用于优化模型在修改区域生成符合安全属性的代码

    $L_\mathrm{LM}=-\sum_{t=1}^{|x|}m_t\cdot\log P(x_t|h_{<t},c)$其中，mtm\_tmt​ 为二元掩码向量，标记哪些位置属于修改区域

    *   对比损失（LCT）：该损失函数用于在修改区域内对比安全和不安全的代码生成。其目的是同时优化安全代码生成和抑制不安全代码生成$L_{\mathrm{CT}}=-\sum_{t=1}^{|x|}m_t\cdot\log\frac{P(x_t|h_{<t},c)}{P(x_t|h_{<t},c)+P(x_t|h_{<t},\neg c)}$

    *   KL散度损失（LKL）：为了保持代码的功能正确性，该损失函数在未修改区域施加约束，促使SVEN模型在未修改区域的行为与原始LLM保持一致$L_{\mathrm{KL}}=\sum_{t=1}^{|x|}(1-m_t)\cdot\mathrm{KL}(P(x_t|h_{<t},c)\|P(x_t|h_{<t}))$KL散度用于衡量SVEN生成的代码分布与原始模型分布之间的差异，该项在未修改区域起到正则化作用

    *   总损失函数：上述三个损失函数的加权和

    $L=L_\mathrm{LM}+w_\mathrm{CT}\cdot L_\mathrm{CT}+w_\mathrm{KL}\cdot L_\mathrm{KL}$其中，$w_{\text{CT}}$​ 和 $w_{\text{KL}}$  是对比损失和KL散度损失的权重，控制它们在总损失中的影响。通过调节这些权重，可以在安全性控制与功能正确性之间取得平衡

  * sven的训练是高效的，可以在小数据集上高效的训练。这是因为：

    *   SVEN仍然执行原始代码生成任务，只对给定的安全属性调整输出代码分布，这与漏洞的检测和修复等全新的任务不同
    *   SVEN的训练只更新小的前缀，而不修改庞大的LM
    *   SVEN在数据效率方面的优势尤为重要，因为获取高质量的漏洞数据集具有挑战性

* 构建高质量的训练数据集

  *   作者首先对现有的漏洞数据集进行了详细审查，例如CrossVul、Big-Vul和VUDENC。这些数据集基于CVE（Common Vulnerabilities and Exposures）记录，覆盖了广泛的漏洞和项目。作者选择了覆盖面广泛、适用于不同项目的CrossVul和Big-Vul数据集，并将它们与VUDENC结合起来，以确保代码数据集涵盖不同语言（主要是C/C++和Python）以及不同类型的漏洞修复。
  *   进一步进行手动清洗和去除误报，确保只保留那些与目标CWE相关的真实安全修复

**使用场景**

*   安全加固

在推理时插入安全前缀，提升生成安全代码的概率

*   对抗性测试

作者揭示了$SVEN_{vul}$可以被恶意使用。攻击者可以将其插入到开源的LM中，评估模型在潜在攻击下的表现。

### Part 7. 实验

*   实验表明，SVEN在多种模型（如CodeGen 2.7B）和多种漏洞类型上表现优异，显著提高了代码生成的安全性。对于不同的温度设置，SVEN在控制安全性和保持功能正确性方面均表现出色。

#### 实验设置

**模型选择**

*   codegen

    *   包括350M、2.7B和6.1B参数的版本

*   其他的LM模型包括InCoder和SantaCoder

**安全性评估**

*   安全性评估采用了现有的最先进的评估方法，包括手动构建的多个安全相关代码场景。这些场景基于现实世界中的代码生成任务，并且涵盖了广泛的安全漏洞类型（如SQL注入、越界读写、路径遍历等）。
*   每个评估场景都会为模型生成若干代码完成示例，经过过滤后用GitHub的CodeQL查询进行安全性检查。**安全率**（security rate）是实验中关键的衡量标准，表示生成的代码中有多少符合安全标准。
*   为了确保评估的可靠性，实验对每个场景进行了多次采样，并计算了不同采样temperature下的安全性表现，最终报告平均安全率及95%置信区间。

**功能正确性评估**

*   功能正确性评估基于**HumanEval**基准，这是一个广泛使用的标准代码生成评估数据集。实验中使用了**pass\@k**作为评估指标，表示生成的k个代码样本中，有多少个样本能够通过全部单元测试。
*   为了减少评估方差，实验对每个问题运行了4种常见的采样温度（0.2、0.4、0.6、0.8），并记录了这些温度下的最高pass\@k得分。

**超参数和计算资源**

*   在超参数选择上，前缀参数的大小被设置为总模型参数的约0.1%，确保SVEN模型的轻量性。
*   训练和推理任务是在NVIDIA A100和H100 GPUs上进行的，即使是最大的6B参数模型，SVEN的训练时间也不到3小时，显著低于通常的LLM预训练需求。

### Part 8. 讨论 & 总结

*   作者讨论了SVEN在实际应用中的前景，特别是如何将其应用于现有的代码补全系统以增强安全性。他们还指出了未来可能的研究方向，例如如何自动化构建更大规模的高质量训练数据集。

*   SVEN的局限

    *   SVEN目前没有捕获某些与安全相关的行为，例如第6.4节中的CWEs，SVEN没有推广到除Python和C / C + +以外的编程语言。

        *   可以通过构建更全面的训练数据集来解决。也可以用自用推理技术或者crowdsourcing

    *   减少方程( 4 )中的损失LKL，减少了令牌概率的差异，这只是维护功能正确性的间接代理

    *   在推理时，SVEN作为前缀独立于用户提供的提示，在SVEN和提示语之间引入依存关系可以带来额外的表达性和准确性。

### Controlled Code Generation 和 SVEN

*   Controlled Code Generation 是任务，SVEN 是解决方案

    *   Controlled Code Generation 是论文提出的一个新任务。这个任务的核心目标是指导LLMs在生成代码时，控制代码是否满足某种安全属性。具体来说，任务通过向模型输入一个二元属性（sec或vul）来控制生成结果，使模型可以生成安全的代码或不安全的代码
    *   SVEN（Security Vulnerability Enhancer） 是为了解决这个受控代码生成任务而设计的具体方法。SVEN的目的是通过学习连续向量前缀，在不修改LLM核心权重的情况下实现对代码生成的安全性控制。
    *   SVEN 在 Controlled Code Generation 中平衡了安全性与功能正确性![\<img alt="" data-attachment-key="3Q5236WT" width="615" height="255" src="attachments/3Q5236WT.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/5cb1805c675e491eacb5fdaeb7e8076b.png)

**在malicious code analysis中的运用**

1.  生成对抗性代码扰乱代码分析
2.  生成复杂和隐蔽的漏洞代码
3.  对抗性攻击提高模型对漏洞检测的难度