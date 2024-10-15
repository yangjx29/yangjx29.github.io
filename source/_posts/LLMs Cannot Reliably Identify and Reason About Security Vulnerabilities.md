---
title: LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities
categories: paper reading
abbrlink: 4153f353
date: 2024-10-09 20:15:51
tags:
mathjax: true
# toc: true
---
<meta name="referrer" content="no-referrer"/>

## \[LLMs Cannot Reliably Identify and Reason About Security Vulnerabilities (Yet?): A Comprehensive Evaluation, Framework, and Benchmarks]

### Paper summary (what)

**TLDR:**

*   The paper evaluates the reliability of Large Language Models (LLMs) in identifying and reasoning about security vulnerabilities.
*   The authors develop an automated evaluation framework, SecLLMHolmes, which tests the capability of eight state-of-the-art LLMs across 228 code scenarios.
*   They find that current LLMs, including advanced models like GPT-4 and PaLM2, show significant non-robustness and inconsistency in real-world vulnerability detection tasks.

**Algorithmic structure:**

*   The core of the framework is built on five key components: parameters, prompt templates, datasets, code augmentations, and an evaluator module.
*   LLMs are tested on a variety of scenarios with different complexity levels and augmented code to measure their robustness and reasoning abilities.
*   The evaluation uses a multi-dimensional approach, considering factors like determinism, prompt diversity, and real-world application.

### Issues or targets addressed by the paper (Why)

*   **Issue:** The reliability of LLMs in detecting and reasoning about security vulnerabilities has been questioned, particularly regarding their use in automated vulnerability repair.
*   **Problem:** The lack of standardized and automated benchmarks to evaluate LLMs' performance in this domain.
*   **Motivation:** As LLMs are increasingly integrated into development environments, it is crucial to understand their limitations and potential risks, particularly in security-related tasks.

### Detailed Information (How)

#### Problem Setting

*   **Setting:** The problem is framed around the evaluation of LLMs in a classification task where they identify security vulnerabilities in code snippets.
*   **Environment:** The evaluation includes scenarios from two programming languages, C and Python, with varying levels of code complexity.
*   **Evaluation:** The LLMs are assessed across eight dimensions, including deterministic response, robustness, and reasoning capabilities.

#### Methodology

*   **Approach:** The authors introduce SecLLMHolmes, an automated framework that generates test prompts, applies them to LLMs, and analyzes their responses.
*   **Testing:** The framework evaluates eight LLMs using 228 scenarios, which include real-world and hand-crafted examples, and tests them with 17 different prompting techniques.
*   **Tools:** The responses are assessed for accuracy and reasoning quality using metrics like Rouge score, cosine similarity, and human-like reasoning patterns.

#### Assumptions

*   **LLMs Capabilities:** The framework assumes that LLMs can be configured and tested in a controlled environment with parameters like temperature and token limits.
*   **Test Integrity:** It is assumed that the LLMs have not seen the test scenarios during their training phase, particularly the real-world CVEs.

#### Prominent Formulas or Structure

*   No specific formulas are highlighted, but the framework involves complex evaluation metrics like Rouge scores and cosine similarity to assess reasoning quality.

#### Results

*   **Findings:** LLMs show non-deterministic behavior, providing inconsistent results under the same conditions. The reasoning provided by LLMs often lacks correctness, even when the correct vulnerability is identified.
*   **Performance:** The LLMs perform poorly in real-world scenarios, with high false positive rates and vulnerability misclassification when code is slightly modified.
*   **Key Statistics:** For example, GPT-4 and PaLM2 showed incorrect answers in 26% and 17% of cases, respectively, when minor code changes were introduced.

#### Limitations

*   **Framework Scope:** The framework might not capture all potential vulnerabilities and may not generalize across all programming languages.
*   **LLM Constraints:** The study is limited by the specific configurations and versions of the LLMs tested, which might not represent the full capabilities of these models.
*   **Real-World Relevance:** While the scenarios are carefully crafted, they might not fully represent the complexity and variety of vulnerabilities in real-world software.

#### Confusing aspects of the paper

*   **Determinism and Temperature Settings:** The relationship between LLM temperature settings and their performance consistency was discussed but could benefit from more detailed explanation.

### Conclusions

#### The author's conclusions

*   The authors conclude that current LLMs are not yet reliable enough for automated security vulnerability detection, especially in real-world applications.
*   They emphasize the need for further advancements in LLMs before they can be trusted as general-purpose security assistants.

### Possible future work / improvements

*   **Algorithm Enhancements:** Further refinement of LLMs to improve their reasoning capabilities and robustness to code modifications.
*   **Expanded Framework:** Incorporating more diverse programming languages and real-world scenarios into the SecLLMHolmes framework.
*   **Hybrid Approaches:** Combining LLMs with traditional static and dynamic analysis tools to enhance overall detection performance.

## Part1. abstract

截至目前，LLMs在安全相关领域的鉴别能力是比较缺乏的，基于此，作者开发了一个完全自动化的评估框架SecLLMHolmes，该框架对LLMs是否能够可靠地识别和推理与安全相关的Bug进行了迄今为止最详细的调查。

研究结果揭示了即使是最先进的模型，如' PaLM2 '和' GPT-4 '，也存在显著的非鲁棒性

## Part2. introduction

作者构建了一组228个代码场景，并使用我们的框架在八个不同的研究维度上分析了八个最有能力的LLM。在最先进的模型也表现出了不稳定性。

实验表明

*   尽管所有模型的误报率都很高，但llm的能力还是决定于模型以及Prompt技术
*   llms的输出是不固定的
*   即使正确地识别了一个漏洞，LLMs为这个决策提供的推理也往往是不正确的
*   llms的COT是不健壮的，最简单的混淆(空格，改函数名)都能对其产生影响

本文的贡献：

1\. 开发了一个全面的框架SecLLMHolmes来测试LLM识别和推理软件漏洞的能力。框架是完全自动化的，包括一组228个代码场景和17种提示技术

2\. 测试了8个最先进的LLM用于漏洞检测任务，表明到目前为止没有一个LLM在漏洞检测任务上取得令人满意的性能

3\. 识别并列举了当前LLMs表现出的的一系列缺点。为研究人员提供了一个checklist，显示了在LLMs被认为可以在野外用于脆弱性检测任务之前需要解决的问题

## Part3. Framework

**SecLLMHolmes**

![\<img alt="" data-attachment-key="83K4VARW" width="552" height="349" src="attachments/83K4VARW.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/f18070a3b4084b069b77896c48c3abeb.png)

**1.LLM configuration**

为了适配各种llms，框架提供3个用户可输入的配置：

*   LLM-Specific Best ‘Prompting Practices’ and Rules

    *   通常根据模型文档完成

*   LLM Initialization and Configuration

*   LLM Chat Structure and Inference Function

![\<img alt="" data-attachment-key="ZCL3Z288" width="616" height="326" src="attachments/ZCL3Z288.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/b653cdedde9a4d5ebe32111ac56d282d.png)

**2. LLM Parameters**

llm的responses收到两个参数的显著影响：

*   **temperature**

    *   控制LLM输出的确定性。温度值越高，模型的输出越随机，越具创造性；温度值越低，模型输出更确定且更一致。为了评估LLM的性能，SecLLMHolmes会根据任务需求调整温度值，例如在需要确定性结果时设置较低温度。

*   **top\_p**

    *   控制核采样(nucleus sampling)，其中LLM考虑具有top p概率质量的令牌的结果。SecLLMHolmes会根据模型的具体需求调整Top P的取值，使其能够生成合适的响应。

**3.Prompt Templates**

*   zero-shot task-oriented (ZS - TO)
*   zero-shot role-oriented (ZS - RO)
*   few-shot task-oriented (FS - TO)
*   few-shot role-oriented (FS- RO)

![\<img alt="" data-attachment-key="S3UTRTN3" width="461" height="859" src="attachments/S3UTRTN3.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/39e5e671c08a420fbf43acbab54bb727.png)

三类：

*   标准( S )

    *   直接向模型提供问题，要求模型判断代码中是否存在某种安全漏洞。标准提示最简单，但也可能无法激发模型的复杂推理能力

*   Step-By-step推理型后视偏差( R )

    *   使用COT推理。要求模型按照逐步推理的方式解决问题，例如使用"链式推理"（Chain of Thought，COT）方法。这种提示模拟了人类安全专家的推理过程，例如：先概览代码，再检查潜在的漏洞子组件，最后基于分析做出判断。

*   基于定义( D )

    *   在提示中为模型提供安全漏洞的定义，并让模型基于此定义判断代码中是否存在该漏洞。这种提示为模型提供了更多背景信息，帮助其做出更精确的判断

**4. Datasets**

设计了228个代码场景(手工制作48个,真实世界30个,代码增强150个)来测试LLMs检测代码中软件漏洞的能力的各个方面。

*   **手工制作的CWE场景**![\<img alt="" data-attachment-key="D9Y8ZVEI" width="580" height="342" src="attachments/D9Y8ZVEI.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/da8a85e05ad34639a22539ed75ff1a68.png)

研究者手工设计了48个代码场景，涵盖8种MITRE排名前25的关键软件漏洞（如越界写、SQL注入、路径遍历等）。每个CWE类别包含六个代码场景（3对漏洞代码和修复代码），并分为**简单**、**中等**和**复杂**三个难度级别。这些场景用于评估LLM在处理不同复杂度的代码时的表现。

*   **真实世界CVE场景**

    *   该数据集包含30个真实的漏洞场景，来源于开源项目的CVE（Common Vulnerabilities and Exposures，常见漏洞和暴露），确保这些场景是在当前LLMs的训练日期之后发布和修复的，以避免模型在训练中见过这些场景

<!---->

*   **Code Augmentations**

为了解决没标准框架来评估代码安全相关任务的鲁棒性评估的问题而提出的，这些增广分为两个不同的类别：

*   Trivial Augmentations

    *   例如随机重命名函数、插入无用的代码等，测试LLM对这些噪音的鲁棒性。增强衡量了LLMs对随机噪声的鲁棒性

*   Non-Trivial Augmentations

    *   例如使用安全函数但对其进行误导性命名，测试LLM对复杂代码结构的敏感度，以测量它们的鲁棒性和对函数或变量名的语义、特定库函数或代码安全实践的偏见

**5. Ground-Truth Reasoning** $G_r$

评估LLM给出的推理是否与其最终回答一致，使用GPT-4来分析LLM的输出，验证其提供的推理是否合理，并通过Rouge得分和余弦相似度来量化推理的准确性

**6.Evaluator**

使用提出的框架来研究LLM，包括以下模型

![\<img alt="" data-attachment-key="L3YP6CFR" width="546" height="212" src="attachments/L3YP6CFR.png" ztype="zimage">](https://img-blog.csdnimg.cn/direct/d131482b40084b679099100c9043a5e0.png)

## Part3. Experimental

*   **Evaluation for Deterministic Responses**

目的是保证在相同的参数下多次运行相同的测试，应该提供相同的最终判决。

结果：0.0是一个LLM获得一致响应的最佳temperature值，尽管我们注意到即使在这个设置下，一些LLM也无法提供一致的响应。

*   **Performance Over Range of Parameters**

低温度可以保证模型的一致性，高温度可以增加模型的创造性。

结果并没有随着模型温度的升高而表现出更好的性能的总体趋势。由于提高温度并不能带来整个模型结果的普遍改善，为了优先考虑结果一致性，我们选择将0.0作为剩余实验的"温度"值，并将" top\_p "设置为LLM特定的默认值。

*   **Diversity of Prompts**

在某些情况下，当前LLM的反应可能并不完全依赖于忠实和准确的推理。不同提示类型对模型的影响有所不同。例如，GPT-4和CodeLlama34b在逐步推理提示（R）和定义提示（D）下表现最好，而GPT-3.5和CodeChat-bison\@001在少样本角色导向提示（FS-RO）下表现较优。

尽管提示类型可以显著影响模型的表现，但没有一个通用的提示类型能适用于所有模型和场景。不同模型在不同提示下表现各异。逐步推理提示和定义提示对某些LLM（如GPT-4）更为有效，而少样本提示技术则能改善部分模型（如GPT-3.5）的表现。因此，在未来评估中，应根据模型特点选择最合适的提示类型

*   **Evaluation Over Variety of Vulnerabilities**

大多数模型对补丁版本的正确分类性能较差，这使得这些LLM不适合实际情况，因为它们大多将安全代码标记为易受攻击，从而导致许多错误的警报

我们观察到，对于几乎所有的模型( pvalue = 0.003)，小样本提示的表现显著优于零样本提示，而角色导向提示的表现略优于任务导向提示( p值= 0.1)

*   **Code Difficulty Levels**

LLMs一般在代码简单的场景下表现较好，只有有限的例外

发现( 1 ) LLMs不熟悉库函数功能的安全规范；( 2 ) LLMs不能处理复杂的多函数和多变量的数据流模式。

*   **Robustness to Code Augmentations**

没有一种提示技术是完全稳健的，因为我们的稳健性测试甚至打破了所有LLM的最佳提示技术和思维链，导致错误的反应。

添加空格和新行字符，在某些情况下也会导致所有的LLMs做出错误的回答和推理，并进一步破坏他们的思维链推理

*   **Real-World Cases**

LLMs除了为易受攻击的代码提供错误的答案外，还经常错误地将补丁示例识别为易受攻击的，如果在生产中使用这些模型，这将会产生特别大的问题，因为这会使假阳性的数量急剧增加。

我们还观察到，在真实世界的情况下，小样本提示并不起作用，这可能是因为LLMs无法外推从提供的实例中得到信息

零样本的面向角色的提示" R2 "对于所有的LLM都表现出相对更好的性能

## Discussion & Conclusion

框架的限制：

*   尽管要求模型按个时候回答，但是研究者发现100个样例中还是有2个没有按照要求格式给出，这里是有隐患的

<!---->

*   knowledge cut-off：之后的CVEs数据需要手动加入到框架中
*   框架使用了3个指标衡量，选择多数结果作为最终结果，当2个同时误判时，可能出现问题
*   文章中考虑的代码场景还需要拓展

结论：该论文使用该框架对最先进的LLM进行了评估，表明它们目前在此任务上是不可靠的，并且当被要求识别源代码中的漏洞时，会错误地回答。基于这些结果，目前最先进的LLM还没有准备好用于漏洞检测，并敦促未来的研究解决突出的问题。