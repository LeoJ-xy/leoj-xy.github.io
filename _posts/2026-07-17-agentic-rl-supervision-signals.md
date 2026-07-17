---
title: "Agent 如何在环境和反馈中学习"
date: 2026-07-17
permalink: /posts/2026/07/agentic-rl-supervision-signals/
lang: zh
translation_key: agentic-rl-supervision-signals
excerpt: "从 Reward、State、Credit 三个角度理解 Agent 训练闭环：Reward 决定学习方向，State 决定训练发生的位置，Credit 决定结果归因于哪些行动。"
tags:
  - Post-training
  - Agents
  - Reinforcement Learning
---

> **从 Reward、State、Credit 理解 Agent 训练闭环**

## TL;DR

普通监督学习面对一批外部给定的样本；Agent 的策略会生成未来状态、观察和数据分布。这个内生性把 Agent 训练展开成三个基本坐标：Reward 决定学习方向，State 决定训练发生的位置，Credit 决定结果归属于哪些行动。从 SFT 到 RL 的演进，可以理解为训练信号逐步进入模型真实运行的闭环。未来的核心竞争会落到环境与 Verifier：谁能持续制造对齐当前策略、信号足够细、又难以被利用的经验，谁就能训练更长程的 Agent。

---

## 一、从答案监督到交互轨迹

设想一个 Coding Agent 接到任务：修复某个偶发的 Bug。

它搜索了 12 次代码，读了 8 个文件，提出 3 个假设，修改 2 处实现，运行 6 轮测试，最后让隐藏测试全部通过。

从部署侧看，这次运行似乎成功完成了该任务。

但从训练侧看，我们需要思考更多的问题：

- 哪次搜索提供了关键证据？
- 第一个假设虽然错误，但有没有帮助排除一条路径？
- 两处修改中，哪一处修复了根因？
- 反复运行的多次测试是否纯属浪费？
- 如果测试通过来自 Hardcode 或篡改测试，成功标签还可信吗？
- 如果实际上失败了，哪些中间步骤依然值得保留？

一次传统的文本生成通常只有输入与目标输出。然而，Agent 的动作会改变环境，环境变化又会改写后续输入。训练对象因而由一个答案扩展成了一条闭环轨迹：

$$
\tau=(o_1,a_1,o_2,a_2,\ldots,o_T,a_T)
$$

其中 $$o_t$$ 是环境观察，$$a_t$$ 是内部决策或外部行动。轨迹分布由策略与环境共同生成：

$$
\tau\sim p(\tau\mid \pi,E)
$$

Agent 训练希望找到策略：

$$
\pi^*
=\arg\max_\pi
\mathbb E_{\tau\sim p(\tau\mid\pi,E)}
[U(\tau)]
$$

这个形式揭示了一个关键点：Agent 的策略同时承担学习与数据生成两个角色。一旦策略改变，它访问的状态、获得的观察、犯下的错误都会变化。

这和普通监督学习有根本差异。普通监督学习可以近似写成：

$$
x\sim D,\qquad y\sim p(y\mid x)
$$

数据分布 $$D$$ 由外部给定，模型参数不会改变下一批输入。Agent 的 History 分布则依赖当前策略：

$$
h_t\sim d_E^\pi
$$

训练更新 $$\pi$$ 后，下一轮采样的 $$d_E^\pi$$ 也跟着改变。模型在学习如何行动的过程中，不断制造新的训练分布。

从经典 RL 视角看，这正是策略诱导的状态访问分布：策略决定会采到什么数据，基于这些数据进行更新后，新策略又会产生新的分布。在这样的场景中，经典 RL 的三个问题尤其突出：如何从复杂的环境状态中得到可靠反馈，如何获得覆盖当前策略行为的训练轨迹，以及如何把延迟结果归因到长轨迹中的具体决策。在具体实践中，人们常将它们概括为 Reward、State 与 Credit：

1. **Reward：策略究竟应该最大化什么？**  
   Reward 将任务结果转化为可优化的信号，决定学习方向。

2. **State：训练数据是否覆盖策略实际到达的状态？**  
   State 关注训练数据覆盖哪一段 Occupancy Distribution，以及教师模型示范能否迁移到当前策略的实际运行分布。

3. **Credit：延迟结果应该归因给哪些决策？**  
   Credit 决定一条长轨迹中的哪些 Action 得到强化，哪些 Action 需要修正。

全文将沿着这三个视角，讨论从大模型进入 Agent 时代至今，不断演进的各种训练方法分别提供了什么信息，又解决了哪一部分问题。

---

## 二、Reward：策略究竟应该最大化什么

训练首先要定义什么是好的，什么是不好的。

对于普通问答，真值可能是一个标准答案。对于 Agent，结果通常表现为外部环境的状态发生变化，例如：

- Bug 被修复，原有功能没有回归；
- 数据库中的工单进入正确状态；
- 网页上的订单被取消；
- 搜索报告中的关键结论得到证据支持；
- 机器人将物体放到目标位置；
- Memory 在未来查询中提供了正确且合规的信息。

因此，高质量 Reward 通常直接读取环境状态。Agent 自身对过程的描述更适合作为辅助信号。

### 2.1 Verifier 是测量仪器

可以把 Verifier 看成一台测量仪器。它把完整环境状态压缩成一个训练信号：

$$
R(\tau)=V(s_T,\tau)
$$

其中 $$s_T$$ 是终局状态。一个好的 Verifier 至少要回答：

- 目标状态是否达成；
- 未声明的副作用是否发生；
- 安全与权限是否被破坏；
- Agent 是否篡改了测试或评估通道；
- 结果是否可重复。

Coding 是较早适用可直接验证奖励的领域之一。编译器、单元测试、静态分析、文件 Diff 和隐藏测试能把程序状态转化为机器可检查的反馈。当这些反馈被直接用作 Reward 更新策略时，就是 RLVR（Reinforcement Learning with Verifiable Rewards）。

### 2.2 可验证 Reward 不一定是正确的 Reward

RLVR 提高了反馈的可重复性和扩展性，但“可验证”只表示 Checker 能稳定执行一套评分规则，不表示这套规则等同于真实任务目标。

[SWE-RL](https://arxiv.org/abs/2502.18449) 就是一个例子。它根据预测 Patch 与标准 Patch 的文本相似度计算训练 Reward。一份语义正确但写法不同的 Patch 可能因此得分较低。这个 Checker 衡量 Patch 与标准答案的文本相似度，没有检查 Bug 是否被修复。

单元测试离程序正确性更近，但仍只能检查已经编码的行为。覆盖不全时，Hardcode 可见样例也可能通过；评测通道缺少保护时，Agent 还可能修改测试或利用 Parser 漏洞。OpenAI 对 SWE-Bench Pro 的审计发现约三成任务存在问题；[BenchJack](https://arxiv.org/html/2605.12673) 在 10 个 Agent Benchmark 中找到 219 个可利用漏洞。

前一个例子的问题是代理指标离真实目标太远，后一个例子的问题是测试覆盖不足。Specification Problem 因此落到了 Verifier 的目标定义与覆盖范围上。评价一种测量方式时，需要同时检查它能否稳定验证、保留了哪些信息，以及遗漏了哪些信息。

### 2.3 Verifier 需要在多个维度之间取舍

任何测量都在压缩信息：

$$
M:\;(s_0,\tau,s_T)\longrightarrow z
$$

完整环境状态、行动轨迹和终局状态，被压缩成一个二值标签、标量分数或偏好排序。压缩是一把双刃剑，压缩越强，信号越直接简洁，训练越方便，遗漏也越多。

选择测量方式时，至少要看五个维度：

1. **Directness**：它离真实环境状态有多近；
2. **Coverage**：它覆盖了目标多少维度的信息；
3. **Repeatability**：同一轨迹重复测量能否得到相近结果；
4. **Cost**：每条轨迹测一次需要多少程序、人工或模型调用的开销；
5. **Adversarial robustness**：策略针对测量规则优化以后，信号还能否保持有效。

程序 Verifier 在 Directness、Repeatability 和 Cost 上常有优势，却可能只覆盖狭窄目标。人类专家能判断更宽的质量维度，吞吐与一致性较弱。LLM Judge 易于扩展，也更容易继承模型偏差和可利用模式。不同方案适合在不同场景和训练的不同阶段中结合使用。

#### 程序与环境状态测量：适合有明确终态的任务

典型形式包括：

- Exact Match；
- 编译与单元测试；
- 定理检查器；
- 数据库 State Diff；
- 文件系统与业务对象状态；
- Closed-world Invariant。

[Search-R1](https://arxiv.org/abs/2503.09516) 用 Final Exact Match；[ReTool](https://arxiv.org/abs/2504.11536) 用最终答案等价；WebAgent-R1 检查网页任务是否完成；[Agent-Diff](https://arxiv.org/abs/2602.11224) 检查企业 API 操作产生的完整状态差异，并让未声明副作用直接清零。

这类测量适合满足三个条件的任务：

```text
目标状态可以机器读取
行动后果可以隔离重算
有效解允许程序判定
```

它的主要风险是覆盖不足。测试全绿只能说明被测试的行为正确；终态正确也可能掩盖背后的违规路径。Agent 修改了测试、泄露了隐藏答案或造成短暂副作用，可能被简单的终态 Checker 完全忽视。

所以程序测量应同时检查：

- 目标条件；
- 未声明副作用；
- Grader 完整性；
- 关键中间约束；
- 可重复执行。

#### 人类偏好：适合开放输出任务

写作质量、Patch 可维护性、客服沟通和研究创意生成等任务很难被压缩成一个确定规则。而人类可以很好地比较两个候选：

$$
\tau_A \succ \tau_B
$$

Pairwise Preference 通常比绝对打分容易，因为人类擅长比较，但很难稳定定义出某个具体的分数。

它的优势是覆盖宽，能吸收难以形式化的判断。代价包括：

- 人工标记成本昂贵；
- 长轨迹比较负担大；
- 不同标注者价值观冲突；
- 偏好只能给出局部顺序，未必支持跨任务的统一标量；
- 标注者看到的轨迹信息也可能不完整。

因此，人类偏好更适合处理软质量维度，例如清晰度、维护性和沟通风格。任务正确性、权限与安全仍应由独立信号负责。

#### LLM Judge 与 Learned Reward Model：适合高吞吐软评价

LLM Judge 可以把专家 Rubric 扩展到成规模的轨迹评估中，Learned RM 则进一步地把偏好进一步蒸馏成低成本标量。

这类方法适合：

- 候选排序；
- Best-of-N；
- Rejection Sampling；
- 软质量筛选；
- 难度估计；
- 过程诊断。

LLM as a Judge 一直以来争议较大。这一类方法面临三类风险。

第一类是**静态偏差**：长度、位置、文风、自偏好和格式会影响评分。

第二类是**分布漂移**：RM 在旧策略数据上训练，Policy 优化后产生的新轨迹可能离开它的可信区域。

第三类是**对抗利用**：[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) 发现单个冒号和固定开头都能提高 Judge Reward。

因此，LLM Judge 更适合充当高吞吐的筛选器。定期人工抽检、Hard Verifier 校准和策略外 Holdout 都是必要的辅助措施。

#### Proxy 与启发式：适合约束局部行为

常见 Proxy 包括：

- 文本相似度；
- 格式合法性；
- 工具调用次数；
- 输出长度；
- 引用数量；
- Token 成本。

它们计算便宜，但通常只是与任务结果相关的表面特征，不能直接说明任务是否成功。

格式合法可以通过 Grammar 或 Tool Schema 直接保证；把它当主要 Reward 会鼓励模型追求标签正确。长度和工具数属于成本代理，直接优化容易出现 Under-use 或 Padding。文本相似度适合快速筛选，但与语义正确性之间仍有距离。

[ToolRL](https://arxiv.org/abs/2504.13958) 的长度实验展示了这种风险：直接奖励回复长度让 Qwen-1.5B 在 BFCL 上从 46.20% 降到 33.23%，动态长度奖励进一步降到 28.51%。[ReTool](https://arxiv.org/abs/2504.11536) 没有加入长度 Reward，RL 后回复却自然缩短约 40%。工具提高解题效率后，成本改善会随任务能力一起出现。

Proxy 最合适的角色是 Gate、诊断指标或成功轨迹内部的次级目标。

### 2.4 测量应当是成体系的框架

成熟的 Agent 测量通常是组合结构：

```text
硬约束层
  权限、安全、Grader 完整性，不满足即失败

任务 Outcome 层
  隐藏测试、State Diff、目标状态，定义任务是否完成

软质量层
  人类或 LLM Preference，比较成功解的质量

过程诊断层
  Turn-level Progress、成本、恢复、工具效率
```

这一框架的不同层次解决了不同的问题。不能把它们线性加权为单一 Reward，因为这会让高任务分补偿安全违规，也会让软质量掩盖任务失败。

训练时可以使用层级目标：

$$
\text{先满足 Constraints}
\;\rightarrow\;
\text{再最大化 Task Success}
\;\rightarrow\;
\text{最后优化 Preference 与 Cost}
$$

这比“为每项指标调一个权重”更符合真实的任务结构和训练需求。

### 2.5 优化会放大 Verifier 的系统性误差

离线评测关注 Verifier 在固定数据分布上是否判断准确。进入 RL 后，Verifier 的分数会直接决定哪些行为得到强化。只要某种行为能够稳定骗取高分，优化过程就会提高它出现的概率。

例如，某个固定开头会让 LLM Judge 错误地给出高分。在静态测试集中，这类样本只占 1%，Verifier 看起来仍有 99% 的准确率。但 RL 一旦发现这个模式，就会反复强化它；原本罕见的 1% 可能逐渐成为 Policy 的主要输出。原始分布上的平均准确率无法代表训练时的可靠性，还需要在持续更新的 Policy 分布上重新评估 Verifier。

因此，训练用测量需要额外检查：

- 错误是否呈系统性；
- Policy 是否能够主动触发错误；
- Verifier 是否和 Policy 共享信息通道；
- 评测规则是否对模型可见；
- Holdout 是否随训练动态更新；
- 攻击策略能否迁移到真实环境。

这也是为什么 Verifier 需要隔离、轮换和红队审计。Agentic RL 中，测量系统本身已经成为环境安全边界的一部分。

综合整个架构来看，一个也许行之有效的策略是：Outcome 提供真值；软评价补充难以直接编码的质量；过程信号提高学习效率；硬约束阻断不可接受的路径。结合每一种测量的优势构建系统。

---

## 三、State：训练是否覆盖策略实际到达的地方

Reward 回答了“怎样评价一条轨迹”，却无法回答“哪些轨迹会进入训练”。即使 Verifier 完全可靠，如果数据只覆盖教师模型的成功路径，学生自己造成的错误状态仍然得不到监督。

这里的 State 指策略在交互中访问到的整套 History Distribution。Agent 的每个动作都会改变下一步输入：学生只要在前面搜错一次、调用错一个工具，后续 History 就可能离开教师模型的轨迹。模型需要学习恢复的地方，不会出现在只由教师模型生成的数据中。

因此，数据由谁生成是训练机制的一部分。Trajectory SFT 在教师模型访问的状态上学习；On-policy 方法则让训练分布跟随当前策略。

### 3.1 Trajectory SFT 拟合教师模型的访问分布

Trajectory SFT 的示范可以来自人类或模型。本节讨论的工作主要使用更强的模型生成轨迹。教师模型负责提供示范，但不假设它是最优策略。

Trajectory SFT 可以看作应用在完整 Agent 轨迹上的 Behavior Cloning。教师模型先与环境交互，再把轨迹中的每个决策点拆成监督样本：

$$
\tau_T\sim p(\tau\mid\pi_T,E),\qquad
\mathcal D_T=\{(h_t^T,a_t^T)\mid \tau_T\}
$$

其中 $$h_t^T$$ 是教师模型执行到第 $$t$$ 步时的 History，$$a_t^T$$ 是它给出的动作。学生模型使用标准交叉熵拟合这些动作：

$$
\mathcal L_{\text{SFT}}
=-\mathbb E_{(h_t^T,a_t^T)\sim\mathcal D_T}
\log \pi_\theta(a_t^T\mid h_t^T)
$$

与只在轨迹结束时给一个 Outcome 相比，这种监督很密集：教师模型的每个动作都能提供 Token-level Label。它适合的任务场景主要是：

- 工具语法；
- 基本 Workflow；
- 搜索与读取顺序；
- 常见状态下的动作；
- 教师模型展示过的错误恢复模式。

[AgentTuning](https://aclanthology.org/2024.findings-acl.181/)、[Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/) 和 [FireAct](https://arxiv.org/abs/2310.05915) 都证明了轨迹蒸馏的价值。

密集标签仍可能覆盖不足。训练 History 来自教师模型的访问分布 $$d_E^{\pi_T}$$，部署 History 来自学生模型的访问分布 $$d_E^{\pi_\theta}$$。两者的差异就是模仿学习中的 Covariate Shift。

学生模型在某一步搜错一个文件，后面的 Context、假设和环境状态都会偏离教师模型的轨迹。Teacher Forcing 教的是“在教师模型生成的 History 上下一步怎么走”，而部署还要求学生模型学会“进入自己造成的异常状态后如何恢复”。

### 3.2 从 SFT 到 RL，本质区别是 Occupancy Distribution

把方法按状态来源排列，会得到一条连续谱：

```text
教师模型 SFT
  状态由教师模型产生

策略 Rollout + Verifier 筛选
  状态由采样策略产生，Outcome 决定保留或排序

On-policy RL
  状态与动作都由当前策略产生，环境提供 Outcome
```

这里的核心变量是 Occupancy Distribution，也就是策略实际访问哪些状态。CE、DPO、Policy Gradient 只是不同的更新工具。

[WebAgent-R1](https://aclanthology.org/2025.emnlp-main.401/) 提供了清晰的脉络。Qwen2.5-3B 从 6.1% 经 Behavior Cloning 提升到 20.0%，再经 RL 提升到 33.9%。直接从 Raw Model 做 RL 的版本略有退化：动作格式尚未掌握，正 Reward 几乎不出现。

这说明 BC 长期承担 Exploration Prior。它先把策略送进“偶尔能够成功”的区域，RL 才有信号继续优化。

### 3.3 失败轨迹提供多层训练信号

当前策略的 Rollout 会同时产生成功样本和真实的失败状态。终局失败的轨迹不适合作为整段正向示范，但它经过的状态和其中的有效动作仍可用于训练。按照使用的监督粒度，失败轨迹可以在四个层面发挥作用：

- **数据筛选与难度估计**：在正向轨迹蒸馏时过滤失败样本，并根据失败率估计任务难度；
- **结果级监督**：在偏好学习中作为 Rejected Sample；在 RL 中以低 Return 参与 Advantage 估计，低于 Baseline 时降低相应动作的概率；
- **过程级监督**：识别失败发生的位置，同时保留此前取得进展的有效片段；
- **状态覆盖**：将失败状态作为分支探索、恢复训练或后续 Rollout 的起点。

[SWE-Master](https://arxiv.org/abs/2602.03411) 用失败判断任务难度；[ETO](https://aclanthology.org/2024.acl-long.409/) 将失败轨迹作为 DPO 负例；[Orchard](https://arxiv.org/abs/2605.15040) 抽取失败轨迹中价值上升的片段。它们分别利用了失败数据中的难度、结果和过程信息。

同一条失败轨迹可以在 Outcome 层提供负信号，在 Process 层保留有效片段，并在 State 层提供稀缺的错误状态。终局失败只能说明整体结果不理想，不能说明轨迹中的每个动作都错误。进一步区分应当抑制和保留的决策，需要解决 Credit Assignment 问题。

---

## 四、Credit：延迟结果应该归因给哪些决策

On-policy 数据让训练发生在当前策略访问的状态上，解决的是训练分布问题。但 Verifier 通常只在轨迹结束时给出一个 Outcome，只能提供整条轨迹的总体结果，缺少各步决策的贡献信息。训练还需要把轨迹级结果转化为动作级更新，区分哪些决策推动了成功，哪些只是无关步骤，哪些导致了后续失败。这就是 Credit Assignment。

假设一条 80 步轨迹最终成功。给所有动作同一个正 Advantage，会同时强化关键决策、冗余搜索和曾经造成问题的操作。失败轨迹也可能只在最后一步犯错，前面的大量正确行为会被一并压低。

Credit Assignment 需要估计：

$$
Q(s_t,a_t)
=\mathbb E[R(\tau)\mid s_t,a_t]
$$

单条轨迹只能告诉我们某个动作与成功同时出现。要估计动作的因果贡献，理想情况是在相同状态尝试多个动作，再比较后续结果。

### 4.1 用同状态分支比较不同动作

从同一个中间状态分叉：

```text
state s_t
├── action a_1 → suffix τ_1 → reward 1
├── action a_2 → suffix τ_2 → reward 0
└── action a_3 → suffix τ_3 → reward 0.4
```

这样的数据比三条互不相关的成功轨迹更有价值，因为状态被固定，动作差异更接近因果干预。

在因果推断语境中，这类比较接近 Counterfactual；在 Agent 训练和环境工程中，更直观的叫法是 Branched Rollout 或同状态分支：固定共同前缀，只改变分叉点的 Action，再比较各个 Suffix。

[RTMC](https://arxiv.org/abs/2604.11037) 将同一任务的多条 Rollout 按共享状态组织成树，在分叉点估计 Step Advantage，SWE-bench Verified 比 GRPO 高 3.2 个百分点。[C3](https://arxiv.org/abs/2603.06859) 则冻结上下文，替换 Action 并重放固定 Continuation。

它们的共同前提是环境支持 Snapshot、Fork 和 Replay：Snapshot 保存分叉点的完整状态，Fork 从同一状态复制出多个后续分支，Replay 则用于复现并检查执行结果。如果只能从任务起点重新运行，两条轨迹在到达分叉点之前就可能因随机观察或工具返回值而不同，最终 Reward 会混入这些前缀差异的影响。因此，算法质量与环境提供受控、可复现分支的能力共同决定 Credit Assignment 的精度。

### 4.2 过程奖励的正确形态是进度差

环境无法大量分叉时，可以训练一个函数 $$\Phi(s)$$ 估计当前状态的成功潜力，再用状态变化作为过程信号：

$$
r_t^{\text{process}}
=\Phi(s_{t+1})-\Phi(s_t)
$$

这个形式有一个关键性质：沿轨迹求和后，中间项会抵消。插入冗余步骤无法凭空增加总 Reward。

[Rewarding Progress](https://arxiv.org/abs/2410.08146) 主张过程监督衡量“正确解概率的变化”，并报告相对 Outcome Reward 超过 8% 的准确率提升。[TRACE](https://arxiv.org/abs/2607.13988) 用 Frozen Reference 对 Gold Answer 的 Log-ratio Potential 计算 Turn Reward，也利用了相同的 Telescoping 结构。

绝对 $$V(s_t)$$ 或 Judge 分逐步累加会产生另一种激励：只要停留在高价值状态，就能重复拿分。[AgentPRM](https://arxiv.org/abs/2502.10325) 的实验里，Validation PRM Score 持续上升，真实成功率却从 82% 降到 70%。这是过程 Reward 被过度优化的典型信号。

过程信号与终局 Outcome 解决的是不同粒度的问题。Outcome 提供轨迹级真值，但信号稀疏；过程信号把监督细化到相邻状态之间，却会继承 $$\Phi$$ 的估计误差。因此，过程信号更适合用于 Advantage Shaping、轨迹筛选或辅助监督，终局 Outcome 仍应作为任务成败的锚点：

> 过程信号回答“这一步带来了多少进展”，终局 Outcome 判断“任务最终有没有完成”。

组合两者时，高过程分不应抵消终局失败；终局成功也不意味着轨迹中的每个动作都应得到同等强化。

### 4.3 越细的 Credit，成本越高

上一节的 Potential Difference 把 Episode-level Outcome 细化到了 Turn-level，但它仍依赖学习得到的 $$\Phi$$。如果还要判断某个具体 Action 的贡献，就需要更细的中间评估、更多策略采样，或者前文介绍的同状态分支。Credit 的分辨率与获取成本由此形成一组直接的权衡。

按照对动作贡献的分辨率，可以大致排列为：

```text
Episode Outcome
→ Turn-level Progress
→ Action-level Advantage
→ Same-state Branched Comparison
```

越靠后，信号越能区分相邻决策的贡献，但通常也需要更多 Forward、更多 Rollout 或更强的环境能力。同状态分支还要固定起始状态并执行多个候选 Action，成本远高于只检查一次终局结果。

训练系统需要在三个目标之间取舍：信号是否接近真实任务结果、是否足够稠密，以及获取成本是否可接受。常见信号位于不同的权衡点：

- Hard Outcome：真实、廉价，但稀疏；
- Process Judge：稠密、相对便宜，但真实性弱；
- Branched Rollout Credit：真实、稠密，但昂贵；
- Proxy 相似度：稠密、廉价，却可能测错目标。

最近一年的许多新方法都在用额外计算将稀疏真值转化为更细的信号。评估这类方法时，除了最终性能，还需要比较每增加一次模型调用或环境 Rollout，换回了多少更接近真实结果的 Credit Resolution。

---

## 五、从闭环需求推导训练配方

Agent 的训练配方首先是一项责任分配：哪些能力由模型学习，哪些边界由系统保证。把确定且不可违反的规则交给 RL 会浪费样本，也无法提供硬保证；将需要适应任务和当前状态的决策写死在系统中，则会限制泛化能力。模型侧的监督应随不确定性增加，从可复用的前置知识和离线示范转向 On-policy 经验，并把昂贵反馈集中在对结果影响较大的分叉点。下面的流程沿着“系统约束—模型先验—在线优化”展开。

### 第一步：在 Mid-training 中学习接口，由系统守住安全边界

接口知识通常稳定且可以跨任务复用。Tool Schema、动作格式、常见 ACI 模式和环境反馈约定，可以通过合成工具调用、完整交互轨迹和错误恢复示例进入 Mid-training 数据。模型由此提前学会构造请求、读取工具返回并修正无效调用，不必等到 RL 阶段再通过试错掌握 JSON 语法和工具协议。

例如，Coding Agent 可以在 Mid-training 中学会调用文件读取与补丁工具的格式；但它能否写入工作区外的文件、执行部署或读取密钥，必须由 Sandbox 和权限系统决定。前者属于模型能力，后者属于不可交给模型自行遵守的安全边界。

### 第二步：用 Trajectory SFT 建立行为先验

在已有接口能力上，Trajectory SFT 进一步学习任务级 Workflow、状态获取与基本恢复。目标是把策略送入“偶尔能够完成任务”的区域，为后续探索提供起点。

SFT 的指标不应只看 Behavior Cloning 分数。WebAgent-R1 中，Long-CoT BC 的初始分数更高，RL 后结果却更低。过强的确定性模板会压缩 Policy Entropy。好的初始化还要保留探索空间。

### 第三步：让当前策略生成 On-policy 状态

完成 Trajectory SFT 后，让当前 Policy 进入环境并收集它实际访问的状态。这一步的目的把训练分布从教师模型的 $$d_E^{\pi_T}$$ 移到当前策略的 $$d_E^{\pi_\theta}$$。Policy 每次更新都会改变后续的状态分布，因此 Rollout 数据也需要随训练持续刷新。成功轨迹可以进入 Rejection Sampling，失败状态可以交给教师模型纠正或用作恢复训练，高不确定状态则适合优先分支。

### 第四步：用环境 Outcome 锚定真值

Verifier 应结合初始状态、行动轨迹和终局状态，检查目标条件、隐藏副作用、权限违规与评测完整性。这个 Outcome 决定哪些轨迹可以作为正例，也为后续 RL 和过程信号提供真值锚点。

Learned Judge 可以补充可读性、效率和方案质量等软评价，尤其适合在已经完成任务的候选轨迹之间排序。但软分数不能补偿任务失败，也不能取代隐藏测试、State Diff 或环境 Invariant 等硬性指标。

### 第五步：把昂贵反馈投到高价值状态

教师模型调用、Branched Rollout 和 Process Audit 的成本都较高。它们应集中在：

- Policy Entropy 高的决策点；
- 成功与失败轨迹的分叉点；
- 高频失败状态；
- 高风险动作之前；
- Verifier 与 Judge 分歧的位置。

这一步把训练信号设计转化为主动实验设计：由训练系统决定在哪里投入更多成本获取信息量更高的标签。

### 第六步：用 RL 优化终局结果和恢复能力

当 Action Prior 已经让策略能够偶尔成功，环境可以持续生成 On-policy 轨迹，Verifier 也能稳定判断结果时，RL 的闭环才具备运行条件。接口能力已由前面的阶段建立，RL 聚焦当前策略实际访问的状态，比较不同决策，强化高回报路径，并学习如何从自身造成的错误中恢复。正 Reward 几乎不出现时，策略缺少可用的探索信号；Verifier 不可靠时，优化又会优先放大测量漏洞。

在这个闭环中，过程信号负责提高 Credit Resolution 和样本效率，终局 Outcome 负责锚定真实任务目标，Hard Constraint 则继续由系统执行。三者分别解决学习速度、优化方向和安全边界。

完整方案可以写成：

```text
Harness / ACI 约束
→ Trajectory SFT
→ 当前策略 Rollout
→ Verifier 筛选
→ 高价值状态追加反馈与分支
→ Outcome-anchored Agentic RL
→ 多维独立评测
```

---

## 六、不同任务场景下的训练闭环

一个任务是否适合 Agentic RL，取决于训练闭环能否成立：环境状态能否读取和重置，动作后果能否重复采样，Outcome 能否在可接受的成本与延迟内验证。不同任务缺失的环节不同，适合的训练方法也会随之变化。本节用几类代表性任务分析不同场景的闭环瓶颈。

### 6.1 Coding、Terminal、SQL

这些任务具备数字化状态、可执行动作、环境 Reset 和程序 Verifier，最容易形成完整训练闭环。

长轨迹 SFT 可以建立工具使用和任务 Workflow 的行为先验；隐藏测试适合充当 Hard Verifier，用于筛选成功轨迹并提供终局 Reward。持续采样当前策略可以暴露真实失败状态；这些轨迹可作为低回报样本，并用于构造回滚、重试和重新规划的恢复数据。环境吞吐足够时，Execution-grounded RL 可以直接优化可执行结果；维护性、资源成本和代码风格适合作为正确候选之间的次级评价，与正确性分开处理。

主要风险是测试覆盖不足、篡改 Grader、Harness 过拟合和环境启动成本。

### 6.2 Search、Web 与 GUI

Search 的短答案可以验证，长报告还要判断证据支持和来源质量。GUI 模拟器可以检查最终状态，真实网站却难以 Reset，且包含支付、邮件、账户等不可逆动作。

搜索与操作轨迹 SFT 可以先教会查询拆解、页面导航和基本工具使用。对于短答案、引用和明确终态，可以使用事实检查或状态检查提供硬反馈；对于长报告，还需要评价证据覆盖、来源质量和结论是否得到支持。在可重置环境中持续采样当前策略，可以收集导航失败、页面变化和工具报错后的恢复数据。RL 更适合在搜索环境或 GUI 模拟器中进行，真实网站则主要用于带权限控制的评估和少量数据采集。

### 6.3 Memory 与企业工作流

Memory 写入发生时，很难立即判断一条信息是否值得保存。它可能在数百 Turn 后帮助回答查询，也可能逐渐过期、与新信息冲突，或造成隐私风险。企业工作流也有类似延迟：工单状态可以立即更新，业务结果却可能数周后才出现，期间还会受到人工处理和其他系统的影响。

训练系统只能在未来结果出现后，回看哪些早期写入或操作应当获得 Credit。时间跨度、外部干扰和多次状态修改都会增加归因难度，因此这类任务通常需要组合即时约束、中间状态检查与延迟 Outcome。

适合的信号包括：

- 未来查询结果；
- 数据库 State Diff；
- SOP 合规；
- ADD / UPDATE / DELETE / NOOP 操作偏好；
- 隐私与存储预算；
- 高风险动作的人类审批。

### 6.4 形式科学与物理世界

定理证明、数值实验和模拟环境拥有强 Verifier，适合 Rejection Sampling、自进化和 RL。

湿实验与 Robotics 的真实 Rollout 昂贵、缓慢且可能不可逆。示范和离线数据承担训练的主体，RL 主要发生在仿真与风险受限的实机环境中。

### 6.5 开放式主观任务

写作、审美、战略和长期人际任务缺少稳定真值锚点。单一 LLM Judge 会把偏好压成一套可被利用的代理标准。

这类任务更适合高质量 SFT、个体化 Preference、用户编辑反馈和 Pluralistic Reward。它们很难出现类似数学、代码领域的 RLVR 跃迁。

### 6.6 可验证性决定闭环的扩展速度

综合这些场景，训练效率主要取决于环境能否低成本地产生可重复、可验证的经验。Coding、Terminal、SQL、封闭搜索和形式科学可以结合 SFT、Verifier 筛选与 Execution-grounded RL，持续扩充由真实结果校验的训练数据。环境吞吐越高、Verifier 越可靠，这个闭环越容易扩大。

开放写作、审美、社交和创造类任务则缺少可重复的真值测量，仍需依赖个体化 Preference、用户反馈和人类判断。反馈成本更高、标准也会随用户和情境变化，因此很难按照可验证域的方式持续放大 RL 训练。

---

## 结语

Agent 训练的基本单位可以理解为一次可重复实验：Policy 在环境中采取行动，Verifier 读取结果，优化算法根据反馈更新 Policy，更新后的 Policy 再生成下一轮实验。Reward、State 与 Credit 在这个过程中形成连续的因果链。Verifier 的微小偏差会改变哪些轨迹获得高回报，策略随后提高这些轨迹的概率，状态访问分布也跟着移动；新分布进入长尾区域后，稀疏 Outcome 更难解释每个动作的贡献，Credit 误差又会进入下一轮更新。局部误差由此沿闭环逐步放大。

这条链路改变了训练信号的评价方式。静态数据上的准确率只能描述更新前的 Verifier；训练真正依赖它在 Policy 持续变化后的可靠性。更多 Rollout 和更强优化会扩大策略的搜索范围，有效解法和测量漏洞都会获得更多探索机会。因此，增加训练计算的同时，需要同步提高环境的可观测性、Verifier 的抗利用能力和反馈的归因精度。

成熟的训练系统会随着 Policy 迭代持续刷新 Rollout，并把新出现的失败状态纳入训练与 Holdout。环境通过重置、重放和同状态分支提供可比较的经验，Verifier 通过隔离评测和对抗审计保持信号有效。最终需要优化的，是单位环境成本能够产生多少有益的经验。算法决定如何使用这些经验，闭环质量决定计算转化为任务能力的效率。Agent 训练的长期进展，则取决于训练系统能否把持续交互转化为支持可靠更新的经验。

---

## 参考资料

- [FireAct](https://arxiv.org/abs/2310.05915)
- [AgentTuning](https://aclanthology.org/2024.findings-acl.181/)
- [Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/)
- [ETO](https://aclanthology.org/2024.acl-long.409/)
- [AgentGym / AgentEvol](https://aclanthology.org/2025.acl-long.1355/)
- [WebAgent-R1](https://aclanthology.org/2025.emnlp-main.401/)
- [SWE-Master](https://arxiv.org/abs/2602.03411)
- [Orchard](https://arxiv.org/abs/2605.15040)
- [Search-R1](https://arxiv.org/abs/2503.09516)
- [ReTool](https://arxiv.org/abs/2504.11536)
- [ToolRL](https://arxiv.org/abs/2504.13958)
- [SWE-RL](https://arxiv.org/abs/2502.18449)
- [Agent-Diff](https://arxiv.org/abs/2602.11224)
- [Rewarding Progress](https://arxiv.org/abs/2410.08146)
- [AgentPRM](https://arxiv.org/abs/2502.10325)
- [RTMC](https://arxiv.org/abs/2604.11037)
- [TRACE](https://arxiv.org/abs/2607.13988)
- [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502)
- [Harness-Bench](https://arxiv.org/abs/2605.27922)
