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

## 一、为什么“给正确答案”不够训练 Agent

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

这个形式揭示了一个关键点：Agent 的策略既是学习者，也是数据生成机制的一部分。一旦策略改变，它访问的状态、获得的观察、犯下的错误都会变化。

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
   State 关注训练数据覆盖哪一段 Occupancy Distribution，以及教师示范能否迁移到当前策略的实际运行分布。

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

Coding 是较早适合使用可验证奖励的领域之一。编译器、单元测试、静态分析、文件 Diff 和隐藏测试能把程序状态转化为机器可检查的反馈。当这些反馈被直接用作 Reward 更新策略时，就是 RLVR（Reinforcement Learning with Verifiable Rewards）。

### 2.2 可验证 Reward 不一定是正确的 Reward

RLVR 提高了反馈的可重复性和扩展性，但“可验证”只表示 Checker 能稳定执行一套评分规则，不表示这套规则等同于真实任务目标。

[SWE-RL](https://arxiv.org/abs/2502.18449) 就是一个例子。它根据预测 Patch 与标准 Patch 的文本相似度计算训练 Reward。一份语义正确但写法不同的 Patch 可能因此得分较低。Checker 验证的是“与标准答案是否相似”，而不是“Bug 是否被修复”。

单元测试离程序正确性更近，但仍只能检查已经编码的行为。覆盖不全时，Hardcode 可见样例也可能通过；评测通道缺少保护时，Agent 还可能修改测试或利用 Parser 漏洞。OpenAI 对 SWE-Bench Pro 的审计发现约三成任务存在问题；[BenchJack](https://arxiv.org/html/2605.12673) 在 10 个 Agent Benchmark 中找到 219 个可利用漏洞。

前一个例子的问题是代理指标离真实目标太远，后一个例子的问题是测试覆盖不足。RLVR 并没有消除 Specification Problem，而是把它转移到了 Verifier 的设计上。因此，评价一种测量方式不能只看它是否可验证，还要比较它保留了哪些信息、遗漏了哪些信息。

### 2.3 Verifier 需要在多个维度之间取舍

任何测量都在压缩信息：

$$
M:\;(s_0,\tau,s_T)\longrightarrow z
$$

完整环境状态、行动轨迹和终局状态，被压成一个二值标签、标量分数或偏好排序。压缩越强，训练越方便，遗漏也越多。

选择测量方式时，至少要看五个维度：

1. **Directness**：它离真实环境状态有多近；
2. **Coverage**：它覆盖了目标的多少侧面；
3. **Repeatability**：同一轨迹重复测量能否得到相近结果；
4. **Cost**：每条轨迹测一次需要多少程序、专家或模型调用；
5. **Adversarial robustness**：策略针对测量规则优化以后，信号还能否保持有效。

程序 Verifier 在 Directness、Repeatability 和 Cost 上常有优势，却可能只覆盖狭窄目标。人类专家能判断更宽的质量维度，吞吐与一致性较弱。LLM Judge 易于扩展，也更容易继承模型偏差和可利用模式。不同仪器适合测量不同性质。

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

它的主要风险是覆盖不足。测试全绿只能说明被测试的行为正确；终态正确也可能掩盖一条违规路径。Agent 修改了测试、泄露了隐藏答案或造成短暂副作用，简单终态 Checker 可能完全看不见。

所以程序测量应同时检查：

- 目标条件；
- 未声明副作用；
- Grader 完整性；
- 关键中间约束；
- 可重复执行。

#### 人类与专家偏好：适合价值宽、答案多的任务

写作质量、Patch 可维护性、客服沟通和研究报告很难压成一个确定规则。人类可以比较两个候选：

$$
\tau_A \succ \tau_B
$$

Pairwise Preference 通常比绝对打分容易，因为人类擅长比较，很难稳定定义“7.3 分的报告”。

它的优势是覆盖宽，能吸收难以形式化的判断。代价包括：

- 专家时间昂贵；
- 长轨迹比较负担大；
- 不同标注者价值观冲突；
- 偏好只能给出局部顺序，未必支持跨任务的统一标量；
- 标注者看到的轨迹信息也可能不完整。

因此，人类偏好更适合处理软质量维度，例如清晰度、维护性和沟通风格。任务正确性、权限与安全仍应由独立硬信号负责。

#### LLM Judge 与 Learned Reward Model：适合高吞吐软评价

LLM Judge 可以把专家 Rubric 扩展到百万级轨迹，Learned RM 则把偏好进一步蒸馏成低成本标量。

这类方法适合：

- 候选排序；
- Best-of-N；
- Rejection Sampling；
- 软质量筛选；
- 难度估计；
- 过程诊断。

它们面临三类风险。

第一类是**静态偏差**：长度、位置、文风、自偏好和格式会影响评分。

第二类是**分布漂移**：RM 在旧策略数据上训练，Policy 优化后产生的新轨迹可能离开它的可信区域。

第三类是**对抗利用**：[One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) 发现单个冒号和固定开头都能提高 Judge Reward。

因此，LLM Judge 更适合充当高吞吐筛选器。定期专家抽检、Hard Verifier 校准和策略外 Holdout 是必要配套。

#### Proxy 与启发式：适合约束局部行为

常见 Proxy 包括：

- 文本相似度；
- 格式合法性；
- 工具调用次数；
- 输出长度；
- 引用数量；
- Token 成本。

它们计算便宜，因果关系通常很弱。

格式合法可以通过 Grammar 或 Tool Schema 直接保证；把它当主要 Reward 会鼓励模型追求标签正确。长度和工具数属于成本代理，直接优化容易出现 Under-use 或 Padding。文本相似度适合快速筛选，与语义正确性之间仍有距离。

[ToolRL](https://arxiv.org/abs/2504.13958) 的长度实验展示了这种风险：直接奖励回复长度让 Qwen-1.5B 在 BFCL 上从 46.20% 降到 33.23%，动态长度奖励进一步降到 28.51%。[ReTool](https://arxiv.org/abs/2504.11536) 没有加入长度 Reward，RL 后回复却自然缩短约 40%。工具提高解题效率后，成本改善会随任务能力一起出现。

Proxy 最合适的角色是 Gate、诊断指标或成功轨迹内部的次级目标。

### 2.4 测量体系应当分工，避免压成单一分数

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

各层回答不同问题。把它们线性加权为单一 Reward，会让高任务分补偿安全违规，也会让软质量掩盖任务失败。

训练时可以使用层级目标：

$$
\text{先满足 Constraints}
\;\rightarrow\;
\text{再最大化 Task Success}
\;\rightarrow\;
\text{最后优化 Preference 与 Cost}
$$

这比“为每项指标调一个权重”更符合真实任务结构。

### 2.5 RL 让测量从统计问题变成对抗问题

离线评测只问一个测量是否和人工判断相关。RL 还会让 Policy 主动搜索测量漏洞。

假设一个 Verifier 在静态数据上有 99% 准确率。剩余 1% 的系统性漏洞可能在随机样本里很少出现；Policy 经过足够多次优化后，会主动集中到这 1% 上。平均准确率很高，最优策略面对的测量却可能完全失效。

因此，训练用测量需要额外检查：

- 错误是否呈系统性；
- Policy 是否能够主动触发错误；
- Verifier 是否和 Policy 共享信息通道；
- 评测规则是否对模型可见；
- Holdout 是否随训练动态更新；
- 攻击策略能否迁移到真实环境。

这也是为什么 Verifier 需要隔离、隐藏、轮换和红队审计。Agentic RL 中，测量系统本身已经成为环境安全边界的一部分。

一个稳健原则是：终局 Outcome 提供真值锚点；软评价补充难以编码的质量；过程信号提高学习效率；硬约束阻断不可接受的路径。每一种测量都在自己擅长的范围内工作。

---

## 三、State：训练是否覆盖策略实际到达的地方

有了可靠 Outcome，下一步是决定在哪些状态上训练。

这正是 Trajectory SFT 与 On-policy 方法的分水岭。

### 3.1 Trajectory SFT 学的是教师分布

专家轨迹 SFT 使用：

$$
\mathcal D=\{(h_t,a_t^*)\}
$$

其中 $$h_t$$ 是专家走到第 $$t$$ 步时的历史，$$a_t^*$$ 是专家动作。优化目标为：

$$
\mathcal L_{\text{SFT}}
=-\mathbb E_{(h_t,a_t^*)\sim\mathcal D}
\log \pi_\theta(a_t^*\mid h_t)
$$

这种 Token-level Label 很密集。它适合教：

- 工具语法；
- 基本 Workflow；
- 搜索与读取顺序；
- 常见状态下的动作；
- 最初的错误恢复模式。

[AgentTuning](https://aclanthology.org/2024.findings-acl.181/)、[Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/) 和 [FireAct](https://arxiv.org/abs/2310.05915) 都证明了轨迹蒸馏的价值。

它的边界也很清楚：训练 History 来自教师策略 $$d_{\pi^*}$$，部署 History 来自学生策略 $$d_{\pi_\theta}$$。

学生在第 3 步搜错一个文件，后面的 Context、假设和环境状态都会偏离教师轨迹。Teacher forcing 提供的是“教师处于理想历史时怎么走”，而部署需要“学生把局面弄乱后怎么恢复”。

### 3.2 从 SFT 到 RL，底层变量是 Occupancy

把方法按状态来源排列，会得到一条连续谱：

```text
专家 SFT
  状态由教师产生

Rejection Sampling
  状态由旧策略产生，只保留成功子集

OEC / DAgger
  状态由学生产生，动作标签由教师提供

On-policy RL
  状态与动作都由当前策略产生，环境提供 Outcome
```

这里的核心变量是 Occupancy Distribution，也就是策略实际访问哪些状态。CE、DPO、Policy Gradient 只是不同的更新工具。

[WebAgent-R1](https://aclanthology.org/2025.emnlp-main.401/) 提供了清晰证据。Qwen2.5-3B 从 6.1% 经 Behavior Cloning 提升到 20.0%，再经 RL 提升到 33.9%。直接从 Raw Model 做 RL 的版本略有退化：动作格式尚未掌握，正 Reward 几乎不出现。

这说明 BC 长期承担 Exploration Prior。它先把策略送进“偶尔能够成功”的区域，RL 才有信号继续优化。

### 3.3 学生状态上的教师纠错

[On-Policy Expert Corrections](https://arxiv.org/abs/2512.14895) 采用了一个很直接的设计：

1. 学生先执行若干步；
2. 在某个状态切换给专家；
3. 专家继承学生造成的 History；
4. 只对专家接管后的动作计算 SFT Loss；
5. 最终轨迹仍由环境 Verifier 检查。

失败前缀的动作没有被当作正例。它的价值在于把专家带到学生真实会遇到的错误状态。

OEC 在 SWE-bench Verified 上，相对传统模仿学习让 7B 和 32B 分别获得约 14% 和 13% 的相对提升。[Revisiting DAgger](https://arxiv.org/abs/2605.12913) 在学生访问的每个状态获取教师标签，4B 从 SFT 的 22.9% 提升到 27.3%。

这也带来一层新问题：学生进入足够异常的状态后，教师本身可能缺乏可靠经验。2026 年的 Guided-OPD、SAGE-OPD 已经开始控制教师何时介入、标签置信度多高。On-policy 缩小了学生分布差距，同时暴露了 Teacher Reliability Shift。

### 3.4 失败轨迹的核心价值是扩展状态覆盖

失败轨迹的用法经历了四步变化：

```text
丢弃
→ 作为负偏好
→ 抽取局部进步
→ 作为恢复训练的起点
```

[SWE-Master](https://arxiv.org/abs/2602.03411) 用失败判断任务难度；[ETO](https://aclanthology.org/2024.acl-long.409/) 将失败轨迹作为 DPO 负例；[Orchard](https://arxiv.org/abs/2605.15040) 抽取失败轨迹中价值上升的片段；OEC 则让专家从学生失败状态开始接管。

失败动作通常不值得模仿，失败状态却非常稀缺。成功轨迹覆盖不到那些状态，恢复、回滚、重规划和求助只能在那里学会。

---

## 四、Credit：延迟结果应该归因给哪些决策

On-policy 数据解决了“在哪训练”，还没有解决“奖励哪一步”。

假设一条 80 步轨迹最终成功。给所有动作同一个正 Advantage，会同时强化关键决策、冗余搜索和曾经造成问题的操作。失败轨迹也可能只在最后一步犯错，前面的大量正确行为会被一并压低。

Credit Assignment 需要估计：

$$
Q(s_t,a_t)
=\mathbb E[R(\tau)\mid s_t,a_t]
$$

单条轨迹只能告诉我们某个动作与成功同时出现。要估计动作的因果贡献，理想情况是在相同状态尝试多个动作，再比较后续结果。

### 4.1 同状态分支是最稀缺的训练数据

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

它们的共同前提是环境支持 Snapshot、Fork 和 Replay。于是 Credit Assignment 与环境基础设施开始合并：算法想要更精确的 Credit，环境必须先提供可比较的同状态分支。

### 4.2 过程奖励的正确形态是进度差

环境无法大量分叉时，可以训练一个函数 $$\Phi(s)$$ 估计当前状态的成功潜力，再用状态变化作为过程信号：

$$
r_t^{\text{process}}
=\Phi(s_{t+1})-\Phi(s_t)
$$

这个形式有一个关键性质：沿轨迹求和后，中间项会抵消。插入冗余步骤无法凭空增加总 Reward。

[Rewarding Progress](https://arxiv.org/abs/2410.08146) 主张过程监督衡量“正确解概率的变化”，并报告相对 Outcome Reward 超过 8% 的准确率提升。[TRACE](https://arxiv.org/abs/2607.13988) 用 Frozen Reference 对 Gold Answer 的 Log-ratio Potential 计算 Turn Reward，也利用了相同的 Telescoping 结构。

绝对 $$V(s_t)$$ 或 Judge 分逐步累加会产生另一种激励：只要停留在高价值状态，就能重复拿分。[AgentPRM](https://arxiv.org/abs/2502.10325) 的实验里，Validation PRM Score 持续上升，真实成功率却从 82% 降到 70%。这是过程 Reward 被过度优化的典型信号。

由此可以得出一条实用设计律：

> 过程信号描述“这一步带来了多少进展”，终局 Outcome 负责判断“任务最终有没有完成”。

### 4.3 越细的 Credit，成本越高

Credit 可以按粒度排成：

```text
Episode
→ Turn
→ Action
→ Branched Rollout
```

粒度越细，通常需要更多 Forward、更多 Rollout 或更强环境能力。

这形成训练信号的一个不可能三角：

- Hard Outcome：真实、廉价，但稀疏；
- Process Judge：稠密、相对便宜，但真实性弱；
- Branched Rollout Credit：真实、稠密，但昂贵；
- Proxy 相似度：稠密、廉价，却可能测错目标。

2025–2026 年许多新方法都在用额外计算，将稀疏真值转化为更细的信号。评估新方法时，可以问一个直接的问题：它花多少计算，换回多少更接近真值的 Credit Resolution？

---

## 五、从闭环需求推导训练配方

现在可以从第一性问题反推出一套训练流程。

### 第一步：把已知规则编译进系统

先用 ACI、Tool Schema、Sandbox 和权限系统消除确定性错误路径。模型无需通过试错学习 JSON 格式，也无需尝试一次真实支付才知道权限边界。

### 第二步：用 Trajectory SFT 建立行为先验

SFT 学习动作语法、常见流程、状态获取与基本恢复。目标是把策略送入“偶尔能够完成任务”的区域，为后续探索提供起点。

SFT 的指标不应只看 Behavior Cloning 分数。WebAgent-R1 中，Long-CoT BC 的初始分数更高，RL 后结果却更低。过强的确定性模板会压缩 Policy Entropy。好的初始化还要保留探索空间。

### 第三步：让当前策略产生真实状态

运行当前 Policy，收集它实际会访问的状态。成功轨迹可做 Rejection Sampling；失败状态可交给教师纠正；高不确定状态可以优先分支。

### 第四步：用环境 Outcome 锚定真值

Verifier 检查终局状态、隐藏副作用与完整性。Learned Judge 可以补充软评价，不能取代硬真值。

### 第五步：把昂贵反馈投到高价值状态

教师调用、Branched Rollout 和 Process Audit 都很贵。它们应集中在：

- Policy Entropy 高的决策点；
- 成功与失败轨迹的分叉点；
- 高频失败状态；
- 高风险动作之前；
- Verifier 与 Judge 分歧的位置。

这一步把训练信号设计转化为主动实验设计：训练系统决定在哪里买一条更昂贵、信息量更高的标签。

### 第六步：用 RL 优化终局结果和恢复能力

当 Action Prior、环境吞吐和 Verifier 都具备以后，On-policy RL 才能稳定工作。过程信号提高效率，终局 Outcome 保持方向，Hard Constraint 保护边界。

完整配方可以写成：

```text
Harness / ACI 约束
→ Trajectory SFT
→ 当前策略 Rollout
→ Verifier 筛选
→ 选择性教师纠错与分支
→ Outcome-anchored Agentic RL
→ 多维独立评测
```

每个任务会选择不同组合，但测量、状态覆盖、因果归因和硬边界始终存在。

---

## 六、什么任务能够形成训练闭环

是否适合 Agentic RL，取决于训练闭环能否成立。

### 6.1 Coding、Terminal、SQL

这些任务具备数字化状态、可执行动作、环境 Reset 和程序 Verifier，最容易形成完整训练闭环。

推荐组合：

```text
长轨迹 SFT
→ 隐藏测试 Rejection Sampling
→ OEC / DAgger 学恢复
→ Execution-grounded RL
→ 成功候选内优化维护性与成本
```

主要风险是测试覆盖不足、篡改 Grader、Harness 过拟合和环境启动成本。

### 6.2 Search、Web 与 GUI

Search 的短答案可以验证，长报告还要判断证据支持和来源质量。GUI 模拟器可以检查最终状态，真实网站却难以 Reset，且包含支付、邮件、账户等不可逆动作。

推荐组合：

```text
搜索 / 操作轨迹 SFT
→ 事实题硬验证
→ OEC 学错误恢复
→ 模拟器内 RL
→ 真实环境只做受控评估
```

### 6.3 Memory 与企业工作流

Memory 的奖励可能在数百 Turn 后才出现；企业流程的真实业务 Outcome 可能延迟数周。两者都有严重的 Hindsight Credit 问题。

适合的信号包括：

- 未来查询结果；
- 数据库 State Diff；
- SOP 合规；
- ADD / UPDATE / DELETE / NOOP 操作偏好；
- 隐私与存储预算；
- 高风险动作的人类审批。

### 6.4 形式科学与物理世界

定理证明、数值实验和模拟环境拥有强 Verifier，适合 Rejection Sampling、自博弈和 RL。

湿实验与 Robotics 的真实 Rollout 昂贵、缓慢且可能不可逆。示范和离线数据承担主体，RL 主要发生在仿真与风险受限的实机环境中。

### 6.5 开放式主观任务

写作、审美、战略和长期人际任务缺少稳定真值锚点。单一 LLM Judge 会把偏好压成一套可被利用的代理标准。

这类任务更适合高质量 SFT、个体化 Preference、用户编辑反馈和 Pluralistic Reward。它们很难出现类似数学、代码领域的 RLVR 跃迁。

---

## 七、几个由框架自然推出的预测

### 7.1 混合训练栈会成为默认方案

单独依赖 SFT、Preference 或 RL，都只能解决训练闭环的一部分。越来越多 Agent 训练会采用组合流程：

```text
Trajectory SFT 建立行为先验
→ Verifier-filtered Rejection Sampling 扩充成功轨迹
→ OEC / DAgger 覆盖学生失败状态
→ Process Signal 提高 Credit 精度
→ Agentic RL 优化终局结果
```

配方设计会围绕几个可测问题展开：当前 Occupancy Gap 有多大，Verifier 可信度多高，Credit 需要细到什么粒度，一次同状态分支采样要花多少成本。SFT、OEC、Preference 和 RL 会成为同一系统内的不同训练算子。

### 7.2 训练数据会从完整轨迹转向同状态动作对照

今天的大部分 Agent 数据以完整轨迹为单位：一条成功轨迹是正例，一条失败轨迹是负例。这样的标签只能说明整段行为的结果，无法区分其中某个 Action 的价值。

更有辨识力的训练样本，是在同一个状态比较多个动作：

$$
(s_t,a_i,G_i),\qquad (s_t,a_j,G_j)
$$

当后续回报 $$G_i>G_j$$ 时，训练系统得到一个条件明确的动作偏好：

$$
a_i\succ a_j\mid s_t
$$

它直接回答 Credit 问题：在相同上下文下，哪个决定带来更好的后果。互联网提供了大量成功文本，却很少提供这种“同一个状态换个动作会怎样”的对照数据；可交互环境可以主动制造它。

生成这类动作对照需要环境支持：

```text
snapshot()
fork()
replay()
inspect_hidden_state()
```

因此，Agent 训练数据的结构会从独立线性 Trace，逐步转向带共享前缀和动作分叉的轨迹图。环境能力服务于信号生成，价值体现在更清晰的动作标签和更低方差的 Credit。

### 7.3 Verifier 会成为安全资产

Agent 会主动寻找奖励漏洞。Verifier 需要隔离、轮换、Hidden Test 与红队审计。[RHB](https://arxiv.org/abs/2605.02964) 已经系统整理了元数据泄漏、篡改、Parser Gaming 等攻击。

未来论文会同时报告任务分数和 Verifier 可信度。缺少 Harness 与 Verifier 披露的 Agent 对比，参考价值会持续下降。

### 7.4 反馈预算会集中到高信息量状态

教师标签、LLM Judge、环境 Rollout 和同状态分支都很昂贵。均匀地给每一步打分，会把大量预算花在模型已经掌握或对结果无影响的状态上。

更有效的做法是把反馈预算投向信息量高的决策点：

- Policy Entropy 高、模型犹豫的位置；
- 成功与失败轨迹首次分叉的位置；
- Verifier 与 Judge 判断不一致的位置；
- 高频失败和重复循环的状态；
- 高风险动作执行之前；
- 教师对标签置信度较低的位置。

这相当于把 Active Learning 引入 Agent 轨迹：训练系统同时学习策略，也学习“下一条昂贵反馈应该买在哪里”。未来训练信号的效率指标，会从标签总量转向单位教师调用、单位 Judge 调用和单位环境 Rollout 带来的性能增益。

### 7.5 可验证域和不可验证域会继续分化

Coding、Terminal、SQL、封闭搜索和形式科学会持续加速。它们的标准配方可能在 2027 年稳定为：

```text
SFT
+ Verifier-filtered Rejection Sampling
+ On-policy Expert Correction
+ Agentic RL
```

开放写作、审美、战略和长期人际任务仍将依赖偏好优化与人类反馈。它们缺少可重复的真值测量，扩张速度会明显慢于可验证域。

---

## 结语

Agent 的训练闭环可以归结为 Reward、State、Credit：

```text
Reward 决定学习方向
State 决定训练分布
Credit 决定信号归属
```

Trajectory SFT 提供程序性先验，On-policy 纠错补齐学生状态，环境 Outcome 锚定真值，过程信号与分支 Rollout 提高 Credit 精度。安全、权限和评估完整性则作为测量体系与 Harness 的工程边界。

这套框架也给出一个清晰判断：Agent 训练的上限越来越取决于闭环反馈的质量。算法决定如何更新参数，环境与 Verifier 决定模型究竟学到了什么。

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
- [On-Policy Expert Corrections](https://arxiv.org/abs/2512.14895)
- [Revisiting DAgger in the Era of LLM-Agents](https://arxiv.org/abs/2605.12913)
- [Guided-OPD](https://arxiv.org/abs/2606.15912)
- [SAGE-OPD](https://arxiv.org/abs/2606.19659)
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
- [RHB](https://arxiv.org/abs/2605.02964)
- [AI Agents That Matter](https://arxiv.org/abs/2407.01502)
- [Harness-Bench](https://arxiv.org/abs/2605.27922)
