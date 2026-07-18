---
title: "How Agents Learn from Environments and Feedback"
date: 2026-07-17
permalink: /posts/2026/07/agentic-rl-supervision-signals-en/
lang: en
translation_key: agentic-rl-supervision-signals
excerpt: "Understanding the agent training loop through Reward, State, and Credit: Reward sets the direction of learning, State sets where training happens, and Credit decides which actions the outcome is attributed to."
tags:
  - Post-training
  - Agents
  - Reinforcement Learning
---

> **Understanding the agent training loop through Reward, State, and Credit**

## TL;DR

Agent training can be unfolded through three questions — Reward, State, and Credit: how to define the objective using environment results, how to make the training data cover the states the current policy actually visits, and how to attribute a delayed outcome to specific actions. Together, the three determine whether the training loop can produce a reliable, sustainable learning signal.

---

## 1. From answer supervision to interaction trajectories

Imagine a coding agent receives a task: fix an intermittent bug.

It searches the code 12 times, reads 8 files, proposes 3 hypotheses, modifies 2 implementations, runs 6 rounds of tests, and finally makes all hidden tests pass.

From the deployment side, this run appears to have completed the task successfully.

But from the training side, we need to think about many more questions:

- Which search provided the key evidence?
- The first hypothesis was wrong, but did it help rule out a path?
- Of the two modifications, which one fixed the root cause?
- Were the repeatedly-run tests pure waste?
- If the tests passed because of hardcoding or tampering, is the success label still trustworthy?
- If it actually failed, which intermediate steps are still worth keeping?

A traditional text generation usually has only an input and a target output. An agent's actions, however, change the environment, and the changed environment rewrites the subsequent inputs. The training object therefore expands from a single answer into a closed-loop trajectory:

$$
\tau=(o_1,a_1,o_2,a_2,\ldots,o_T,a_T)
$$

where $$o_t$$ is an environment observation and $$a_t$$ is an internal decision or external action. The trajectory distribution is generated jointly by the policy and the environment:

$$
\tau\sim p(\tau\mid \pi,E)
$$

Agent training seeks a policy:

$$
\pi^*
=\arg\max_\pi
\mathbb E_{\tau\sim p(\tau\mid\pi,E)}
[U(\tau)]
$$

This form reveals a key point: the agent's policy simultaneously plays two roles — learner and data generator. Once the policy changes, the states it visits, the observations it obtains, and the mistakes it makes all change.

This is fundamentally different from ordinary supervised learning, which can be approximated as:

$$
x\sim D,\qquad y\sim p(y\mid x)
$$

The data distribution $$D$$ is externally given, and the model parameters do not change the next batch of inputs. An agent's history distribution, by contrast, depends on the current policy:

$$
h_t\sim d_E^\pi
$$

After a training update to $$\pi$$, the $$d_E^\pi$$ sampled in the next round changes with it. While learning how to act, the model continuously manufactures new training distributions.

From the classical RL perspective, this is precisely the policy-induced state-visitation distribution: the policy determines what data gets sampled, and after updating on that data, the new policy produces a new distribution. In such a setting, three classical RL questions stand out especially: how to obtain reliable feedback from a complex environment state, how to obtain training trajectories that cover the current policy's behavior, and how to attribute a delayed outcome to specific decisions in a long trajectory. In practice, these are often summarized as Reward, State, and Credit:

1. **Reward: what exactly should the policy maximize?**  
   Reward turns the task result into an optimizable signal and determines the direction of learning.

2. **State: does the training data cover the states the policy actually reaches?**  
   State concerns which portion of the occupancy distribution the training data covers, and whether teacher-model demonstrations can transfer to the current policy's actual operating distribution.

3. **Credit: which decisions should a delayed outcome be attributed to?**  
   Credit determines which actions in a long trajectory get reinforced and which need correcting.

Along these three perspectives, this piece discusses what information the various training methods — evolving from the arrival of large models into the agent era — each provide, and which part of the problem each one solves.

---

## 2. Reward: what exactly should the policy maximize

Training must first define what is good and what is not.

For ordinary question answering, the ground truth may be a reference answer. For an agent, the result usually shows up as a change in the state of the external environment, for example:

- the bug is fixed, with no regression of existing functionality;
- a ticket in the database moves into the correct state;
- an order on a web page is cancelled;
- a key conclusion in a search report is supported by evidence;
- a robot places an object at the target position;
- memory provides correct and compliant information in a future query.

Therefore, a high-quality reward usually reads the environment state directly. An agent's own description of its process is better used as an auxiliary signal.

### 2.1 The verifier is a measurement instrument

You can think of the verifier as a measurement instrument. It compresses the full environment state into a training signal:

$$
R(\tau)=V(s_T,\tau)
$$

where $$s_T$$ is the final state. A good verifier must answer at least:

- whether the goal state is reached;
- whether undeclared side effects occurred;
- whether safety and permissions were violated;
- whether the agent tampered with tests or the evaluation channel;
- whether the result is reproducible.

Coding is one of the earlier domains suited to directly verifiable rewards. Compilers, unit tests, static analysis, file diffs, and hidden tests can turn program state into machine-checkable feedback. When this feedback is used directly as the reward to update the policy, that is RLVR (Reinforcement Learning with Verifiable Rewards).

### 2.2 A verifiable reward is not necessarily a correct reward

RLVR improves the repeatability and scalability of feedback, but "verifiable" only means the checker can stably execute a set of scoring rules; it does not mean those rules are equivalent to the real task objective.

[SWE-RL](https://arxiv.org/abs/2502.18449) is one example. It computes the training reward from the textual similarity between the predicted patch and the reference patch. A semantically correct but differently-written patch may thus score lower. This checker measures the textual similarity between the patch and the reference answer; it does not check whether the bug is fixed.

Unit tests are closer to program correctness, but still only check behavior that has already been encoded. When coverage is incomplete, hardcoding visible examples can also pass; when the evaluation channel lacks protection, the agent can also modify tests or exploit parser bugs. OpenAI's audit of SWE-Bench Pro found problems in about thirty percent of tasks; [BenchJack](https://arxiv.org/html/2605.12673) found 219 exploitable vulnerabilities across 10 agent benchmarks.

The problem in the first example is that the proxy metric is too far from the real objective; the problem in the second is insufficient test coverage. The specification problem therefore lands on the verifier's objective definition and coverage. When evaluating a measurement, one must simultaneously check whether it can verify stably, what information it preserves, and what information it omits.

### 2.3 The verifier must trade off across several dimensions

Any measurement compresses information:

$$
M:\;(s_0,\tau,s_T)\longrightarrow z
$$

The full environment state, the action trajectory, and the final state are compressed into a binary label, a scalar score, or a preference ordering. Compression is a double-edged sword: the stronger the compression, the more direct and concise the signal and the easier the training — and the more that is omitted.

When choosing a measurement, look at at least five dimensions:

1. **Directness**: how close it is to the real environment state;
2. **Coverage**: how many dimensions of the objective's information it covers;
3. **Repeatability**: whether measuring the same trajectory repeatedly yields similar results;
4. **Cost**: how much program, human, or model-call overhead one measurement per trajectory requires;
5. **Adversarial robustness**: whether the signal stays valid after the policy optimizes against the measurement rules.

Program verifiers often have an advantage in directness, repeatability, and cost, yet may cover only a narrow objective. Human experts can judge broader quality dimensions, with weaker throughput and consistency. LLM judges are easy to scale, and also more prone to inheriting model bias and exploitable patterns. Different approaches suit being combined across different scenarios and different stages of training.

#### Program and environment-state measurement: suited to tasks with a clear final state

Typical forms include:

- exact match;
- compilation and unit tests;
- theorem checkers;
- database state diffs;
- filesystem and business-object state;
- closed-world invariants.

[Search-R1](https://arxiv.org/abs/2503.09516) uses final exact match; [ReTool](https://arxiv.org/abs/2504.11536) uses final-answer equivalence; WebAgent-R1 checks whether a web task is completed; [Agent-Diff](https://arxiv.org/abs/2602.11224) checks the full state difference produced by enterprise API operations and zeroes out any undeclared side effect.

This kind of measurement suits tasks that satisfy three conditions:

```text
the goal state is machine-readable
the consequences of actions can be isolated and recomputed
a valid solution can be decided programmatically
```

Its main risk is insufficient coverage. All-green tests only show that the tested behavior is correct; a correct final state can still hide a violating path behind it. If the agent modified tests, leaked hidden answers, or caused a transient side effect, a simple final-state checker may miss it entirely.

So program measurement should simultaneously check:

- the goal condition;
- undeclared side effects;
- grader integrity;
- key intermediate constraints;
- reproducible execution.

#### Human preference: suited to open-ended output tasks

Tasks such as writing quality, patch maintainability, customer-service communication, and research-idea generation are hard to compress into a deterministic rule. But humans can compare two candidates well:

$$
\tau_A \succ \tau_B
$$

Pairwise preference is usually easier than absolute scoring, because humans are good at comparison but find it hard to stably define a particular score.

Its advantage is broad coverage, absorbing judgments that are hard to formalize. Its costs include:

- human annotation is expensive;
- comparing long trajectories is burdensome;
- annotators' values conflict;
- preferences give only a local ordering, and may not support a unified scalar across tasks;
- the trajectory information annotators see may also be incomplete.

So human preference is better suited to soft quality dimensions such as clarity, maintainability, and communication style. Task correctness, permissions, and safety should still be handled by independent signals.

#### LLM judges and learned reward models: suited to high-throughput soft evaluation

An LLM judge can scale an expert rubric to large-scale trajectory evaluation, and a learned RM can further distill preferences into a low-cost scalar.

This kind of method suits:

- candidate ranking;
- best-of-N;
- rejection sampling;
- soft-quality filtering;
- difficulty estimation;
- process diagnosis.

LLM-as-a-judge has long been controversial. This kind of method faces three kinds of risk.

The first is **static bias**: length, position, style, self-preference, and format all affect the score.

The second is **distribution drift**: the RM is trained on old-policy data, and the new trajectories produced after policy optimization may leave its trustworthy region.

The third is **adversarial exploitation**: [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) found that a single colon or a fixed opening can raise the judge's reward.

So an LLM judge is better used as a high-throughput filter. Periodic human spot-checks, hard-verifier calibration, and off-policy holdouts are all necessary supporting measures.

#### Proxies and heuristics: suited to constraining local behavior

Common proxies include:

- textual similarity;
- format validity;
- number of tool calls;
- output length;
- number of citations;
- token cost.

They are cheap to compute, but are usually just surface features correlated with the task result, and cannot directly indicate whether the task succeeded.

Format validity can be guaranteed directly by a grammar or tool schema; treating it as the main reward encourages the model to chase label correctness. Length and tool count are cost proxies, and optimizing them directly easily produces under-use or padding. Textual similarity suits quick filtering, but is still some distance from semantic correctness.

[ToolRL](https://arxiv.org/abs/2504.13958)'s length experiment shows this risk: directly rewarding response length dropped Qwen-1.5B on BFCL from 46.20% to 33.23%, and a dynamic length reward dropped it further to 28.51%. [ReTool](https://arxiv.org/abs/2504.11536) added no length reward, yet after RL its responses naturally shrank by about 40%. Once tools improve solving efficiency, cost improvements appear alongside task capability.

The most fitting role for a proxy is a gate, a diagnostic metric, or a secondary objective inside a successful trajectory.

### 2.4 Measurement should be a systematic framework

Mature agent measurement is usually a composite structure:

```text
Hard-constraint layer
  permissions, safety, grader integrity — failure means the task fails

Task-outcome layer
  hidden tests, state diff, goal state — defines whether the task is done

Soft-quality layer
  human or LLM preference — compares the quality of successful solutions

Process-diagnosis layer
  turn-level progress, cost, recovery, tool efficiency
```

The different layers of this framework solve different problems. They cannot be linearly weighted into a single reward, because that would let a high task score compensate for a safety violation, and let soft quality mask a task failure.

Training can use a hierarchical objective:

$$
\text{satisfy Constraints first}
\;\rightarrow\;
\text{then maximize Task Success}
\;\rightarrow\;
\text{finally optimize Preference and Cost}
$$

This matches the real structure of tasks and training needs better than "tuning one weight per metric".

### 2.5 Optimization amplifies the verifier's systematic errors

Offline evaluation cares about whether the verifier judges accurately on a fixed data distribution. Once RL begins, the verifier's score directly determines which behaviors get reinforced. As long as some behavior can stably game a high score, the optimization process will raise the probability of it appearing.

For example, a certain fixed opening makes the LLM judge wrongly give a high score. In a static test set, such samples are only 1%, and the verifier still looks 99% accurate. But once RL discovers this pattern, it will reinforce it repeatedly; the originally rare 1% may gradually become the policy's main output. The average accuracy on the original distribution cannot represent reliability during training — the verifier must be re-evaluated on the continuously updated policy distribution.

So a training measurement needs extra checks:

- whether errors are systematic;
- whether the policy can actively trigger errors;
- whether the verifier shares an information channel with the policy;
- whether the evaluation rules are visible to the model;
- whether the holdout updates dynamically with training;
- whether attack strategies transfer to the real environment.

This is also why verifiers need isolation, rotation, and red-team auditing. In agentic RL, the measurement system itself has become part of the environment's safety boundary.

Looking at the whole architecture, one perhaps-effective strategy is: outcome provides ground truth; soft evaluation supplements quality that is hard to encode directly; process signals improve learning efficiency; hard constraints block unacceptable paths. Build the system by combining the strengths of each kind of measurement.

---

## 3. State: does training cover where the policy actually goes

Reward answers "how to evaluate a trajectory", but cannot answer "which trajectories enter training". Even if the verifier is fully reliable, if the data only covers the teacher model's successful paths, the error states the student creates itself still get no supervision.

State here refers to the entire history distribution the policy visits during interaction. Every action an agent takes changes the next input: if the student searches wrong once earlier or calls a wrong tool, the subsequent history may leave the teacher model's trajectory. The places where the model needs to learn to recover will not appear in data generated only by the teacher model.

Therefore, who generates the data is part of the training mechanism. Trajectory SFT learns on the states the teacher model visits; on-policy methods let the training distribution follow the current policy.

### 3.1 Trajectory SFT fits the teacher model's visitation distribution

The demonstrations for trajectory SFT can come from humans or models. The work discussed in this section mainly uses a stronger model to generate trajectories. The teacher model provides demonstrations, but is not assumed to be the optimal policy.

Trajectory SFT can be seen as behavior cloning applied to full agent trajectories. The teacher model first interacts with the environment, then decomposes each decision point in the trajectory into supervised samples:

$$
\tau_T\sim p(\tau\mid\pi_T,E),\qquad
\mathcal D_T=\{(h_t^T,a_t^T)\mid \tau_T\}
$$

where $$h_t^T$$ is the history when the teacher model reaches step $$t$$, and $$a_t^T$$ is the action it gives. The student model fits these actions with standard cross-entropy:

$$
\mathcal L_{\text{SFT}}
=-\mathbb E_{(h_t^T,a_t^T)\sim\mathcal D_T}
\log \pi_\theta(a_t^T\mid h_t^T)
$$

Compared with giving a single outcome only at the end of the trajectory, this supervision is dense: every action of the teacher model provides a token-level label. The task scenarios it suits are mainly:

- tool syntax;
- basic workflows;
- search and read ordering;
- actions in common states;
- error-recovery patterns the teacher model has demonstrated.

[AgentTuning](https://aclanthology.org/2024.findings-acl.181/), [Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/), and [FireAct](https://arxiv.org/abs/2310.05915) all demonstrate the value of trajectory distillation.

Dense labels can still be insufficient in coverage. The training history comes from the teacher model's visitation distribution $$d_E^{\pi_T}$$, while the deployment history comes from the student model's visitation distribution $$d_E^{\pi_\theta}$$. The difference between the two is the covariate shift in imitation learning.

If the student model searches the wrong file at some step, the subsequent context, hypotheses, and environment state all diverge from the teacher model's trajectory. Teacher forcing teaches "how to move at the next step on a history generated by the teacher model", but deployment further requires the student model to learn "how to recover after entering an abnormal state it created itself".

### 3.2 From SFT to RL, the essential difference is the occupancy distribution

Ordering methods by the source of states yields a continuous spectrum:

```text
Teacher-model SFT
  states produced by the teacher model

Policy rollout + verifier filtering
  states produced by the sampling policy, outcome decides keep or rank

On-policy RL
  states and actions both produced by the current policy, outcome provided by the environment
```

The core variable here is the occupancy distribution — which states the policy actually visits. CE, DPO, and policy gradient are just different update tools.

[WebAgent-R1](https://aclanthology.org/2025.emnlp-main.401/) provides a clear thread. Qwen2.5-3B rose from 6.1% to 20.0% via behavior cloning, then to 33.9% via RL. The version that did RL directly from the raw model degraded slightly: the action format was not yet mastered, and positive rewards almost never appeared.

This shows that BC serves as a long-standing exploration prior. It first moves the policy into a region where it "occasionally succeeds", so that RL has a signal to keep optimizing.

### 3.3 Failure trajectories provide multi-layered training signals

Rollouts of the current policy produce both successful samples and real failure states. A trajectory that fails at the end is not suitable as a full positive demonstration, but the states it passed through and the effective actions within it can still be used for training. By the granularity of supervision used, failure trajectories can play a role at four levels:

- **Data filtering and difficulty estimation**: filter out failure samples during positive-trajectory distillation, and estimate task difficulty from the failure rate;
- **Outcome-level supervision**: use them as rejected samples in preference learning; in RL, let them participate in advantage estimation with a low return, lowering the probability of the corresponding actions when below the baseline;
- **Process-level supervision**: identify where the failure occurred, while keeping the effective segments that made progress beforehand;
- **State coverage**: use failure states as starting points for branch exploration, recovery training, or subsequent rollouts.

[SWE-Master](https://arxiv.org/abs/2602.03411) uses failures to judge task difficulty; [ETO](https://aclanthology.org/2024.acl-long.409/) uses failure trajectories as DPO negatives; [Orchard](https://arxiv.org/abs/2605.15040) extracts value-increasing segments from failure trajectories. They respectively exploit the difficulty, outcome, and process information in failure data.

The same failure trajectory can provide a negative signal at the outcome layer, keep effective segments at the process layer, and provide scarce error states at the state layer. A final failure only shows that the overall result is unsatisfactory; it does not show that every action in the trajectory was wrong. Further distinguishing the decisions that should be suppressed from those that should be kept requires solving the credit-assignment problem.

---

## 4. Credit: which decisions should a delayed outcome be attributed to

On-policy data makes training happen on the states the current policy visits, solving the training-distribution problem. But the verifier usually gives only one outcome at the end of the trajectory, which can provide only the overall result of the whole trajectory and lacks information about each step's contribution. Training still needs to turn a trajectory-level result into action-level updates, distinguishing which decisions drove success, which were merely irrelevant steps, and which led to the later failure. This is credit assignment.

Suppose an 80-step trajectory finally succeeds. Giving all actions the same positive advantage reinforces the key decisions, the redundant searches, and the operations that once caused problems, all at once. A failure trajectory may also err only at the last step, and the large amount of correct behavior before it gets pushed down along with it.

Credit assignment needs to estimate:

$$
Q(s_t,a_t)
=\mathbb E[R(\tau)\mid s_t,a_t]
$$

A single trajectory can only tell us that some action co-occurred with success. To estimate an action's causal contribution, the ideal is to try multiple actions in the same state and compare the subsequent results.

### 4.1 Comparing different actions with same-state branching

Forking from the same intermediate state:

```text
state s_t
├── action a_1 → suffix τ_1 → reward 1
├── action a_2 → suffix τ_2 → reward 0
└── action a_3 → suffix τ_3 → reward 0.4
```

Such data is more valuable than three unrelated successful trajectories, because the state is fixed and the action differences are closer to a causal intervention.

In the language of causal inference, this kind of comparison is close to a counterfactual; in agent training and environment engineering, the more intuitive name is branched rollout or same-state branching: fix a shared prefix, change only the action at the branch point, then compare the suffixes.

[RTMC](https://arxiv.org/abs/2604.11037) organizes multiple rollouts of the same task into a tree by shared state, estimating step advantage at branch points, and beats GRPO by 3.2 points on SWE-bench Verified. [C3](https://arxiv.org/abs/2603.06859) freezes the context, replaces the action, and replays a fixed continuation.

Their common premise is that the environment supports snapshot, fork, and replay: snapshot saves the full state at the branch point, fork copies multiple subsequent branches from the same state, and replay is used to reproduce and check execution results. If one can only rerun from the task's starting point, the two trajectories may already differ before reaching the branch point due to random observations or tool return values, and the final reward will mix in the effect of these prefix differences. So the precision of credit assignment is jointly determined by algorithm quality and the environment's ability to provide controlled, reproducible branches.

### 4.2 The correct form of process reward is a progress difference

When the environment cannot branch heavily, one can train a function $$\Phi(s)$$ that estimates the success potential of the current state, and use the change in state as a process signal:

$$
r_t^{\text{process}}
=\Phi(s_{t+1})-\Phi(s_t)
$$

This form has a key property: after summing along the trajectory, the intermediate terms cancel. Inserting redundant steps cannot increase the total reward out of thin air.

[Rewarding Progress](https://arxiv.org/abs/2410.08146) argues that process supervision measures "the change in the probability of a correct solution", and reports an accuracy improvement of over 8% relative to outcome reward. [TRACE](https://arxiv.org/abs/2607.13988) computes turn reward from the log-ratio potential of a frozen reference against the gold answer, also exploiting the same telescoping structure.

Accumulating absolute $$V(s_t)$$ or judge scores step by step produces another kind of incentive: as long as you stay in a high-value state, you can collect points repeatedly. In [AgentPRM](https://arxiv.org/abs/2502.10325)'s experiments, the validation PRM score kept rising while the real success rate dropped from 82% to 70%. This is a typical signal of over-optimized process reward.

Process signals and the final outcome solve problems at different granularities. Outcome provides trajectory-level ground truth but a sparse signal; process signals refine supervision to between adjacent states, but inherit the estimation error of $$\Phi$$. So process signals are better used for advantage shaping, trajectory filtering, or auxiliary supervision, while the final outcome should remain the anchor of task success or failure:

> The process signal answers "how much progress this step brought"; the final outcome judges "whether the task was ultimately completed".

When combining the two, a high process score should not offset a final failure; and final success does not mean every action in the trajectory should be reinforced equally.

### 4.3 The finer the credit, the higher the cost

The potential difference in the previous section refines an episode-level outcome to the turn level, but it still depends on a learned $$\Phi$$. If we also want to judge the contribution of a specific action, we need finer intermediate evaluation, more policy sampling, or the same-state branching introduced earlier. The resolution of credit and the cost of obtaining it thus form a direct trade-off.

By the resolution of attribution to actions, one can roughly order:

```text
Episode outcome
→ Turn-level progress
→ Action-level advantage
→ Same-state branched comparison
```

The later it is, the more the signal can distinguish the contribution of adjacent decisions, but it usually also requires more forwards, more rollouts, or stronger environment capabilities. Same-state branching further requires fixing the starting state and executing multiple candidate actions, at a cost far higher than checking a single final result.

The training system must trade off among three goals: whether the signal is close to the real task result, whether it is dense enough, and whether the acquisition cost is acceptable. Common signals sit at different trade-off points:

- Hard outcome: real and cheap, but sparse;
- Process judge: dense and relatively cheap, but weak in realness;
- Branched-rollout credit: real and dense, but expensive;
- Proxy similarity: dense and cheap, but may measure the wrong objective.

Many new methods over the past year use extra computation to turn a sparse ground truth into a finer signal. When evaluating such a method, beyond final performance, one must also compare how much closer-to-real-result credit resolution is bought back per additional model call or environment rollout.

---

## 5. Deriving a training recipe from the requirements of the loop

An agent's training recipe is first of all an allocation of responsibility: which capabilities the model learns, and which boundaries the system guarantees. Handing deterministic, inviolable rules to RL wastes samples and cannot provide a hard guarantee; hard-coding decisions that need to adapt to the task and the current state into the system limits generalization. Supervision on the model side should shift with increasing uncertainty — from reusable prior knowledge and offline demonstrations toward on-policy experience — and concentrate expensive feedback on the branch points that most affect the result. The pipeline below unfolds along "system constraints — model prior — online optimization".

### Step 1: learn the interface in mid-training, let the system hold the safety boundary

Interface knowledge is usually stable and reusable across tasks. Tool schemas, action formats, common ACI patterns, and environment-feedback conventions can enter mid-training data through synthetic tool calls, full interaction trajectories, and error-recovery examples. The model thereby learns in advance to construct requests, read tool returns, and correct invalid calls, instead of mastering JSON syntax and tool protocols through trial and error only in the RL stage.

For example, a coding agent can learn in mid-training the format for calling file-read and patch tools; but whether it can write files outside the workspace, execute a deployment, or read secrets must be decided by the sandbox and permission system. The former is a model capability; the latter is a safety boundary that cannot be left to the model to observe on its own.

### Step 2: build a behavior prior with trajectory SFT

On top of existing interface capability, trajectory SFT further learns task-level workflows, state acquisition, and basic recovery. The goal is to move the policy into a region where it "occasionally completes the task", providing a starting point for later exploration.

SFT's metric should not be only the behavior-cloning score. In WebAgent-R1, long-CoT BC had a higher initial score yet a lower result after RL. An overly strong deterministic template compresses policy entropy. A good initialization must also preserve exploration space.

### Step 3: let the current policy generate on-policy states

After finishing trajectory SFT, let the current policy enter the environment and collect the states it actually visits. The purpose of this step is to move the training distribution from the teacher model's $$d_E^{\pi_T}$$ to the current policy's $$d_E^{\pi_\theta}$$. Every policy update changes the subsequent state distribution, so the rollout data also needs to be continuously refreshed with training. Successful trajectories can enter rejection sampling, failure states can be handed to the teacher model for correction or used for recovery training, and high-uncertainty states are suited to being prioritized for branching.

### Step 4: anchor ground truth with the environment outcome

The verifier should combine the initial state, the action trajectory, and the final state to check the goal condition, hidden side effects, permission violations, and evaluation integrity. This outcome decides which trajectories can serve as positive examples, and provides a ground-truth anchor for later RL and process signals.

A learned judge can supplement soft evaluations such as readability, efficiency, and solution quality, and is especially suited to ranking among candidate trajectories that have already completed the task. But a soft score cannot compensate for a task failure, nor replace hard metrics such as hidden tests, state diff, or environment invariants.

### Step 5: spend expensive feedback on high-value states

Teacher-model calls, branched rollout, and process audits all have relatively high cost. They should concentrate on:

- decision points with high policy entropy;
- the branch points between successful and failed trajectories;
- high-frequency failure states;
- moments before high-risk actions;
- places where the verifier and the judge disagree.

This step turns training-signal design into active experiment design: the training system decides where to invest more cost to obtain more informative labels.

### Step 6: use RL to optimize the final outcome and recovery ability

Only when the action prior already lets the policy occasionally succeed, the environment can continuously generate on-policy trajectories, and the verifier can stably judge results, does the RL loop have the conditions to run. Interface capability has been built by the earlier stages; RL focuses on the states the current policy actually visits, compares different decisions, reinforces high-return paths, and learns how to recover from errors it created itself. When positive rewards almost never appear, the policy lacks a usable exploration signal; when the verifier is unreliable, optimization instead prioritizes amplifying measurement holes.

In this loop, process signals are responsible for improving credit resolution and sample efficiency, the final outcome is responsible for anchoring the real task objective, and hard constraints continue to be enforced by the system. The three respectively address learning speed, optimization direction, and the safety boundary.

The full scheme can be written as:

```text
Harness / ACI constraints
→ Trajectory SFT
→ Current-policy rollout
→ Verifier filtering
→ Additional feedback and branching at high-value states
→ Outcome-anchored agentic RL
→ Multi-dimensional independent evaluation
```

---

## 6. Training loops across different task scenarios

Whether a task suits agentic RL depends on whether the training loop can hold: whether the environment state can be read and reset, whether the consequences of actions can be sampled repeatedly, and whether the outcome can be verified within acceptable cost and latency. Different tasks lack different links, and the suitable training method changes accordingly. This section analyzes the loop bottleneck in different scenarios through a few representative task classes.

### 6.1 Coding, terminal, SQL

These tasks have digital state, executable actions, environment reset, and program verifiers, and most easily form a complete training loop.

Long-trajectory SFT can build a behavior prior for tool use and task workflows; hidden tests suit acting as a hard verifier, used to filter successful trajectories and provide the final reward. Continuously sampling the current policy can expose real failure states; these trajectories can serve as low-return samples and be used to construct recovery data for rollback, retry, and replanning. When environment throughput is sufficient, execution-grounded RL can directly optimize executable results; maintainability, resource cost, and code style suit being a secondary evaluation among correct candidates, handled separately from correctness.

The main risks are insufficient test coverage, grader tampering, harness overfitting, and environment startup cost.

### 6.2 Search, web, and GUI

Search's short answers can be verified, but a long report also requires judging evidence support and source quality. A GUI simulator can check the final state, but a real website is hard to reset and contains irreversible actions such as payments, emails, and accounts.

Search and operation trajectory SFT can first teach query decomposition, page navigation, and basic tool use. For short answers, citations, and clear final states, one can use fact-checking or state-checking to provide hard feedback; for long reports, one still needs to evaluate evidence coverage, source quality, and whether the conclusion is supported. Continuously sampling the current policy in a resettable environment can collect recovery data after navigation failures, page changes, and tool errors. RL is better done in a search environment or a GUI simulator, while a real website is mainly used for permission-controlled evaluation and a small amount of data collection.

### 6.3 Memory and enterprise workflows

When a memory write happens, it is hard to immediately judge whether a piece of information is worth saving. It may help answer a query hundreds of turns later, or gradually go stale, conflict with new information, or create a privacy risk. Enterprise workflows have a similar delay: a ticket's status can be updated immediately, but the business outcome may appear only weeks later, affected in the meantime by human handling and other systems.

The training system can only look back, after the future result appears, at which early writes or operations should receive credit. The time span, external interference, and multiple state modifications all increase attribution difficulty, so such tasks usually need a combination of immediate constraints, intermediate-state checks, and delayed outcomes.

Suitable signals include:

- future query results;
- database state diff;
- SOP compliance;
- ADD / UPDATE / DELETE / NOOP operation preferences;
- privacy and storage budget;
- human approval for high-risk actions.

### 6.4 Formal science and the physical world

Theorem proving, numerical experiments, and simulated environments have strong verifiers, and suit rejection sampling, self-evolution, and RL.

Real rollouts for wet-lab work and robotics are expensive, slow, and possibly irreversible. Demonstrations and offline data carry the main body of training, and RL mostly happens in simulation and risk-limited real hardware.

### 6.5 Open-ended subjective tasks

Writing, aesthetics, strategy, and long-term interpersonal tasks lack a stable ground-truth anchor. A single LLM judge would compress preference into a set of exploitable proxy standards.

Such tasks suit high-quality SFT, individualized preference, user-edit feedback, and pluralistic reward. They are unlikely to see an RLVR-style jump like the one in math and code.

### 6.6 Verifiability determines how fast the loop scales

Across these scenarios, training efficiency mainly depends on whether the environment can produce reproducible, verifiable experience at low cost. Coding, terminal, SQL, closed search, and formal science can combine SFT, verifier filtering, and execution-grounded RL to continuously expand training data validated by real results. The higher the environment throughput and the more reliable the verifier, the easier this loop is to scale up.

Open-ended writing, aesthetics, social, and creative tasks lack a reproducible ground-truth measurement, and still rely on individualized preference, user feedback, and human judgment. The feedback cost is higher and the standards shift with users and context, so it is hard to scale up RL training the way verifiable domains do.

---

## Conclusion

The basic unit of agent training can be understood as one repeatable experiment: the policy takes actions in the environment, the verifier reads the result, the optimization algorithm updates the policy based on the feedback, and the updated policy then generates the next round of experiments. Reward, State, and Credit form a continuous causal chain in this process. A tiny bias in the verifier changes which trajectories get high return; the policy then raises the probability of those trajectories, and the state-visitation distribution shifts along with it; once the new distribution enters the long tail, a sparse outcome makes it harder to explain each action's contribution, and the credit error re-enters the next round of updates. Local errors thus amplify step by step along the loop.

This chain changes how training signals are evaluated. Accuracy on static data can only describe the verifier before the update; training truly depends on its reliability after the policy keeps changing. More rollouts and stronger optimization expand the policy's search range, and both effective solutions and measurement holes get more chances to be explored. So while increasing training compute, one must simultaneously improve the environment's observability, the verifier's resistance to exploitation, and the attribution precision of feedback.

A mature training system continuously refreshes rollouts as the policy iterates, and incorporates newly appearing failure states into training and holdouts. The environment provides comparable experience through reset, replay, and same-state branching, and the verifier keeps the signal valid through isolated evaluation and adversarial auditing. What ultimately needs to be optimized is how much useful experience a unit of environment cost can produce. The algorithm decides how to use this experience; the quality of the loop decides the efficiency with which compute is converted into task capability. The long-term progress of agent training then depends on whether the training system can turn continuous interaction into experience that supports reliable updates.

---

## References

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
