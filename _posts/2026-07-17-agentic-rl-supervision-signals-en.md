---
title: "How Agents Learn from Environments and Feedback"
date: 2026-07-17
permalink: /posts/2026/07/agentic-rl-supervision-signals-en/
lang: en
translation_key: agentic-rl-supervision-signals
excerpt: "Understanding the agent training loop through Reward, State, and Credit: Reward sets the direction, State sets where training happens, and Credit decides which actions receive the signal."
tags:
  - Post-training
  - Agents
  - Reinforcement Learning
---

> **Understanding the agent training loop through Reward, State, and Credit**

## TL;DR

Ordinary supervised learning faces a batch of externally given samples; an agent's policy generates its own future states, observations, and data distribution. This endogeneity unfolds agent training along three basic coordinates: Reward determines the direction of learning, State determines where training takes place, and Credit determines which actions a result should be attributed to. The evolution from SFT to RL can be understood as training signals gradually entering the loop in which the model actually operates. The core competition of the future will come down to environments and verifiers: whoever can continuously manufacture experience that is aligned with the current policy, fine-grained enough in signal, and hard to exploit will be able to train longer-horizon agents.

---

## 1. Why "giving the right answer" is not enough to train an agent

Imagine a coding agent receives a task: fix an intermittent concurrency bug.

It searches the code 12 times, reads 8 files, proposes 3 hypotheses, modifies two implementations, runs 6 rounds of tests, and finally makes all hidden tests pass.

The final outcome is clear: success.

But the training problem is far from over:

- Which search provided the key evidence?
- The first hypothesis was wrong, but did it help rule out a path?
- Of the two modifications, which one fixed the root cause?
- Were the 4 repeatedly-run tests pure waste?
- If the tests passed because of hardcoding or tampering, is the success label still trustworthy?
- When the agent fails, which intermediate steps are still worth keeping?

A single text generation usually has only an input and a target output. An agent's actions change the environment, and the changed environment rewrites the subsequent inputs. The training object expands from a single answer into a closed-loop trajectory:

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

This form reveals a key point: the agent's policy is both the learner and part of the data-generating mechanism. Once the policy changes, the states it visits, the observations it obtains, and the mistakes it makes all change.

This is fundamentally different from ordinary supervised learning, which can be approximated as:

$$
x\sim D,\qquad y\sim p(y\mid x)
$$

The data distribution $$D$$ is externally given, and the model parameters do not change the next batch of inputs. An agent's history distribution, by contrast, depends on the current policy:

$$
h_t\sim d_E^\pi
$$

After a training update to $$\pi$$, the $$d_E^\pi$$ sampled in the next round changes with it. The model is learning how to act while continuously manufacturing new training distributions.

This means agent training is a feedback architecture. It has two responsibilities:

- **Measurement**: judging what value a trajectory brought to the environment;
- **Intervention**: providing a learning signal on the states the current policy actually encounters.

Along this loop, training design reduces directly to three questions — Reward, State, and Credit:

1. **Reward: what exactly should the policy maximize?**  
   Reward compresses the real task into an optimization objective, determines the direction of learning, and determines where reward hacking will emerge.

2. **State: does the training data cover the states the policy actually reaches?**  
   State determines which portion of the occupancy distribution training covers, and whether teacher demonstrations can transfer to deployment.

3. **Credit: which decisions should a delayed outcome be attributed to?**  
   Credit determines which actions in a long trajectory get reinforced and which need correcting.

The rest of this piece discusses these three questions in turn. SFT, OEC, RL, process reward, and branched rollout each provide a different information source and training method.

---

## 2. Reward: what exactly should the policy maximize

Training must first define what "getting it done" means.

For ordinary question answering, the ground truth may be a reference answer. For an agent, the result usually shows up as a change in world state:

- The bug is fixed, with no regression of existing functionality;
- A ticket in the database moves into the correct state;
- An order on a web page is cancelled;
- A key conclusion in a search report is supported by evidence;
- A robot places an object at the target position;
- Memory provides correct and compliant information in a future query.

Therefore, a high-quality reward usually reads the environment state directly. An agent's description of its own process is better used as an auxiliary signal.

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

Coding has progressed quickly for exactly this reason. Compilers, unit tests, static analysis, file diffs, and hidden tests together provide a fairly strong measurement system. Web and enterprise workflows can also check URLs, database records, or the state of business objects.

### 2.2 RLVR did not eliminate the objective-definition problem

Programmatic rewards are more stable than a learned reward model, but the checker can still measure the wrong thing.

[SWE-RL](https://arxiv.org/abs/2502.18449) uses the textual similarity between the predicted patch and the reference patch as its training reward. It is cheap and dense, yet it penalizes a semantically equivalent alternative fix. What it measures is "how much it looks like the reference answer"; program correctness is not directly verified.

Tests can also be incomplete. An agent can modify tests, hardcode visible examples, or exploit parser bugs. OpenAI's audit of SWE-Bench Pro found problems in about thirty percent of tasks; [BenchJack](https://arxiv.org/html/2605.12673) found 219 exploitable vulnerabilities across 10 agent benchmarks.

So RLVR pushed the specification problem down to the verifier layer. The reward is deterministic, yet it can still measure the wrong real objective.

### 2.3 There is no single leaderboard for measurement

Any measurement compresses information:

$$
M:\;(s_0,\tau,s_T)\longrightarrow z
$$

The full environment state, the action trajectory, and the final state are compressed into a binary label, a scalar score, or a preference ordering. The stronger the compression, the more convenient the training — and the more that is lost.

When choosing a measurement, look at at least five dimensions:

1. **Directness**: how close it is to the real environment state;
2. **Coverage**: how many facets of the objective it covers;
3. **Repeatability**: whether measuring the same trajectory repeatedly yields similar results;
4. **Cost**: how many program, expert, or model calls one measurement per trajectory requires;
5. **Adversarial robustness**: whether the signal stays valid after the policy optimizes against the measurement rules.

Program verifiers often have an advantage in directness, repeatability, and cost, yet may cover only a narrow objective. Human experts can judge broader quality dimensions, with weaker throughput and consistency. LLM judges are easy to scale, and also more prone to inheriting model bias and exploitable patterns. Different instruments suit measuring different properties.

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

Its main risk is insufficient coverage. All-green tests only show that the tested behavior is correct; a correct final state can still hide a violating path. If the agent modified tests, leaked hidden answers, or caused a transient side effect, a simple final-state checker may see nothing at all.

So program measurement should simultaneously check:

- the goal condition;
- undeclared side effects;
- grader integrity;
- key intermediate constraints;
- reproducible execution.

#### Human and expert preference: suited to tasks with broad value and many answers

Writing quality, patch maintainability, customer-service communication, and research reports are hard to compress into a deterministic rule. Humans can compare two candidates:

$$
\tau_A \succ \tau_B
$$

Pairwise preference is usually easier than absolute scoring, because humans are good at comparison and find it hard to stably define "a 7.3-point report".

Its advantage is broad coverage, absorbing judgments that are hard to formalize. Its costs include:

- expert time is expensive;
- comparing long trajectories is burdensome;
- annotators' values conflict;
- preferences give only a local ordering, and may not support a unified scalar across tasks;
- the trajectory information annotators see may also be incomplete.

So human preference is better suited to soft quality dimensions such as clarity, maintainability, and communication style. Task correctness, permissions, and safety should still be handled by independent hard signals.

#### LLM judges and learned reward models: suited to high-throughput soft evaluation

An LLM judge can scale an expert rubric to millions of trajectories, and a learned RM can further distill preferences into a low-cost scalar.

This kind of method suits:

- candidate ranking;
- best-of-N;
- rejection sampling;
- soft-quality filtering;
- difficulty estimation;
- process diagnosis.

They face three kinds of risk.

The first is **static bias**: length, position, style, self-preference, and format all affect the score.

The second is **distribution drift**: the RM is trained on old-policy data, and the new trajectories produced after policy optimization may leave its trustworthy region.

The third is **adversarial exploitation**: [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794) found that a single colon or a fixed opening can raise the judge's reward.

So an LLM judge is better used as a high-throughput filter. Periodic expert spot-checks, hard-verifier calibration, and off-policy holdouts are necessary companions.

#### Proxies and heuristics: suited to constraining local behavior

Common proxies include:

- textual similarity;
- format validity;
- number of tool calls;
- output length;
- number of citations;
- token cost.

They are cheap to compute, and their causal relationship is usually weak.

Format validity can be guaranteed directly by a grammar or tool schema; treating it as the main reward encourages the model to chase label correctness. Length and tool count are cost proxies, and optimizing them directly easily produces under-use or padding. Textual similarity suits quick filtering, but is still some distance from semantic correctness.

[ToolRL](https://arxiv.org/abs/2504.13958)'s length experiment shows this risk: directly rewarding response length dropped Qwen-1.5B on BFCL from 46.20% to 33.23%, and a dynamic length reward dropped it further to 28.51%. [ReTool](https://arxiv.org/abs/2504.11536) added no length reward, yet after RL its responses naturally shrank by about 40%. Once tools improve solving efficiency, cost improvements appear alongside task capability.

The most fitting role for a proxy is a gate, a diagnostic metric, or a secondary objective inside a successful trajectory.

### 2.4 The measurement system should divide labor and avoid collapsing into a single score

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

Each layer answers a different question. Collapsing them into a single reward via linear weighting lets a high task score compensate for a safety violation, and lets soft quality mask a task failure.

Training can use a hierarchical objective:

$$
\text{satisfy Constraints first}
\;\rightarrow\;
\text{then maximize Task Success}
\;\rightarrow\;
\text{finally optimize Preference and Cost}
$$

This matches the real structure of tasks better than "tuning one weight per metric".

### 2.5 RL turns measurement from a statistical problem into an adversarial one

Offline evaluation only asks whether a measurement correlates with human judgment. RL additionally lets the policy actively search for holes in the measurement.

Suppose a verifier has 99% accuracy on static data. The remaining 1% of systematic holes may rarely appear in a random sample; after enough optimization steps, the policy will actively concentrate on that 1%. The average accuracy is high, yet the measurement facing the optimal policy may fail completely.

So a training measurement needs extra checks:

- whether errors are systematic;
- whether the policy can actively trigger errors;
- whether the verifier shares an information channel with the policy;
- whether the evaluation rules are visible to the model;
- whether the holdout updates dynamically with training;
- whether attack strategies transfer to the real environment.

This is also why verifiers need isolation, hiding, rotation, and red-team auditing. In agentic RL, the measurement system itself has become part of the environment's safety boundary.

A robust principle: the final outcome provides a ground-truth anchor; soft evaluation supplements quality that is hard to encode; process signals improve learning efficiency; hard constraints block unacceptable paths. Each kind of measurement works within the range it is good at.

---

## 3. State: does training cover where the policy actually goes

With a reliable outcome, the next step is to decide which states to train on.

This is precisely the watershed between trajectory SFT and on-policy methods.

### 3.1 Trajectory SFT learns the teacher distribution

Expert-trajectory SFT uses:

$$
\mathcal D=\{(h_t,a_t^*)\}
$$

where $$h_t$$ is the history when the expert reaches step $$t$$, and $$a_t^*$$ is the expert action. The objective is:

$$
\mathcal L_{\text{SFT}}
=-\mathbb E_{(h_t,a_t^*)\sim\mathcal D}
\log \pi_\theta(a_t^*\mid h_t)
$$

This token-level label is dense. It suits teaching:

- tool syntax;
- basic workflows;
- search and read ordering;
- actions in common states;
- initial error-recovery patterns.

[AgentTuning](https://aclanthology.org/2024.findings-acl.181/), [Agent-FLAN](https://aclanthology.org/2024.findings-acl.557/), and [FireAct](https://arxiv.org/abs/2310.05915) all demonstrate the value of trajectory distillation.

Its boundary is also clear: the training history comes from the teacher policy $$d_{\pi^*}$$, while the deployment history comes from the student policy $$d_{\pi_\theta}$$.

If the student searches the wrong file at step 3, the subsequent context, hypotheses, and environment state all diverge from the teacher trajectory. Teacher forcing provides "how the teacher moves when in an ideal history", but deployment needs "how the student recovers after making a mess".

### 3.2 From SFT to RL, the underlying variable is occupancy

Ordering methods by the source of states yields a continuous spectrum:

```text
Expert SFT
  states produced by the teacher

Rejection Sampling
  states produced by the old policy, keeping only the successful subset

OEC / DAgger
  states produced by the student, action labels provided by the teacher

On-policy RL
  states and actions both produced by the current policy, outcome provided by the environment
```

The core variable here is the occupancy distribution — which states the policy actually visits. CE, DPO, and policy gradient are just different update tools.

[WebAgent-R1](https://aclanthology.org/2025.emnlp-main.401/) provides clear evidence. Qwen2.5-3B rose from 6.1% to 20.0% via behavior cloning, then to 33.9% via RL. The version that did RL directly from the raw model degraded slightly: the action format was not yet mastered, and positive rewards almost never appeared.

This shows that BC serves as a long-standing exploration prior. It first moves the policy into a region where it "occasionally succeeds", so that RL has a signal to keep optimizing.

### 3.3 Teacher correction on student states

[On-Policy Expert Corrections](https://arxiv.org/abs/2512.14895) uses a very direct design:

1. the student executes several steps first;
2. at some state, control switches to the expert;
3. the expert inherits the history the student created;
4. SFT loss is computed only on the actions after the expert takes over;
5. the final trajectory is still checked by the environment verifier.

The actions of the failing prefix are not treated as positive examples. Its value lies in bringing the expert to the error states the student actually encounters.

On SWE-bench Verified, OEC gives 7B and 32B models about 14% and 13% relative improvement respectively over traditional imitation learning. [Revisiting DAgger](https://arxiv.org/abs/2605.12913) obtains teacher labels at every state the student visits, raising a 4B model from SFT's 22.9% to 27.3%.

This also brings a new layer of problems: once the student enters sufficiently abnormal states, the teacher itself may lack reliable experience. In 2026, Guided-OPD and SAGE-OPD have begun to control when the teacher intervenes and how confident the label is. On-policy narrows the student's distribution gap while exposing a teacher-reliability shift.

### 3.4 The core value of failure trajectories is expanding state coverage

The use of failure trajectories has gone through four stages:

```text
discard
→ use as negative preference
→ extract local progress
→ use as a starting point for recovery training
```

[SWE-Master](https://arxiv.org/abs/2602.03411) uses failures to judge task difficulty; [ETO](https://aclanthology.org/2024.acl-long.409/) uses failure trajectories as DPO negatives; [Orchard](https://arxiv.org/abs/2605.15040) extracts value-increasing segments from failure trajectories; OEC lets the expert take over from the student's failure states.

Failure actions are usually not worth imitating, but failure states are extremely scarce. Successful trajectories never cover those states; recovery, rollback, replanning, and asking for help can only be learned there.

---

## 4. Credit: which decisions should a delayed outcome be attributed to

On-policy data solves "where to train", but not yet "which step to reward".

Suppose an 80-step trajectory finally succeeds. Giving all actions the same positive advantage reinforces the key decisions, the redundant searches, and the operations that once caused problems, all at once. A failure trajectory may also err only at the last step, and the large amount of correct behavior before it gets pushed down along with it.

Credit assignment needs to estimate:

$$
Q(s_t,a_t)
=\mathbb E[R(\tau)\mid s_t,a_t]
$$

A single trajectory can only tell us that some action co-occurred with success. To estimate an action's causal contribution, the ideal is to try multiple actions in the same state and compare the subsequent results.

### 4.1 Same-state branching is the scarcest training data

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

Their common premise is that the environment supports snapshot, fork, and replay. So credit assignment and environment infrastructure begin to merge: the algorithm wants more precise credit, and the environment must first provide comparable same-state branches.

### 4.2 The correct form of process reward is a progress difference

When the environment cannot branch heavily, one can train a function $$\Phi(s)$$ that estimates the success potential of the current state, and use the change in state as a process signal:

$$
r_t^{\text{process}}
=\Phi(s_{t+1})-\Phi(s_t)
$$

This form has a key property: after summing along the trajectory, the intermediate terms cancel. Inserting redundant steps cannot increase the total reward out of thin air.

[Rewarding Progress](https://arxiv.org/abs/2410.08146) argues that process supervision measures "the change in the probability of a correct solution", and reports an accuracy improvement of over 8% relative to outcome reward. [TRACE](https://arxiv.org/abs/2607.13988) computes turn reward from the log-ratio potential of a frozen reference against the gold answer, also exploiting the same telescoping structure.

Accumulating absolute $$V(s_t)$$ or judge scores step by step produces another kind of incentive: as long as you stay in a high-value state, you can collect points repeatedly. In [AgentPRM](https://arxiv.org/abs/2502.10325)'s experiments, the validation PRM score kept rising while the real success rate dropped from 82% to 70%. This is a typical signal of over-optimized process reward.

From this we get a practical design law:

> The process signal describes "how much progress this step brought"; the final outcome is responsible for judging "whether the task was ultimately completed".

### 4.3 The finer the credit, the higher the cost

Credit can be arranged by granularity:

```text
Episode
→ Turn
→ Action
→ Branched Rollout
```

The finer the granularity, the more forwards, more rollouts, or stronger environment capabilities it usually requires.

This forms an impossible triangle for training signals:

- Hard outcome: real and cheap, but sparse;
- Process judge: dense and relatively cheap, but weak in realness;
- Branched-rollout credit: real and dense, but expensive;
- Proxy similarity: dense and cheap, but may measure the wrong objective.

Many new methods in 2025–2026 use extra computation to turn a sparse ground truth into a finer signal. When evaluating a new method, you can ask a direct question: how much compute does it spend, and how much closer-to-truth credit resolution does it buy back?

---

## 5. Deriving a training recipe from the requirements of the loop

We can now work backward from first-principles questions to a training pipeline.

### Step 1: compile known rules into the system

First use ACI, tool schemas, sandboxes, and permission systems to eliminate deterministic error paths. The model need not learn JSON format through trial and error, nor attempt one real payment to discover a permission boundary.

### Step 2: build a behavior prior with trajectory SFT

SFT learns action syntax, common workflows, state acquisition, and basic recovery. The goal is to move the policy into a region where it "occasionally completes the task", providing a starting point for later exploration.

SFT's metric should not be only the behavior-cloning score. In WebAgent-R1, long-CoT BC had a higher initial score yet a lower result after RL. An overly strong deterministic template compresses policy entropy. A good initialization must also preserve exploration space.

### Step 3: let the current policy produce real states

Run the current policy and collect the states it actually visits. Successful trajectories can go to rejection sampling; failure states can be handed to the teacher for correction; high-uncertainty states can be prioritized for branching.

### Step 4: anchor ground truth with the environment outcome

The verifier checks the final state, hidden side effects, and integrity. A learned judge can supplement soft evaluation, but cannot replace hard ground truth.

### Step 5: spend expensive feedback on high-value states

Teacher calls, branched rollout, and process audits are all expensive. They should concentrate on:

- decision points with high policy entropy;
- the branch points between successful and failed trajectories;
- high-frequency failure states;
- moments before high-risk actions;
- places where the verifier and the judge disagree.

This step turns training-signal design into active experiment design: the training system decides where to buy a more expensive, more informative label.

### Step 6: use RL to optimize the final outcome and recovery ability

Only once the action prior, environment throughput, and verifier are all in place can on-policy RL work stably. Process signals improve efficiency, the final outcome keeps the direction, and hard constraints protect the boundary.

The full recipe can be written as:

```text
Harness / ACI constraints
→ Trajectory SFT
→ Current-policy rollout
→ Verifier filtering
→ Selective teacher correction and branching
→ Outcome-anchored agentic RL
→ Multi-dimensional independent evaluation
```

Each task chooses a different combination, but measurement, state coverage, causal attribution, and hard boundaries are always present.

---

## 6. Which tasks can form a training loop

Whether a task suits agentic RL depends on whether the training loop can hold.

### 6.1 Coding, terminal, SQL

These tasks have digital state, executable actions, environment reset, and program verifiers, and most easily form a complete training loop.

Recommended combination:

```text
Long-trajectory SFT
→ hidden-test rejection sampling
→ OEC / DAgger to learn recovery
→ execution-grounded RL
→ optimize maintainability and cost within successful candidates
```

The main risks are insufficient test coverage, grader tampering, harness overfitting, and environment startup cost.

### 6.2 Search, web, and GUI

Search's short answers can be verified, but a long report also requires judging evidence support and source quality. A GUI simulator can check the final state, but a real website is hard to reset and contains irreversible actions such as payments, emails, and accounts.

Recommended combination:

```text
Search / operation trajectory SFT
→ hard verification for factual questions
→ OEC to learn error recovery
→ RL inside the simulator
→ controlled evaluation only in the real environment
```

### 6.3 Memory and enterprise workflows

Memory's reward may appear only after hundreds of turns; the real business outcome of an enterprise workflow may be delayed by weeks. Both have severe hindsight-credit problems.

Suitable signals include:

- future query results;
- database state diff;
- SOP compliance;
- ADD / UPDATE / DELETE / NOOP operation preferences;
- privacy and storage budget;
- human approval for high-risk actions.

### 6.4 Formal science and the physical world

Theorem proving, numerical experiments, and simulated environments have strong verifiers, and suit rejection sampling, self-play, and RL.

Real rollouts for wet-lab work and robotics are expensive, slow, and possibly irreversible. Demonstrations and offline data carry the main load, and RL mostly happens in simulation and risk-limited real hardware.

### 6.5 Open-ended subjective tasks

Writing, aesthetics, strategy, and long-term interpersonal tasks lack a stable ground-truth anchor. A single LLM judge would compress preference into a set of exploitable proxy standards.

Such tasks suit high-quality SFT, individualized preference, user-edit feedback, and pluralistic reward. They are unlikely to see an RLVR-style jump like the one in math and code.

---

## 7. A few predictions that follow naturally from the framework

### 7.1 Hybrid training stacks will become the default

Relying on SFT, preference, or RL alone can only solve part of the training loop. More and more agent training will adopt a combined pipeline:

```text
Trajectory SFT builds a behavior prior
→ verifier-filtered rejection sampling expands successful trajectories
→ OEC / DAgger covers student failure states
→ process signals improve credit precision
→ agentic RL optimizes the final outcome
```

Recipe design will revolve around a few measurable questions: how large the current occupancy gap is, how trustworthy the verifier is, how fine the credit needs to be, and how much one same-state branch sample costs. SFT, OEC, preference, and RL will become different training operators inside the same system.

### 7.2 Training data will shift from full trajectories to same-state action comparisons

Today most agent data is organized in units of full trajectories: a successful trajectory is a positive example, a failed one a negative example. Such labels can only tell the result of an entire span of behavior, not distinguish the value of a particular action within it.

A more discriminative training sample compares multiple actions in the same state:

$$
(s_t,a_i,G_i),\qquad (s_t,a_j,G_j)
$$

When the subsequent return $$G_i>G_j$$, the training system gets a clearly conditioned action preference:

$$
a_i\succ a_j\mid s_t
$$

It answers the credit question directly: in the same context, which decision led to better consequences. The internet provides plenty of successful text but rarely this kind of "what if the same state took a different action" comparison data; an interactive environment can actively manufacture it.

Generating this kind of action comparison requires environment support:

```text
snapshot()
fork()
replay()
inspect_hidden_state()
```

So the structure of agent training data will gradually shift from independent linear traces to trajectory graphs with shared prefixes and action branches. Environment capabilities serve signal generation, and their value shows up as clearer action labels and lower-variance credit.

### 7.3 Verifiers will become a safety asset

Agents will actively search for reward holes. Verifiers need isolation, rotation, hidden tests, and red-team auditing. [RHB](https://arxiv.org/abs/2605.02964) has already systematically catalogued attacks such as metadata leakage, tampering, and parser gaming.

Future papers will report both the task score and verifier trustworthiness. Agent comparisons that lack harness and verifier disclosure will keep losing reference value.

### 7.4 The feedback budget will concentrate on high-information states

Teacher labels, LLM judges, environment rollouts, and same-state branching are all expensive. Scoring every step uniformly spends a large budget on states the model already masters or that do not affect the outcome.

A more effective approach directs the feedback budget toward high-information decision points:

- positions with high policy entropy where the model hesitates;
- the position where successful and failed trajectories first diverge;
- positions where the verifier and the judge disagree;
- high-frequency failure and repeated-loop states;
- moments before a high-risk action executes;
- positions where the teacher's label confidence is low.

This amounts to introducing active learning into agent trajectories: the training system learns the policy while also learning "where the next expensive feedback should be bought". The future efficiency metric for training signals will shift from total label count to the performance gain per teacher call, per judge call, and per environment rollout.

### 7.5 Verifiable and non-verifiable domains will keep diverging

Coding, terminal, SQL, closed search, and formal science will keep accelerating. Their standard recipe may stabilize by 2027 as:

```text
SFT
+ verifier-filtered rejection sampling
+ on-policy expert correction
+ agentic RL
```

Open-ended writing, aesthetics, strategy, and long-term interpersonal tasks will still rely on preference optimization and human feedback. They lack a reproducible ground-truth measurement, and will expand noticeably slower than verifiable domains.

---

## Conclusion

The agent training loop reduces to Reward, State, and Credit:

```text
Reward determines the direction of learning
State determines the training distribution
Credit determines the attribution of the signal
```

Trajectory SFT provides a procedural prior, on-policy correction fills in student states, the environment outcome anchors ground truth, and process signals and branched rollout improve credit precision. Safety, permissions, and evaluation integrity act as the engineering boundary of the measurement system and the harness.

This framework also gives a clear judgment: the ceiling of agent training increasingly depends on the quality of closed-loop feedback. The algorithm decides how to update parameters; the environment and the verifier decide what the model actually learns.

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
