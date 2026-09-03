# Speaker notes · Teach It in Public (15:00)

Talk day path is **HF**: `make demo`. Open the slide deck with:

```bash
uv run python -m http.server 4174 --directory talk
```

Then open http://localhost:4174 and click **Go live**. Keep a second display or window on the CLI.

These notes match the slides in [`talk/index.html`](../talk/index.html). Say the mantras more than once. Prefer short sentences. Cut Q and A before you cut the finale photo.

## Operator cues

**T-30**

- Load `DAYTONA_API_KEY` and optional `HF_TOKEN`.
- Rehearse `make demo` once and confirm a visible before to after lift.
- Enlarge the terminal font.
- Keep a ghost recording ready if the network dies.

**During the talk**

- Around **7:00**, start `make demo` while the Two Sum game is still running.
- Around **10:30**, turn the room to the CLI.
- Do not start the demo cold at minute twelve.

## Mantras (say at least four times)

1. Try, grade, reinforce. You do not need an answer key for every sample.
2. Sandboxes are safe parallel graders.
3. Finetuning can be a few minutes of clear infra plus a reward you can explain in one sentence.
4. Own the loop. Daytona is a reliable place to run the pieces that must stay isolated.

Short GRPO line for the screen moment:

> The model tries several answers. A grader scores them. Training nudges the policy toward the answers that scored better.

---

## 0:00 to 1:30 · Slide 1 · Warm hook

**On screen:** Finetuning does not need a research lab.

**Say:**

Welcome. In the next fifteen minutes we will finetune a weak coding model live. The trainer runs on a Daytona GPU sandbox. The graders are CPU sandboxes that run unit tests.

I want one idea to stick even if you glance at your phone. You can own a real training loop on infrastructure you understand. Finetuning can be this approachable.

We will keep returning to the same loop. Collect answers. Score them. Update the policy.

**Do:** Smile. Point at the room. Do not open the CLI yet.

---

## 1:30 to 3:00 · Slide 2 · RL and Toronto

**On screen:** Reinforcement learning is older than chatbots.

**Say:**

My own path into this work came through reinforcement learning, not through chat products. Agents act. Environments score. Policies improve.

Toronto has a long thread of that research, from games and control into modern model training. You do not need the full history tonight. You only need the shape.

Coding fits that shape. The state is the unfinished program. The action is the next token. The environment is a sandbox that returns a reward.

**Do:** One personal sentence about why you care. Keep it under twenty seconds. Then move.

---

## 3:00 to 5:30 · Slides 3 to 7 · Five words

Spend about twenty to thirty seconds per slide. Point at the visual. Do not lecture.

**Slide 3 · Policy**

On screen: interactive walkthrough starting at `the quick`.

Say: David Silver calls a policy the agent’s behaviour, a map from state to action. In our scope the state is text and the action is the next token. Click through. Watch the bars change after brown, fox, jumped, and the rest.

**Slide 4 · Rollout**

On screen: token chain into a finished `two_sum`.

Say: A rollout is one full trajectory. For us that is one complete code attempt.

**Slide 5 · Reward**

On screen: pass and fail tests totaling 0.67.

Say: Silver says a reward is a scalar feedback signal for how well the agent is doing at that step. Our grader emits that scalar from unit tests.

**Slide 6 · Group**

On screen: four rollouts plus the same-response question.

Say: We score several rollouts for the same prompt relative to each other. Ask the room what happens if all four answers are identical. Use the clues. Reveal if needed. No difference means almost no learning signal.

**Slide 7 · Update**

On screen: before and after bars.

Say: An update changes the policy toward higher reward behaviour. We nudge weights so good code becomes more likely next sample.

**Do:** Skip advantage and loss unless someone asks.

---

## 5:30 to 8:30 · Slide 8 · Human GRPO with Two Sum

**On screen:** Four approaches with live score buttons.

**Participation without a whiteboard:**

1. Ask who has done Two Sum in an interview.
2. Assign volunteers to A hash map, B nested loops, C sort and two pointers, D clever or broken.
3. Give them about one minute to sketch on a laptop.
4. Invite one at a time to AirPlay, HDMI, or Zoom share, or just talk the idea in thirty seconds.
5. While they present, tap + and − on that card. Scores are the group rewards on screen.

**Say:**

We are not starting from one gold solution. We compare the group. Better answers get reinforced. Weaker answers fall away. That is GRPO in a room.

**At ~7:00 (quietly):**

Start `make demo` on the second screen. Do not stare at it. Keep facilitating the four scores.

**Close the game with:**

We are about to do the same thing with a weak model and parallel Daytona sandboxes as graders.

---

## 8:30 to 10:30 · Slides 9 and 10 · Daytona stack and primitive

**Slide 9 · Where Daytona sits**

Say: GPU sandbox trains. CPU sandboxes grade and never run the model. This laptop shows the story and starts the remote job.

If you want projects that go past a thin wrapper around a hosted chat API, you need isolation for untrusted code and a way to scale graders.

**Slide 10 · One use case of a wider primitive**

Say: Tonight is one use case. The same sandbox primitive powers Cursor Cloud Agents running in Daytona while you shut off your laptop. No more clamshelling. Same layer for data jobs and agent civilizations.

**If the CLI still says Starting trainer or Loading policy:**

Stay here or move to the next slide. Do not apologize for waiting. Narrate what the infra is doing.

---

## While the trainer wakes · Slide 11 · Spin-up bridge

Use these lines in order. Skip any that the screen has already passed.

1. Daytona is creating an ephemeral GPU sandbox for the trainer.
2. We load a deliberately weak base policy so the before shot looks honest.
3. CPU graders are warming. Those sandboxes only run tests.
4. Baseline eval uses the same checks we will use after training.
5. When the grid fills, each tile is one rollout with a return from the grader.

**Measured HF phases once rollouts start (rehearsal):**

- baseline generate about 1.9s
- baseline grade about 1.6s
- trainer init about 1.8s
- training about 29.5s
- checkpoint eval about 18.9s
- final eval about 2.3s
- total measured about 56s

The long wait is usually provision and model load before those numbers.

---

## 10:30 to 13:30 · Slide 12 · Live demo

**On screen:** Watch the policy improve on Two Sum Plus. Then look at the CLI.

**Say:**

Watch the init policy bar. That is the untrained policy on Two Sum Plus.

The footer walks collect, reward, and update. Same loop as our four humans.

Each training step samples a group, grades in parallel, and updates weights.

When the finale lands, read the before and after code if it is large enough on the projector.

**Optional aside if someone asks about speed:**

We use Hugging Face generate for training rollouts on talk day. It is the reliable path for this room.

**Do:** Ask one question while bars move. For example, what should the reward punish besides failing tests. Keep answers short.

---

## 13:30 to 15:00 · Slide 13 · Close and questions

**On screen:** What should stick.

**Say:**

Try, grade, reinforce. You do not need an answer key for every sample.

Sandboxes are safe parallel graders for coding rewards.

Finetuning can be a few minutes of clear infra plus a reward you can explain in one sentence. Own the loop. Daytona is a reliable place to run the pieces that must stay isolated.

What questions do you have?

**If you are over time:** Cut questions. Keep the before and after visible for the photo.

---

## Failure lines (keep calm)

| What happens | What you say |
|--------------|--------------|
| Spin-up still running at 10:30 | Stay on the Daytona slides and the bridge list. The wait is the provision story. |
| Train step hangs past about 90s | Stop narration. Switch to ghost replay or a prior finale screenshot. |
| Wi-Fi dies | Ghost mode on the laptop. The story still lands. |
| Lift is weak | Point at the method. Honest before and after still teaches the loop. |
| Nobody wants to share screen | Have them talk the approach for thirty seconds. Score it anyway. |

---

## Luma outcomes (map as you speak)

- How GRPO trains models without traditional labeled answers → human game plus short GRPO line
- Using Daytona sandboxes as safe, parallel code graders → stack slide plus CPU never runs the model
- Designing simple, explainable rewards for coding tasks → reward equals test outcomes you can say in one sentence
- Observing a model improve in real time → CLI window from 10:30 to 13:30
- Wider primitive → Cursor Cloud Agents, analytics, long running experiments on slide 10
