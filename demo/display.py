"""Rich terminal display for the GRPO demo."""

from __future__ import annotations

import time

from rich import box
from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from demo.config import DEMO_CONFIG
from demo.events import DemoState, apply_event, parse_event
from trainer.progress import format_completion_for_display
from tasks import TaskLoader


class DemoDisplay:
    """RL-framed Rich Live dashboard."""

    def __init__(self) -> None:
        self.state = DemoState(
            task_id=DEMO_CONFIG.task_id,
            model_name=DEMO_CONFIG.model_name,
            total_steps=DEMO_CONFIG.train_steps,
            rollout_size=4,
            rollouts=["pending"] * 4,
        )
        self._started = time.monotonic()
        self._live: Live | None = None

    def start(self) -> Live:
        self._live = Live(self.render(), refresh_per_second=8, transient=False)
        return self._live

    def handle_line(self, line: str) -> None:
        event = parse_event(line)
        if event:
            self.state = apply_event(self.state, event)
        if self._live is not None:
            self._live.update(self.render())

    def render(self) -> RenderableType:
        if self.state.complete:
            return self._render_finale()
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(name="metrics", size=7),
            Layout(self._trajectory_panel(), name="log", ratio=1),
            Layout(self._footer(), name="footer", size=1),
        )
        layout["metrics"].split_row(
            Layout(self._rollout_panel(), name="rollouts", ratio=1),
            Layout(self._curve_panel(), name="curve", ratio=1),
        )
        return layout

    def _elapsed(self) -> str:
        seconds = int(time.monotonic() - self._started)
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes}:{seconds:02d}"

    def _lift_text(self) -> Text | None:
        baseline = self.state.baseline_success_rate
        best = self.state.best_success_rate
        if baseline is None or best is None:
            return None
        delta = best - baseline
        style = "bold green" if delta > 0 else "yellow" if delta == 0 else "red"
        return Text.assemble(
            (f"{baseline:.0%}", "dim"),
            (" → ", "dim"),
            (f"{best:.0%}", style),
            (f"  ({delta:+.0%})", style),
        )

    def _header(self) -> Panel:
        step = self.state.training_step
        total = self.state.total_steps
        mean = "—" if self.state.mean_return is None else f"{self.state.mean_return:.2f}"
        success = "—" if self.state.success_rate is None else f"{self.state.success_rate:.0%}"
        title = Text.assemble(
            ("GRPO", "bold"),
            " · ",
            (self.state.task_id, "cyan"),
            " · ",
            (self.state.model_name, "dim"),
            " · ",
            (f"gen:{self.state.generation_backend}", "yellow"),
        )
        body_parts: list[tuple[str, str]] = [
            (f"step {step}/{total}   ", "bold"),
            ("batch return ", "dim"),
            (mean, "green" if self.state.mean_return and self.state.mean_return > 0 else "white"),
            ("   val success ", "dim"),
            (success, "cyan"),
            ("   ", ""),
            (self._elapsed(), "white"),
        ]
        body = Text.assemble(*body_parts)
        lift = self._lift_text()
        group: list[RenderableType] = [title, body]
        if lift is not None:
            group.append(lift)
        return Panel(
            Group(*group),
            title=self.state.phase_message,
            border_style="red" if self.state.phase == "error" else "blue",
        )

    def _rollout_panel(self) -> Panel:
        size = max(1, min(self.state.rollout_size, 16))
        columns = min(4, size)
        rows = (size + columns - 1) // columns
        table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        for _ in range(columns):
            table.add_column(justify="center", ratio=1)
        rollouts = list(self.state.rollouts[:size])
        while len(rollouts) < size:
            rollouts.append("pending")
        for row in range(rows):
            cells = []
            for col in range(columns):
                index = row * columns + col
                if index >= size:
                    cells.append("")
                    continue
                glyph, style = self._rollout_glyph(rollouts[index])
                cells.append(Text(glyph, style=style))
            table.add_row(*cells)
        return Panel(table, title=f"Rollout batch (n={size})", height=rows + 2)

    def _rollout_glyph(self, state: str) -> tuple[str, str]:
        if state == "positive":
            return "r+", "bold green"
        if state == "negative":
            return "r-", "bold red"
        return "·", "dim"

    def _curve_panel(self) -> Panel:
        lines: list[RenderableType] = []
        if self.state.baseline_success_rate is not None:
            lines.append(self._bar_line("init policy", self.state.baseline_success_rate))
        if self.state.success_rate is not None and self.state.training_step > 0:
            lines.append(
                self._bar_line(f"step {self.state.training_step}", self.state.success_rate)
            )
        if self.state.best_success_rate is not None and self.state.best_step is not None:
            if self.state.best_step != self.state.training_step or self.state.success_rate is None:
                lines.append(
                    self._bar_line(
                        f"best s{self.state.best_step}",
                        self.state.best_success_rate,
                    )
                )
        if self.state.curve:
            rates = [self.state.baseline_success_rate, *[point.pass_rate for point in self.state.curve]]
            rates = [rate for rate in rates if rate is not None]
            if len(rates) > 1:
                lines.append(Text(self._sparkline(rates), style="green"))
            lines.append(Text("validation success", style="dim"))
        elif self.state.baseline_success_rate is not None:
            if self.state.phase == "training" and self.state.training_step == 0:
                backend = self.state.generation_backend
                if backend == "vllm":
                    lines.append(Text("vLLM warming first batch…", style="yellow"))
                else:
                    lines.append(Text("training…", style="dim"))
            else:
                lines.append(Text("training…", style="dim"))
        else:
            lines.append(Text("waiting for returns…", style="dim"))
        if self.state.lift is not None:
            style = "bold green" if self.state.lift > 0 else "yellow"
            lines.append(Text(f"lift {self.state.lift:+.0%} vs init", style=style))
        return Panel(Group(*lines), title="Return curve", height=7)

    def _bar_line(self, label: str, value: float) -> Text:
        width = 10
        filled = round(value * width)
        if value > 0 and filled == 0:
            filled = 1
        filled = max(0, min(width, filled))
        bar = "█" * filled + "░" * (width - filled)
        return Text.assemble((f"{label:<10} ", "dim"), (bar, "cyan"), (f" {value:.0%}", "bold"))

    def _sparkline(self, values: list[float]) -> str:
        if not values:
            return ""
        chars = "▁▂▃▄▅▆▇█"
        low = min(values)
        high = max(values)
        span = high - low or 1.0
        return "".join(
            chars[min(len(chars) - 1, int((value - low) / span * (len(chars) - 1)))]
            for value in values
        )

    def _trajectory_panel(self) -> Panel:
        if not self.state.trajectories and not self.state.best_completion:
            return Panel(Text("Trajectory log will appear here…", style="dim"), title="Trajectory log")
        blocks: list[RenderableType] = []
        for entry in self.state.trajectories[:10]:
            sign = "+" if entry.reward > 0 else ""
            header = Text.assemble(
                ("▸ rollout ", "dim"),
                (f"{entry.index + 1:02d}", "bold"),
                ("   r = ", "dim"),
                (f"{sign}{entry.reward:.2f}", "green" if entry.reward > 0 else "red"),
                (f"   {entry.note}", "cyan"),
            )
            blocks.append(header)
            if entry.preview:
                blocks.append(Syntax(entry.preview, "python", theme="monokai", line_numbers=False))
        if self.state.best_completion and self.state.complete:
            blocks.append(Text("Best trajectory:", style="bold green"))
            blocks.append(
                Syntax(self.state.best_completion, "python", theme="monokai", line_numbers=False)
            )
        return Panel(Group(*blocks), title="Trajectory log")

    def _footer(self) -> Text:
        phases = [
            ("collect", "collect"),
            ("reward", "reward"),
            ("update", "update"),
        ]
        footer = Text()
        for index, (key, label) in enumerate(phases):
            if index:
                footer.append(" → ", style="dim")
            footer.append(label, "bold cyan" if key == self.state.cycle_phase else "dim")
        if self.state.error:
            footer.append(f"   error: {self.state.error}", style="bold red")
        return footer

    def _finale_code(self, raw: str | None) -> str:
        if not raw:
            return "# (no sample captured)"
        task = TaskLoader().load(self.state.task_id)
        return format_completion_for_display(task, raw)

    def _render_finale(self) -> RenderableType:
        elapsed = self._elapsed()
        baseline = self.state.baseline_success_rate
        final = self.state.best_success_rate or self.state.success_rate
        lift = self.state.lift
        holdout = self.state.holdout_pass_rate

        headline = Text.assemble(
            ("✓ ", "bold green"),
            ("Live GRPO training complete", "bold white"),
            (f"  ·  {elapsed}", "dim"),
            (f"  ·  gen:{self.state.generation_backend}", "yellow"),
        )

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="dim", justify="right")
        summary.add_column()
        summary.add_column(style="dim", justify="right")
        summary.add_column()
        steps = self.state.total_steps
        summary.add_row("model", self.state.model_name, "task", self.state.task_id)
        summary.add_row(
            "policy updates",
            f"{steps} GRPO steps",
            "learning rate",
            f"{self.state.learning_rate:.0e}" if self.state.learning_rate else "5e-6",
        )
        summary.add_row(
            "parallel evaluators",
            f"{self.state.pool_size} environments",
            "checkpoint",
            self.state.checkpoint or f"step {self.state.best_step}",
        )
        graded = steps * 4 * 4 + self.state.eval_samples
        summary.add_row("rollouts graded", f"~{graded}+", "method", "reward = test pass rate")
        summary.add_row(
            "GRPO rollouts",
            self.state.generation_backend,
            "eval generate",
            "hf",
        )

        results = Table.grid(padding=(0, 1))
        results.add_column(style="dim")
        results.add_column(justify="right")
        if baseline is not None:
            results.add_row("init policy", f"{baseline:.0%}")
        if final is not None:
            results.add_row("after training", Text(f"{final:.0%}", style="bold green"))
        if lift is not None:
            style = "bold green" if lift > 0 else "yellow"
            results.add_row("validation lift", Text(f"{lift:+.0%}", style=style))
        if holdout is not None:
            results.add_row("holdout eval", f"{holdout:.0%}")

        if self.state.curve and len(self.state.curve) > 1:
            rates = [self.state.baseline_success_rate, *[p.pass_rate for p in self.state.curve]]
            rates = [r for r in rates if r is not None]
            results.add_row("learning curve", Text(self._sparkline(rates), style="green"))

        speed_panel = self._speed_panel()

        top = Columns(
            [
                Panel(summary, title="What we just did", border_style="blue"),
                Panel(results, title="Results", border_style="green"),
            ],
            equal=True,
            expand=True,
        )

        before_code = self._finale_code(self.state.baseline_completion)
        after_code = self._finale_code(self.state.best_completion)
        before_label = "BEFORE · base model"
        after_label = "AFTER · fine-tuned policy"
        if baseline is not None:
            before_label += f"  ({baseline:.0%} validation success)"
        if final is not None:
            after_label += f"  ({final:.0%} validation success)"

        code_compare = Columns(
            [
                Panel(
                    Syntax(before_code, "python", theme="monokai", line_numbers=False),
                    title=before_label,
                    border_style="red",
                ),
                Panel(
                    Syntax(after_code, "python", theme="monokai", line_numbers=False),
                    title=after_label,
                    border_style="green",
                ),
            ],
            equal=True,
            expand=True,
        )

        payoff = Text(
            "Same puzzle. Same unit tests. We sampled code, scored it in parallel, "
            "and updated the policy weights on GPU — live, in minutes.",
            style="italic dim",
        )
        exit_hint = Text("Press Enter to exit", style="bold cyan")

        return Panel(
            Group(headline, Text(""), top, Text(""), speed_panel, Text(""), code_compare, Text(""), payoff, exit_hint),
            title="GRPO · two_sum_plus",
            border_style="green",
            padding=(1, 2),
        )

    def _speed_panel(self) -> Panel:
        if not self.state.timings:
            return Panel(Text("No timing data captured.", style="dim"), title="Speed breakdown")
        table = Table(box=box.SIMPLE, show_header=True, expand=True)
        table.add_column("phase", style="dim")
        table.add_column("seconds", justify="right")
        table.add_column("backend", justify="right")
        order = [
            "baseline_generate",
            "baseline_grade",
            "trainer_init",
            "training",
            "checkpoint_eval",
            "final_eval",
        ]
        seen = set()
        for phase in order:
            if phase in self.state.timings:
                table.add_row(
                    phase,
                    f"{self.state.timings[phase]:.1f}s",
                    self.state.timing_backends.get(phase, "hf"),
                )
                seen.add(phase)
        for phase, seconds in sorted(self.state.timings.items()):
            if phase not in seen:
                table.add_row(
                    phase,
                    f"{seconds:.1f}s",
                    self.state.timing_backends.get(phase, "hf"),
                )
        total = sum(self.state.timings.values())
        table.add_row(Text("total measured", style="bold"), Text(f"{total:.1f}s", style="bold"), "")
        return Panel(table, title="Speed breakdown")

    def wait_for_exit(self) -> None:
        if self._live is not None:
            self._live.update(self.render())
        try:
            input()
        except EOFError:
            pass
