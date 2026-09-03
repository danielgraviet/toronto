(() => {
  const cover = document.getElementById("cover");
  const chrome = document.getElementById("chrome");
  const goLive = document.getElementById("go-live");
  const prev = document.getElementById("prev");
  const next = document.getElementById("next");
  const slideLabel = document.getElementById("slide-label");
  const barFill = document.getElementById("bar-fill");
  const elapsed = document.getElementById("elapsed");
  const approachGrid = document.getElementById("approach-grid");
  const humanMean = document.getElementById("human-mean");
  const policySlide = document.getElementById("policy-slide");
  const policyPrompt = document.getElementById("policy-prompt");
  const policyChart = document.getElementById("policy-chart");
  const policyStatus = document.getElementById("policy-status");
  const policyCaption = document.getElementById("policy-caption");
  const groupReveal = document.getElementById("group-reveal");
  const groupAnswer = document.getElementById("group-answer");
  const slides = Array.from(document.querySelectorAll(".slide"));

  let live = false;
  let index = 0;
  let startedAt = 0;
  let timerId = 0;
  let policyStep = 0;
  let policyPhase = "dist";
  const scores = [0, 0, 0, 0];

  const policyFrames = [
    {
      text: "the quick",
      chosen: "brown",
      options: [
        { tok: "brown", p: 0.48 },
        { tok: "fox", p: 0.18 },
        { tok: "red", p: 0.14 },
        { tok: "lazy", p: 0.12 },
        { tok: "and", p: 0.08 },
      ],
    },
    {
      text: "the quick brown",
      chosen: "fox",
      options: [
        { tok: "fox", p: 0.52 },
        { tok: "dog", p: 0.16 },
        { tok: "cat", p: 0.14 },
        { tok: "bird", p: 0.1 },
        { tok: ",", p: 0.08 },
      ],
    },
    {
      text: "the quick brown fox",
      chosen: "jumped",
      options: [
        { tok: "jumped", p: 0.44 },
        { tok: "jumps", p: 0.2 },
        { tok: "ran", p: 0.16 },
        { tok: "is", p: 0.12 },
        { tok: ".", p: 0.08 },
      ],
    },
    {
      text: "the quick brown fox jumped",
      chosen: "over",
      options: [
        { tok: "over", p: 0.58 },
        { tok: "on", p: 0.14 },
        { tok: "across", p: 0.12 },
        { tok: "up", p: 0.1 },
        { tok: "the", p: 0.06 },
      ],
    },
    {
      text: "the quick brown fox jumped over",
      chosen: "the",
      options: [
        { tok: "the", p: 0.5 },
        { tok: "a", p: 0.22 },
        { tok: "lazy", p: 0.14 },
        { tok: "my", p: 0.08 },
        { tok: "that", p: 0.06 },
      ],
    },
    {
      text: "the quick brown fox jumped over the",
      chosen: "lazy",
      options: [
        { tok: "lazy", p: 0.46 },
        { tok: "sleeping", p: 0.18 },
        { tok: "brown", p: 0.14 },
        { tok: "quick", p: 0.12 },
        { tok: "dog", p: 0.1 },
      ],
    },
    {
      text: "the quick brown fox jumped over the lazy",
      chosen: "dog",
      options: [
        { tok: "dog", p: 0.55 },
        { tok: "cat", p: 0.16 },
        { tok: "hound", p: 0.12 },
        { tok: ".", p: 0.1 },
        { tok: "and", p: 0.07 },
      ],
    },
    {
      text: "the quick brown fox jumped over the lazy dog",
      chosen: null,
      options: [
        { tok: ".", p: 0.42 },
        { tok: "!", p: 0.2 },
        { tok: "\n", p: 0.16 },
        { tok: "and", p: 0.14 },
        { tok: "who", p: 0.08 },
      ],
    },
  ];

  function formatTime(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return `${minutes}:${String(seconds).padStart(2, "0")}`;
  }

  function tick() {
    elapsed.textContent = formatTime(Date.now() - startedAt);
  }

  function onPolicySlide() {
    return slides[index] === policySlide;
  }

  function renderPolicy() {
    const frame = policyFrames[policyStep];
    const maxP = Math.max(...frame.options.map((item) => item.p));
    policyPrompt.innerHTML = `${frame.text}<span class="cursor">▮</span>`;
    policyChart.innerHTML = "";
    frame.options.forEach((item) => {
      const row = document.createElement("div");
      row.className = "chart-row";
      if (policyPhase === "pick" && item.tok === frame.chosen) {
        row.classList.add("picked");
      }
      const height = `${Math.round((item.p / maxP) * 100)}%`;
      const label = item.tok === "\n" ? "↵" : item.tok;
      row.innerHTML = `
        <span class="tok">${label}</span>
        <div class="chart-track"><i style="--h: ${height}"></i></div>
        <span class="pct">${item.p.toFixed(2)}</span>
      `;
      policyChart.appendChild(row);
    });

    if (frame.chosen === null) {
      policyStatus.textContent = "Sentence complete. The policy now sits over what comes next.";
      policyCaption.textContent =
        "Same policy idea at every step. New text means a new state and a new distribution.";
      return;
    }

    if (policyPhase === "dist") {
      policyStatus.textContent = `State: “${frame.text}” · policy over next tokens`;
      policyCaption.textContent =
        "These bar heights are π(token | text). Advance once to sample the top choice.";
    } else {
      policyStatus.textContent = `Sampled “${frame.chosen}”. That action becomes part of the next state.`;
      policyCaption.textContent =
        "Advance again to see the distribution update after the new token.";
    }
  }

  function advancePolicy() {
    const frame = policyFrames[policyStep];
    if (frame.chosen === null) {
      return false;
    }
    if (policyPhase === "dist") {
      policyPhase = "pick";
      renderPolicy();
      return true;
    }
    if (policyStep < policyFrames.length - 1) {
      policyStep += 1;
      policyPhase = "dist";
      renderPolicy();
      return true;
    }
    return false;
  }

  function rewindPolicy() {
    if (policyPhase === "pick") {
      policyPhase = "dist";
      renderPolicy();
      return true;
    }
    if (policyStep > 0) {
      policyStep -= 1;
      policyPhase = "pick";
      renderPolicy();
      return true;
    }
    return false;
  }

  function resetPolicy() {
    policyStep = 0;
    policyPhase = "dist";
    renderPolicy();
  }

  function show(i) {
    index = Math.max(0, Math.min(slides.length - 1, i));
    cover.hidden = true;
    cover.classList.remove("active");
    slides.forEach((slide, slideIndex) => {
      slide.hidden = slideIndex !== index;
    });
    chrome.hidden = false;
    slideLabel.textContent = `${index + 1} / ${slides.length}`;
    barFill.style.width = `${((index + 1) / slides.length) * 100}%`;
    prev.disabled = index === 0 && !(onPolicySlide() && (policyStep > 0 || policyPhase === "pick"));
    next.disabled = index === slides.length - 1;
    if (onPolicySlide()) {
      renderPolicy();
    }
  }

  async function startLive() {
    live = true;
    startedAt = Date.now();
    tick();
    clearInterval(timerId);
    timerId = window.setInterval(tick, 1000);
    resetPolicy();
    groupAnswer.hidden = true;
    groupReveal.hidden = false;
    show(0);
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
      }
    } catch {
      // Fullscreen can fail in some browsers. The deck still works.
    }
  }

  function step(delta) {
    if (!live) return;
    if (delta > 0 && onPolicySlide() && advancePolicy()) {
      prev.disabled = false;
      return;
    }
    if (delta < 0 && onPolicySlide() && rewindPolicy()) {
      prev.disabled = index === 0 && policyStep === 0 && policyPhase === "dist";
      return;
    }
    const nextIndex = index + delta;
    if (nextIndex < 0 || nextIndex >= slides.length) return;
    if (slides[nextIndex] === policySlide) {
      resetPolicy();
    }
    show(nextIndex);
  }

  function renderScores() {
    const cards = approachGrid.querySelectorAll(".approach");
    cards.forEach((card, cardIndex) => {
      const value = scores[cardIndex];
      const label = value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
      card.querySelector(".score-value").textContent = label;
      card.classList.toggle("above", value > 0);
      card.classList.toggle("below", value < 0);
    });
    const mean = scores.reduce((sum, value) => sum + value, 0) / scores.length;
    const meanLabel = mean > 0 ? `+${mean.toFixed(2)}` : mean.toFixed(2);
    humanMean.textContent = `group mean = ${meanLabel} · better than average gets reinforced`;
  }

  approachGrid.addEventListener("click", (event) => {
    const button = event.target.closest(".score-btn");
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    const card = button.closest(".approach");
    const cardIndex = Number(card.dataset.approach);
    const delta = Number(button.dataset.delta) * 0.1;
    scores[cardIndex] = Math.max(-1, Math.min(1, Number((scores[cardIndex] + delta).toFixed(1))));
    renderScores();
  });

  policySlide.addEventListener("click", (event) => {
    if (!live || !onPolicySlide()) return;
    if (event.target.closest("a,button")) return;
    step(1);
  });

  groupReveal.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    groupAnswer.hidden = false;
    groupReveal.hidden = true;
  });

  goLive.addEventListener("click", startLive);
  prev.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));

  window.addEventListener("keydown", (event) => {
    if (!live) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        startLive();
      }
      return;
    }
    if (event.key === "ArrowRight" || event.key === "PageDown" || event.key === " ") {
      event.preventDefault();
      step(1);
    } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      step(-1);
    } else if (event.key === "Home") {
      event.preventDefault();
      resetPolicy();
      show(0);
    } else if (event.key === "End") {
      event.preventDefault();
      show(slides.length - 1);
    }
  });

  renderScores();
  renderPolicy();
})();
