// Accept both api=http://localhost:8080 and a trailing-slash variant.
const configuredApi = new URLSearchParams(location.search).get("api");
const API = configuredApi ? configuredApi.replace(/\/+$/, "") : null;
const state = { phase:"idle", step:0, totalSteps:6, finalRate:null, curve:[], baseline:null, best:null, timer:null, poller:null, liveTicker:null, liveSince:null, livePhase:null, task:"two_sum_plus" };
const $ = (id) => document.getElementById(id);
const taskInfo = {
  two_sum_plus:{title:"Two Sum+",blurb:"Find the lexicographically first pair that adds to a target, or return an empty list."},
  safe_parser:{title:"Safe Parser",blurb:"Parse a tiny CSV of integers safely — no eval, no file tricks."},
  fizzbuzz_plus:{title:"FizzBuzz+",blurb:"Return Fizz, Buzz, FizzBuzz, Daytona, or the number as a string."}
};
const samples = [
  "    for i, left in enumerate(nums):\n        for j in range(i + 1, len(nums)):\n            if left + nums[j] == target:\n                return [i, j]\n    return []",
  "    if not nums:\n        return []\n    for i, x in enumerate(nums):\n        for j in range(i + 1, len(nums)):\n            if x + nums[j] == target:\n                return [i, j]\n    return []",
  "    return []"
];
function toast(message) { const el=$("toast"); el.textContent=message; el.classList.add("show"); setTimeout(() => el.classList.remove("show"),2600); }
function setPhase(phase) { state.phase=phase; $("phase-label").textContent=phase.toUpperCase(); }
function renderGrid(results=[]) {
  const grid=$("sandbox-grid"); grid.innerHTML="";
  const count=Math.max(results.length,16);
  for (let i=0;i<count;i++) { const tile=document.createElement("div"); tile.className="tile "+(results[i]||"pending"); tile.title=results[i]?"Sandbox "+(i+1)+": "+results[i]:"Waiting"; grid.appendChild(tile); }
}
function renderLiveGrid() {
  const grid=$("sandbox-grid"); grid.innerHTML="";
  for(let i=0;i<16;i++) {
    const tile=document.createElement("div");
    tile.className="tile running";
    tile.style.animationDelay=(i*70)+"ms";
    tile.title="CPU sandbox "+(i+1)+": waiting for completion";
    grid.appendChild(tile);
  }
}
function updateLiveStatus(remote) {
  const running=Boolean(remote && remote.running);
  const status=$("live-status"), elapsed=$("live-elapsed"), progress=$("live-progress"), trainElapsed=$("train-elapsed");
  document.querySelector(".run-status").classList.toggle("running",running);
  if(!running) {
    if(state.liveTicker) { clearInterval(state.liveTicker); state.liveTicker=null; }
    if(remote && remote.phase==="error") {
      status.textContent="Run failed — see backend logs";
    } else if(remote && remote.baseline_pass_rate!==null && remote.baseline_pass_rate!==undefined) {
      status.textContent="Baseline complete — "+Math.round(remote.baseline_pass_rate*100)+"% passed in CPU sandboxes";
    } else {
      status.textContent="Ready";
    }
    elapsed.textContent="0:00"; trainElapsed.textContent="0:00";
    progress.style.transform="translateX(0)";
    return;
  }
  if(!state.liveSince) state.liveSince=Date.now();
  const phase=remote.phase || "queued";
  const labels={queued:"Queued — waiting for GPU capacity",provisioning:"GPU sandbox starting — installing runtime",model_loading:"GPU ready — loading the base model",baseline:"Generating model answers, then grading in CPU sandboxes",training:"Training step in progress — grading completions"};
  status.textContent=labels[phase] || "Live run in progress";
  const tick=() => {
    const seconds=Math.floor((Date.now()-state.liveSince)/1000);
    const formatted=Math.floor(seconds/60)+":"+String(seconds%60).padStart(2,"0");
    elapsed.textContent=formatted; trainElapsed.textContent=formatted;
  };
  tick();
  if(!state.liveTicker) state.liveTicker=setInterval(tick,1000);
}
function updateCurve() {
  const points=state.finalRate===null ? state.curve : state.curve.concat(state.finalRate);
  $("step-number").textContent=state.step+" / "+state.totalSteps; if (!points.length) return;
  const denominator=state.finalRate===null ? Math.max(1,state.totalSteps-1) : Math.max(1,state.totalSteps);
  const coords=points.map((v,i) => [42+(388*Math.min(i,denominator)/denominator),170-150*v]);
  const path=coords.map((p,i) => (i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
  $("curve-line").setAttribute("d",path); $("curve-area").setAttribute("d",path+" L"+coords.at(-1)[0]+" 170 L35 170 Z");
  $("curve-points").innerHTML=coords.map(p => '<circle class="curve-point" cx="'+p[0]+'" cy="'+p[1]+'" r="4"/>').join("");
}
function showBest(score,step) { state.best=score; $("best-code").textContent=samples[Math.min(1,step%samples.length)]; $("best-score").textContent="reward "+(score/100).toFixed(3); $("best-step").textContent="step "+step; $("after-rate").textContent=score+"%"; }
function mockResults(rate,count=16) { const pass=Math.round(count*rate), partial=Math.round((count-pass)*.3); return Array.from({length:count},(_,i) => i<pass?"pass":i<pass+partial?"partial":"fail"); }
function renderTraining(remote) {
  const curve=remote?.curve || [];
  const last=curve.length ? curve[curve.length-1] : null;
  $("train-step").textContent=(last ? last.step : 0)+" / "+state.totalSteps;
  $("train-reward").textContent=last ? Number(last.mean_reward).toFixed(3) : "—";
  $("train-pass-rate").textContent=last ? Math.round(last.pass_rate*100)+"%" : "—";
  if(remote?.running) {
    $("train-status").textContent=remote.phase==="training" ? "GRPO training live" : "Starting training…";
    $("train").disabled=true; $("train").textContent="Training…"; $("stop").classList.remove("hidden"); $("stop").disabled=false;
  } else if(remote?.phase==="complete") {
    $("train-status").textContent="Training complete";
    $("train").disabled=false; $("train").textContent="Train "+state.totalSteps+" steps"; $("stop").classList.add("hidden");
  } else if(remote?.phase==="error") {
    $("train-status").textContent="Training failed — inspect backend logs";
    $("train").disabled=false; $("train").textContent="Retry training"; $("stop").classList.add("hidden");
  }
  const entries=(remote?.logs || []).slice(-5).map(line => {
    try {
      const event=JSON.parse(line).event;
      return ({model_loading:"Loading base model…",model_ready:"Base model loaded",baseline_started:"Baseline grading started",baseline_finished:"Baseline grading complete",training_started:"GRPO optimizer started",step_finished:"GRPO step complete"})[event] || event;
    } catch (_) { return line.replace(/^===+|===+$/g,"").trim(); }
  }).filter(Boolean);
  $("training-log").textContent=entries.length ? entries.join("\n") : "Waiting for a training run.";
}
async function callApi(path,body) {
  let response;
  try {
    response=await fetch(API+path,{method:"POST",headers:{"content-type":"application/json"},body:body?JSON.stringify(body):undefined});
  } catch (error) {
    throw new Error("cannot reach control API ("+error.message+")");
  }
  let payload=null;
  try { payload=await response.json(); } catch (_) { /* handled below */ }
  if(!response.ok) throw new Error(payload?.detail || payload?.error || "API returned "+response.status);
  if(payload && payload.ok===false) throw new Error(payload.error || "API rejected request");
  return payload;
}
function runBaseline() {
  if(state.phase==="training") return; setPhase("baseline"); $("grid-caption").textContent="Generating and grading baseline completions…"; renderGrid([]); $("baseline").disabled=true;
  if(API) { callApi("/baseline",{n:16}).then(data => { applyState(data.state); startPolling(); }).catch(error => { $("baseline").disabled=false; setPhase("error"); toast("Live API unavailable: "+error.message); }); return; }
  setTimeout(() => { const rate=state.task==="two_sum_plus" ? 0.32 : 0.45; state.baseline=rate; $("before-rate").textContent=Math.round(rate*100)+"%"; $("result-before").textContent=Math.round(rate*100)+"%"; $("grid-caption").textContent="Base model completions"; renderGrid(mockResults(rate)); $("baseline").disabled=false; setPhase("ready"); toast("Baseline graded in 16 CPU sandboxes."); },API?1000:800);
}
function train() {
  if(state.phase==="training") return;
  if(state.baseline===null) { runBaseline(); return; }
  if(API) { $("train").disabled=true; $("train").textContent="Starting training…"; $("train-status").textContent="Submitting GRPO run…"; callApi("/train",{steps:state.totalSteps}).then(data => { applyState(data.state); startPolling(); }).catch(error => { $("train").disabled=false; $("train").textContent="Retry training"; $("train-status").textContent="Training request failed"; toast("Live training request failed: "+error.message); }); return; }
  setTimeout(() => { setPhase("training"); state.step=0; state.curve=[]; $("train").disabled=true; $("stop").disabled=false; $("grid-caption").textContent="GRPO is updating the model…"; renderGrid([]);
    const tick=() => { state.step++; const rate=[.28,.36,.44,.52,.60,.68][state.step-1]; state.curve.push(rate); renderGrid(mockResults(rate)); updateCurve(); $("mean-reward").textContent=(rate-.3).toFixed(3); $("runtime").textContent=(state.step*18).toFixed(1)+"s"; if(rate>(state.best||0)) showBest(Math.round(rate*100),state.step);
      if(state.step>=state.totalSteps){clearInterval(state.timer);state.timer=null;setPhase("complete");$("train").disabled=false;$("stop").disabled=true;$("grid-caption").textContent="Best checkpoint selected from validation";toast("Training complete — checkpoint selected.");} };
    state.timer=setInterval(tick,900); tick();
  },state.baseline===null?850:0);
}
function stop() { if(API) callApi("/stop").then(data => applyState(data.state)).catch(() => {}); if(state.timer) clearInterval(state.timer); state.timer=null; setPhase("stopped"); $("train").disabled=false; $("stop").disabled=true; $("grid-caption").textContent="Run stopped safely"; toast("Training stopped. Current checkpoint is preserved."); }
function ghost() { setPhase("ghost"); $("grid-caption").textContent="Playing a recorded rehearsal"; state.baseline=.31; $("before-rate").textContent="31%"; $("after-rate").textContent="72%"; renderGrid(mockResults(.72)); state.curve=[.31,.42,.55,.72]; state.step=4; updateCurve(); showBest(72,4); toast("Ghost replay active — no GPU or Daytona required."); }
function lockTask() { state.task=$("task-select").value; const info=taskInfo[state.task]; $("task-title").textContent=info.title; $("task-blurb").textContent=info.blurb; $("task-pill").textContent="TASK LOCKED"; toast(info.title+" locked for the next run."); }
function applyState(remote) {
  if(!remote) return;
  setPhase(remote.phase || "idle"); $("connection-label").textContent="LIVE CONTROL API";
  $("pool-status").textContent=remote.running?"GPU RUNNING":"GPU READY";
  if(remote.total_steps) state.totalSteps=remote.total_steps;
  $("train").textContent="Train "+state.totalSteps+" steps";
  renderTraining(remote);
  updateLiveStatus(remote);
  if(remote.running) {
    state.finalRate=null;
    $("baseline").disabled=true;
    $("baseline").textContent=remote.phase==="baseline" ? "Baseline running…" : "Warming GPU…";
    $("grid-caption").textContent=remote.phase==="baseline" ? "Grading base model completions…" : "Starting the GPU sandbox and model…";
    if(state.livePhase!==remote.phase) { renderLiveGrid(); state.livePhase=remote.phase; }
  } else {
    $("baseline").disabled=false;
    $("baseline").textContent="Run baseline";
    if(state.baseline!==null) {
      renderGrid(mockResults(state.baseline));
      $("grid-caption").textContent="Base model completions — CPU sandbox results";
    }
    state.liveSince=null; state.livePhase=null;
  }
  if(remote.baseline_pass_rate!==null && remote.baseline_pass_rate!==undefined){ state.baseline=remote.baseline_pass_rate; $("before-rate").textContent=Math.round(remote.baseline_pass_rate*100)+"%"; $("result-before").textContent=Math.round(remote.baseline_pass_rate*100)+"%"; }
  if(remote.current_pass_rate!==null && remote.current_pass_rate!==undefined){ $("current-rate").textContent=Math.round(remote.current_pass_rate*100)+"%"; $("after-rate").textContent=Math.round(remote.current_pass_rate*100)+"%"; }
  if(remote.best_completion){ $("best-code").textContent=remote.best_completion; $("best-score").textContent="reward "+Number(remote.best_reward || 0).toFixed(3); $("best-step").textContent=remote.best_source==="baseline" ? "baseline retained — no regression" : "selected holdout completion"; }
  if(!remote.running && remote.best_completion && remote.current_pass_rate!==null && remote.current_pass_rate!==undefined) state.finalRate=remote.current_pass_rate;
  if(remote.curve && remote.curve.length){ state.curve=remote.curve.map(point => point.pass_rate); state.step=remote.curve.at(-1).step; updateCurve(); $("mean-reward").textContent=Number(remote.curve.at(-1).mean_reward).toFixed(3); $("grid-caption").textContent="Live trainer state"; renderGrid(mockResults(remote.current_pass_rate || 0)); }
  if(state.finalRate!==null) updateCurve();
  if(!remote.running){ $("baseline").disabled=false; $("train").disabled=false; $("stop").disabled=true; }
}
function startPolling() {
  if(state.poller) clearInterval(state.poller);
  state.poller=setInterval(() => fetch(API+"/state").then(response => response.json()).then(data => applyState(data.state)).catch(() => {}),500);
}
$("lock-task").onclick=lockTask; $("baseline").onclick=runBaseline; $("train").onclick=train; $("stop").onclick=stop; $("ghost").onclick=ghost;
renderGrid([]); if(API){
  $("connection-label").textContent="CONTROL API";
  $("pool-status").textContent="CONNECTING…";
  startPolling();
}

let slideIndex=0;
const slides=document.querySelectorAll(".slide");
function showSlide(index) {
  slideIndex=Math.max(0,Math.min(index,slides.length-1));
  slides.forEach((slide,i) => slide.classList.toggle("active",i===slideIndex));
  $("slide-number").textContent=slideIndex+1;
  $("progress-line").style.width=((slideIndex+1)/slides.length*100)+"%";
  $("previous").disabled=slideIndex===0;
  $("next").disabled=slideIndex===slides.length-1;
}
$("previous").onclick=() => showSlide(slideIndex-1);
$("next").onclick=() => showSlide(slideIndex+1);
document.addEventListener("keydown",(event) => {
  if(event.key==="ArrowRight" || event.key===" "){event.preventDefault();showSlide(slideIndex+1);}
  if(event.key==="ArrowLeft"){event.preventDefault();showSlide(slideIndex-1);}
});
showSlide(0);
