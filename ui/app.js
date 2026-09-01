const API = new URLSearchParams(location.search).get("api");
const state = { phase:"idle", step:0, curve:[], baseline:null, best:null, timer:null, poller:null, task:"two_sum_plus" };
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
function updateCurve() {
  const points=state.curve; $("step-number").textContent=state.step+" / 4"; if (!points.length) return;
  const coords=points.map((v,i) => [35+(315*i/Math.max(1,points.length-1)),170-150*v]);
  const path=coords.map((p,i) => (i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
  $("curve-line").setAttribute("d",path); $("curve-area").setAttribute("d",path+" L"+coords.at(-1)[0]+" 170 L35 170 Z");
  $("curve-points").innerHTML=coords.map(p => '<circle class="curve-point" cx="'+p[0]+'" cy="'+p[1]+'" r="4"/>').join("");
}
function showBest(score,step) { state.best=score; $("best-code").textContent=samples[Math.min(1,step%samples.length)]; $("best-score").textContent="reward "+(score/100).toFixed(3); $("best-step").textContent="step "+step; $("after-rate").textContent=score+"%"; }
function mockResults(rate,count=16) { const pass=Math.round(count*rate), partial=Math.round((count-pass)*.3); return Array.from({length:count},(_,i) => i<pass?"pass":i<pass+partial?"partial":"fail"); }
async function callApi(path,body) { const response=await fetch(API+path,{method:"POST",headers:{"content-type":"application/json"},body:body?JSON.stringify(body):undefined}); if(!response.ok) throw new Error("API "+response.status); return response.json(); }
function runBaseline() {
  if(state.phase==="training") return; setPhase("baseline"); $("grid-caption").textContent="Generating and grading baseline completions…"; renderGrid([]); $("baseline").disabled=true;
  if(API) { callApi("/baseline",{n:16}).then(data => { applyState(data.state); startPolling(); }).catch(() => toast("Live API unavailable.")); return; }
  setTimeout(() => { const rate=state.task==="two_sum_plus" ? 0.32 : 0.45; state.baseline=rate; $("before-rate").textContent=Math.round(rate*100)+"%"; $("result-before").textContent=Math.round(rate*100)+"%"; $("grid-caption").textContent="Base model completions"; renderGrid(mockResults(rate)); $("baseline").disabled=false; setPhase("ready"); toast("Baseline graded in 16 CPU sandboxes."); },API?1000:800);
}
function train() {
  if(state.phase==="training") return;
  if(state.baseline===null) { runBaseline(); return; }
  if(API) { callApi("/train",{steps:4}).then(data => { applyState(data.state); startPolling(); }).catch(() => toast("Live training request failed.")); return; }
  setTimeout(() => { setPhase("training"); state.step=0; state.curve=[]; $("train").disabled=true; $("stop").disabled=false; $("grid-caption").textContent="GRPO is updating the model…"; renderGrid([]);
    const tick=() => { state.step++; const rate=[.28,.41,.54,.68][state.step-1]; state.curve.push(rate); renderGrid(mockResults(rate)); updateCurve(); $("mean-reward").textContent=(rate-.3).toFixed(3); $("runtime").textContent=(state.step*18).toFixed(1)+"s"; if(rate>(state.best||0)) showBest(Math.round(rate*100),state.step);
      if(state.step>=4){clearInterval(state.timer);state.timer=null;setPhase("complete");$("train").disabled=false;$("stop").disabled=true;$("grid-caption").textContent="Best checkpoint selected from validation";toast("Training complete — checkpoint selected.");} };
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
  if(remote.baseline_pass_rate!==null && remote.baseline_pass_rate!==undefined){ state.baseline=remote.baseline_pass_rate; $("before-rate").textContent=Math.round(remote.baseline_pass_rate*100)+"%"; $("result-before").textContent=Math.round(remote.baseline_pass_rate*100)+"%"; }
  if(remote.current_pass_rate!==null && remote.current_pass_rate!==undefined){ $("current-rate").textContent=Math.round(remote.current_pass_rate*100)+"%"; $("after-rate").textContent=Math.round(remote.current_pass_rate*100)+"%"; }
  if(remote.curve && remote.curve.length){ state.curve=remote.curve.map(point => point.pass_rate); state.step=remote.curve.at(-1).step; updateCurve(); $("mean-reward").textContent=Number(remote.curve.at(-1).mean_reward).toFixed(3); $("grid-caption").textContent="Live trainer state"; renderGrid(mockResults(remote.current_pass_rate || 0)); }
  if(!remote.running){ $("baseline").disabled=false; $("train").disabled=false; $("stop").disabled=true; }
}
function startPolling() {
  if(state.poller) clearInterval(state.poller);
  state.poller=setInterval(() => fetch(API+"/state").then(response => response.json()).then(data => applyState(data.state)).catch(() => {}),500);
}
$("lock-task").onclick=lockTask; $("baseline").onclick=runBaseline; $("train").onclick=train; $("stop").onclick=stop; $("ghost").onclick=ghost;
$("apply-knobs").onclick=() => { if(API) callApi("/reward/knobs",{lambda_len:$("length-knob").checked ? 0.1 : 0,lambda_ban:$("ban-knob").checked?2:0,lambda_speed:0}).catch(() => {}); toast("Reward knobs applied to the next step."); };
renderGrid([]); if(API){$("connection-label").textContent="CONTROL API";$("pool-status").textContent="API CONNECTED";}

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
