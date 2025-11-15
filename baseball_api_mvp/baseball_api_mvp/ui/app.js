const $ = id => document.getElementById(id);

function setLoading(on){
  $('loading').style.display = on ? 'inline-block' : 'none';
  $('predict').disabled = on;
}

function fmtPct(v){
  if (v === undefined || v === null) return '-';
  const p = Math.round(10000 * Number(v)) / 100;
  return p.toFixed(2) + '%';
}

async function doPredict(){
  const game_id = $('game_id').value;
  const inning = parseInt($('inning').value || '1', 10);
  const outs = parseInt($('outs').value || '0', 10);
  const bases = $('bases').value.split(',').map(s=>parseInt(s.trim()||'0',10));
  const batter_id = $('batter_id').value;
  const pitcher_id = $('pitcher_id').value;

  const payload = { game_id, inning, outs, bases, batter_id, pitcher_id };
  setLoading(true);
  try{
    const resp = await fetch('/predict/matchup', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    if(!resp.ok){
      const txt = await resp.text();
      $('insights').textContent = `Error: ${resp.status} ${txt}`;
      $('jsonOut').textContent = txt;
      return;
    }
    const data = await resp.json();
    // update bars
    const hit = Number(data.hit_prob || 0);
    const k = Number(data.k_prob || 0);
    const w = Number(data.walk_prob || 0);
    $('hitBar').style.width = Math.min(100, Math.round(hit*10000)/100) + '%';
    $('kBar').style.width = Math.min(100, Math.round(k*10000)/100) + '%';
    $('wBar').style.width = Math.min(100, Math.round(w*10000)/100) + '%';
    $('hitPct').textContent = fmtPct(hit);
    $('kPct').textContent = fmtPct(k);
    $('wPct').textContent = fmtPct(w);

    // insights
    if(data.batter_strategy && Object.keys(data.batter_strategy).length){
      $('insights').textContent = data.batter_strategy.advice || JSON.stringify(data.batter_strategy);
    } else if(data.explanation){
      $('insights').textContent = data.explanation.join(' | ');
    } else {
      $('insights').textContent = 'No insights available';
    }
    $('jsonOut').textContent = JSON.stringify(data, null, 2);
    // model status
    if(data.explanation && data.explanation.includes('Model-based')){
      $('modelStatus').textContent = 'model: loaded';
    } else if(data.explanation && String(data.explanation).toLowerCase().includes('mock')){
      $('modelStatus').textContent = 'model: mock';
    }
    // pulse active runners briefly to give visual feedback
    ['runner1','runner2','runner3'].forEach(id => {
      const el = document.getElementById(id);
      if(!el) return;
      if(el.getAttribute('fill') !== 'transparent'){
        el.classList.add('runner-pulse');
        setTimeout(()=>el.classList.remove('runner-pulse'), 700);
      }
    });
  }catch(err){
    $('insights').textContent = 'Request failed: ' + err;
    $('jsonOut').textContent = ''+err;
  }finally{
    setLoading(false);
  }
}

$('predict').addEventListener('click', doPredict);
$('example').addEventListener('click', ()=>{
  $('game_id').value='G20251103A';
  $('inning').value='5';
  $('outs').value='1';
  $('bases').value='1,0,0';
  $('batter_id').value='444482';
  $('pitcher_id').value='445926';
});
$('reset').addEventListener('click', ()=>{
  $('probs').querySelectorAll('.bar > i').forEach(i=>i.style.width='0%');
  $('hitPct').textContent=''; $('kPct').textContent=''; $('wPct').textContent='';
  $('insights').textContent='No insights yet';
  $('jsonOut').textContent='{}';
});

// ping model status on load
window.addEventListener('load', async ()=>{
  try{
    const r = await fetch('/');
    if(r.ok){
      $('modelStatus').textContent = 'service up';
    }
  }catch(e){}
  // initialize field visualization based on bases input
  initField();
  updateFieldFromInput();
});

// Field visualization helpers
function initField(){
  const container = $('field');
  if(!container) return;
  const svg = `
  <svg viewBox="0 0 200 200" width="160" height="160" xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="200" height="200" rx="10" ry="10" fill="rgba(255,255,255,0.02)" />
    <g transform="translate(100,100)">
      <rect x="-60" y="-60" width="120" height="120" transform="rotate(45)" fill="rgba(10,20,30,0.6)" stroke="rgba(255,255,255,0.03)"/>
      <circle id="base1" cx="50" cy="50" r="10" fill="#0b1220" stroke="#fff" stroke-opacity="0.08" />
      <circle id="base2" cx="0" cy="-70" r="10" fill="#0b1220" stroke="#fff" stroke-opacity="0.08" />
      <circle id="base3" cx="-50" cy="50" r="10" fill="#0b1220" stroke="#fff" stroke-opacity="0.08" />
      <circle id="home"  cx="0" cy="70" r="12" fill="#0b1220" stroke="#fff" stroke-opacity="0.08" />
      <circle cx="0" cy="0" r="8" fill="#0b1220" stroke="#fff" stroke-opacity="0.03" />
      <circle id="runner1" cx="50" cy="50" r="6" fill="transparent" />
      <circle id="runner2" cx="0" cy="-70" r="6" fill="transparent" />
      <circle id="runner3" cx="-50" cy="50" r="6" fill="transparent" />
    </g>
  </svg>`;
  container.innerHTML = svg;
  ['runner1','runner2','runner3'].forEach(id=>{
    const el = document.getElementById(id);
    if(!el) return;
    el.style.cursor='pointer';
    el.addEventListener('click', ()=>{
      toggleRunner(id);
      syncBasesToInput();
    });
  });
  // add click handlers to bases so clicking a base toggles the runner and triggers a quick prediction
  ['base1','base2','base3'].forEach((baseId, idx)=>{
    const b = document.getElementById(baseId);
    if(!b) return;
    b.style.cursor = 'pointer';
    b.addEventListener('click', (ev)=>{
      // toggle corresponding runner
      const rid = ['runner1','runner2','runner3'][idx];
      toggleRunner(rid);
      syncBasesToInput();
      // if user clicked intentionally, fire a prediction (small debounce)
      // allow Shift+click to only toggle without predicting
      if (!ev.shiftKey) doPredict();
    });
  });
}

function setRunner(idx, on){
  const id = ['runner1','runner2','runner3'][idx];
  const el = document.getElementById(id);
  if(!el) return;
  el.setAttribute('fill', on ? '#f97316' : 'transparent');
  if(on){
    el.classList.add('runner-on');
  } else {
    el.classList.remove('runner-on');
  }
}

function toggleRunner(id){
  const el = document.getElementById(id);
  if(!el) return;
  const on = el.getAttribute('fill') !== 'transparent';
  el.setAttribute('fill', on ? 'transparent' : '#f97316');
  el.classList.toggle('runner-on', !on);
}

// Animate runners advancing one base (front-end only). This moves each runner circle to the next base
// and then updates their cx/cy attributes so future animations are relative to new positions.
function animateRunnersAdvance(){
  try{
    const mapping = {
      runner1: document.getElementById('runner1'),
      runner2: document.getElementById('runner2'),
      runner3: document.getElementById('runner3')
    };
    const bases = {
      base1: document.getElementById('base1'),
      base2: document.getElementById('base2'),
      base3: document.getElementById('base3'),
      home: document.getElementById('home')
    };
    if(!mapping.runner1 || !bases.base1) return;

    // helper to get numeric cx/cy
    const getPos = (el)=>({x: parseFloat(el.getAttribute('cx')), y: parseFloat(el.getAttribute('cy'))});

    // compute targets: runner3 -> home, runner2 -> base3, runner1 -> base2, new batter appears at base1
    const targets = {};
    targets.runner3 = getPos(bases.home);
    targets.runner2 = getPos(bases.base3);
    targets.runner1 = getPos(bases.base2);

    const duration = 900; // ms
    Object.keys(mapping).forEach(rid=>{
      const el = mapping[rid];
      if(!el) return;
      const fill = el.getAttribute('fill');
      if(fill === 'transparent') return; // no runner here
      const cur = getPos(el);
      const tgt = targets[rid];
      if(!tgt) return;
      const dx = tgt.x - cur.x;
      const dy = tgt.y - cur.y;
      // apply CSS transform-based animation
      el.style.transition = `transform ${duration}ms cubic-bezier(.2,.8,.2,1)`;
      el.style.transform = `translate(${dx}px, ${dy}px)`;
      // after animation completes, snap to new coords and reset transform
      setTimeout(()=>{
        el.setAttribute('cx', String(tgt.x));
        el.setAttribute('cy', String(tgt.y));
        el.style.transition = '';
        el.style.transform = '';
      }, duration + 50);
    });

    // small stagger: place a new runner at first base (batter to first)
    setTimeout(()=>{
      const r1 = document.getElementById('runner1');
      if(!r1) return;
      // ensure runner1 is present (we just moved the previous one forward)
      r1.setAttribute('fill','#f97316');
      r1.classList.add('runner-on');
      syncBasesToInput();
    }, duration + 60);
  }catch(e){console.warn('animateRunnersAdvance failed', e)}
}

function updateFieldFromInput(){
  const basesStr = $('bases').value || '';
  const arr = basesStr.split(',').map(s=>parseInt(s.trim()||'0',10));
  setRunner(0, !!arr[0]);
  setRunner(1, !!arr[1]);
  setRunner(2, !!arr[2]);
}

function syncBasesToInput(){
  const arr = [document.getElementById('runner1').getAttribute('fill')!=='transparent'?1:0,
               document.getElementById('runner2').getAttribute('fill')!=='transparent'?1:0,
               document.getElementById('runner3').getAttribute('fill')!=='transparent'?1:0];
  $('bases').value = arr.join(',');
}

$('bases').addEventListener('change', updateFieldFromInput);
