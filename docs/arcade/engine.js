/* RoboDyna Arcade engine.
 *
 * Each game lives in arcade/games/<task_id>.js and calls Arcade.register(def).
 *
 * def = {
 *   id: "catch_ramp_ball",                 // paper task id (must exist in ARCADE_DATA)
 *   levels: { base: "...", var1: "...", var2: "...", "var1+2": "..." },  // household: { base: "..." }
 *   controls: [["Click", "place the cup (once)"], ["Space", "..."]],
 *   maxTime: 20,                           // seconds per episode before it fails as "Time up" (default 30)
 *   create(api) { return game; }           // called once per episode
 * }
 *
 * game = {                                 // every method is optional
 *   update(dt),                            // dt in seconds (capped at 1/20)
 *   draw(g),                               // g: drawing helpers bound to the canvas (see makeG)
 *   pointerDown(x, y), pointerMove(x, y), pointerUp(x, y),   // virtual coordinates, 0..W x 0..H
 *   keyDown(key), keyUp(key),              // KeyboardEvent.key: "ArrowLeft", " ", "1", ...
 *   hud()                                  // -> [{k: "Label", v: "value", warn: bool}]
 * }
 *
 * api = {
 *   W: 810, H: 540, level: "base"|"var1"|"var2"|"var1+2", var1: bool, var2: bool,
 *   seed, rng() -> [0,1), rand(a, b), randInt(a, b), pick(arr), chance(p),
 *   time,                                  // seconds since the episode started (read-only)
 *   succeed(msg), fail(msg),               // end the episode (first call wins)
 *   ended,                                 // true once succeed/fail was called
 *   sfx: { click, good, bad, tick, whoosh },
 *   P                                      // palette (see below)
 * }
 */
(function () {
  "use strict";
  const W = 810, H = 540, EPISODES = 5;
  const LEVELS = ["base", "var1", "var2", "var1+2"];
  const CATS = {
    traj:      { label: "Trajectory prediction", color: "#16968a" },
    periodic:  { label: "Periodic pattern",      color: "#2f6db5" },
    avoid:     { label: "Dynamic avoidance",     color: "#d98a1c" },
    state:     { label: "State transition",      color: "#3f8f4a" },
    static:    { label: "Static to dynamic",     color: "#c8433a" },
    household: { label: "Household",             color: "#7b5ea7" }
  };
  // Shared palette: keep game art consistent with the RoboDyna site and the SAPIEN renders.
  const P = {
    bg: "#eef2f6", wall: "#f5e9ec", wallLine: "#ead7dc", table: "#ffffff", tableEdge: "#d9e1e8",
    floor: "#e6ebf0", wood: "#c9a77c", woodDark: "#a8865c", metal: "#9aa6b2", metalDark: "#6f7c88",
    navy: "#0b2a45", ink: "#0f2233", muted: "#55687a", line: "#dde5ec",
    teal: "#139a9a", orange: "#ee8a1d", red: "#e04b3c", green: "#3fa55b", blue: "#2f6db5",
    yellow: "#f2c230", black: "#2b2f36", white: "#ffffff", pink: "#f2a7b5", purple: "#7b5ea7",
    shadow: "rgba(15,34,51,0.16)", goodZone: "rgba(63,165,91,0.16)", badZone: "rgba(224,75,60,0.14)",
    cat: Object.fromEntries(Object.entries(CATS).map(([k, v]) => [k, v.color]))
  };
  const POLICIES = [["pi05", "π0.5"], ["xvla", "X-VLA"], ["fwam", "FastWAM"]];
  const DATA = window.ARCADE_DATA || {};
  const registry = {};
  const titleOf = id => id.split("_").map(w => w[0].toUpperCase() + w.slice(1)).join(" ");

  // ---------- sound ----------
  let AC = null, soundOn = true;
  function beep(freq, dur, type, vol) {
    if (!soundOn) return;
    try {
      AC = AC || new (window.AudioContext || window.webkitAudioContext)();
      const o = AC.createOscillator(), gn = AC.createGain(), t = AC.currentTime;
      o.type = type || "sine"; o.frequency.value = freq;
      gn.gain.setValueAtTime(vol || 0.05, t); gn.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(gn).connect(AC.destination); o.start(t); o.stop(t + dur);
    } catch (e) { /* audio unavailable */ }
  }
  const sfx = {
    click: () => beep(520, 0.06, "triangle", 0.04),
    tick: () => beep(980, 0.03, "square", 0.02),
    good: () => { beep(660, 0.09, "triangle"); setTimeout(() => beep(990, 0.12, "triangle"), 80); },
    bad: () => beep(170, 0.25, "sawtooth", 0.04),
    whoosh: () => beep(300, 0.12, "sine", 0.03)
  };

  // ---------- seeded rng ----------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- drawing helpers ----------
  function makeG(ctx) {
    const g = {
      ctx, W, H, P,
      clear(c) { ctx.fillStyle = c || P.bg; ctx.fillRect(0, 0, W, H); },
      // Standard scene: pink back wall strip + white table, as in the SAPIEN head-camera view.
      scene(opts) {
        const o = Object.assign({ wall: 70, x: 30, y: 70, w: W - 60, h: H - 90 }, opts || {});
        g.clear(P.floor);
        ctx.fillStyle = P.wall; ctx.fillRect(0, 0, W, o.wall);
        ctx.fillStyle = P.wallLine; ctx.fillRect(0, o.wall - 3, W, 3);
        g.rr(o.x, o.y, o.w, o.h, 14, P.table, P.tableEdge);
      },
      rr(x, y, w, h, r, fill, stroke, lw) {
        const k = Math.max(0, Math.min(r, w / 2, h / 2));
        ctx.beginPath();
        ctx.moveTo(x + k, y); ctx.arcTo(x + w, y, x + w, y + h, k); ctx.arcTo(x + w, y + h, x, y + h, k);
        ctx.arcTo(x, y + h, x, y, k); ctx.arcTo(x, y, x + w, y, k); ctx.closePath();
        if (fill) { ctx.fillStyle = fill; ctx.fill(); }
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
      },
      circle(x, y, r, fill, stroke, lw) {
        ctx.beginPath(); ctx.arc(x, y, Math.max(0, r), 0, Math.PI * 2);
        if (fill) { ctx.fillStyle = fill; ctx.fill(); }
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
      },
      ellipse(x, y, rx, ry, fill, stroke, lw, rot) {
        ctx.beginPath(); ctx.ellipse(x, y, Math.max(0, rx), Math.max(0, ry), rot || 0, 0, Math.PI * 2);
        if (fill) { ctx.fillStyle = fill; ctx.fill(); }
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
      },
      line(x1, y1, x2, y2, color, lw, dash) {
        ctx.save(); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
        ctx.strokeStyle = color || P.ink; ctx.lineWidth = lw || 2; ctx.lineCap = "round";
        if (dash) ctx.setLineDash(dash);
        ctx.stroke(); ctx.restore();
      },
      poly(pts, fill, stroke, lw) {
        ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
        if (fill) { ctx.fillStyle = fill; ctx.fill(); }
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1.5; ctx.stroke(); }
      },
      text(s, x, y, o) {
        o = o || {};
        ctx.font = `${o.weight || 700} ${o.size || 14}px Inter, ui-sans-serif, system-ui, sans-serif`;
        ctx.fillStyle = o.color || P.ink; ctx.textAlign = o.align || "left"; ctx.textBaseline = o.baseline || "middle";
        ctx.fillText(s, x, y);
      },
      // Soft contact shadow under an object at height z (bigger/lighter as z grows).
      shadow(x, y, r, z) {
        const k = 1 + (z || 0) / 120;
        g.ellipse(x, y, r * k, r * 0.55 * k, `rgba(15,34,51,${0.18 / k})`);
      },
      // Shaded ball (top-down) with a highlight.
      ball(x, y, r, color) {
        const gr = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.15, x, y, r);
        gr.addColorStop(0, "rgba(255,255,255,0.85)"); gr.addColorStop(0.25, color); gr.addColorStop(1, shade(color, -0.28));
        g.circle(x, y, r, gr);
      },
      // Progress / timer ring (pie timer style).
      pie(x, y, r, frac, colors) {
        g.circle(x, y, r, "#fff", P.line, 2);
        const c = colors || [P.green, P.yellow, P.red];
        const col = frac < 1 / 3 ? c[0] : frac < 2 / 3 ? c[1] : c[2];
        ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r - 3, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.min(1, frac));
        ctx.closePath(); ctx.fillStyle = col; ctx.fill();
      },
      // Small label chip, e.g. above a target.
      chip(s, x, y, color) {
        ctx.font = "800 11px Inter, ui-sans-serif, system-ui, sans-serif";
        const w = ctx.measureText(s).width + 14;
        g.rr(x - w / 2, y - 10, w, 20, 10, color || P.navy);
        g.text(s, x, y + 0.5, { size: 11, weight: 800, color: "#fff", align: "center" });
      }
    };
    return g;
  }
  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    let r = n >> 16, gg = (n >> 8) & 255, b = n & 255;
    const f = v => Math.max(0, Math.min(255, Math.round(amt < 0 ? v * (1 + amt) : v + (255 - v) * amt)));
    return "#" + ((1 << 24) | (f(r) << 16) | (f(gg) << 8) | f(b)).toString(16).slice(1);
  }

  // ---------- storage ----------
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem("robodyna-arcade:" + k)); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem("robodyna-arcade:" + k, JSON.stringify(v)); } catch (e) { /* ignore */ } }
  };
  const best = (id, lvl) => (store.get("best") || {})[`${id}/${lvl}`];
  function saveBest(id, lvl, sr) {
    const b = store.get("best") || {}, k = `${id}/${lvl}`;
    if (b[k] == null || sr > b[k]) { b[k] = sr; store.set("best", b); }
  }

  // ---------- page wiring ----------
  const $ = id => document.getElementById(id);
  let cv, ctx, g, def = null, level = "base", game = null, api = null;
  let episode = 0, results = [], phase = "idle", raf = null, last = 0, banner = null, paused = false;

  // The arcade plays only the base condition of each task; games still define var levels for later use.
  const PLAYED = ["base"];
  function levelsOf(d) { return Object.keys(d.levels || { base: "" }).filter(l => PLAYED.includes(l)); }

  function buildHub() {
    const grid = $("arcade-grid"), filters = $("arcade-filters");
    const ids = Object.keys(DATA);
    ids.forEach(id => {
      const info = DATA[id], cat = CATS[info.cat], d = registry[id];
      const card = document.createElement("button");
      card.type = "button"; card.className = "acard"; card.dataset.cat = info.cat;
      card.style.setProperty("--c", cat.color);
      card.disabled = !d;
      const lv = d ? levelsOf(d) : ["base"];
      const pips = lv.map(l => {
        const b = best(id, l);
        return `<span class="pip${b != null ? (b >= 0.6 ? " full" : " part") : ""}" title="${l}${b != null ? ": best " + Math.round(b * 100) + "%" : ""}"></span>`;
      }).join("");
      card.innerHTML = `<span class="thumb"><img loading="lazy" src="arcade/thumbs/${id}.jpg" alt=""></span>
        <span class="abody"><span class="atag">${cat.label}</span><b>${titleOf(id)}</b>
        <span class="pips">${d ? pips : "<em>coming soon</em>"}</span></span>`;
      if (d) card.addEventListener("click", () => openGame(id));
      grid.append(card);
    });
    const all = [["all", "All games", "#0b2a45"], ...Object.entries(CATS).map(([k, v]) => [k, v.label, v.color])];
    all.forEach(([k, label, color]) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "filter"; b.dataset.cat = k; b.style.setProperty("--c", color);
      b.innerHTML = (k === "all" ? "" : "<span class='dot'></span>") + label;
      b.addEventListener("click", () => {
        grid.querySelectorAll(".acard").forEach(c => (c.hidden = !(k === "all" || c.dataset.cat === k)));
        filters.querySelectorAll(".filter").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      });
      b.setAttribute("aria-pressed", String(k === "all"));
      filters.append(b);
    });
  }

  function openGame(id, lvl) {
    def = registry[id];
    const info = DATA[id], cat = CATS[info.cat];
    $("hub").hidden = true; $("play").hidden = false;
    $("g-title").textContent = titleOf(id);
    $("g-tag").textContent = cat.label; $("play").style.setProperty("--c", cat.color);
    $("g-instr").textContent = `“${info.instruction}”`;
    $("g-controls").innerHTML = (def.controls || []).map(([k, v]) => `<span><kbd>${k}</kbd> ${v}</span>`).join("");
    const tabs = $("g-levels"); tabs.innerHTML = "";
    levelsOf(def).forEach(l => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "lvl"; b.dataset.l = l;
      b.innerHTML = `${l}<small>${{ base: "easy", var1: "medium", var2: "medium", "var1+2": "hard" }[l]}</small>`;
      b.addEventListener("click", () => selectLevel(l));
      tabs.append(b);
    });
    tabs.hidden = levelsOf(def).length < 2;
    $("g-levelbox").hidden = levelsOf(def).length < 2;

    try { history.replaceState(null, "", "#" + id); } catch (e) { /* file:// */ }
    window.scrollTo({ top: 0 });
    fit();
    selectLevel(lvl && levelsOf(def).includes(lvl) ? lvl : levelsOf(def)[0]);
  }

  function backToHub() {
    stop(); def = null; game = null;
    $("play").hidden = true; $("hub").hidden = false;
    try { history.replaceState(null, "", location.pathname); } catch (e) { /* ignore */ }
    // refresh pips
    $("arcade-grid").innerHTML = ""; $("arcade-filters").innerHTML = ""; buildHub();
  }

  function selectLevel(l) {
    level = l;
    $("g-levels").querySelectorAll(".lvl").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.l === l)));
    $("g-leveldesc").textContent = def.levels ? def.levels[l] || "" : "";
    // Optional on-screen keys for touch play: def.keys = [{ key: "ArrowLeft", label: "◀" }, ...]
    // or a function of the level returning such a list.
    const keys = (typeof def.keys === "function" ? def.keys(l) : def.keys) || [];
    $("g-keys").innerHTML = keys.map(k => `<button type="button" data-key="${k.key}">${k.label}</button>`).join("");
    $("g-keys").hidden = !keys.length;
    episode = 0; results = [];
    newEpisode();
    showOverlay(`<p class="ov-k">${l} · ${EPISODES} episodes</p><h3>${titleOf(def.id)}</h3>
      <p>${def.levels && def.levels[l] ? def.levels[l] : ""}</p>
      <button class="obtn" id="ov-go">Start</button>`);
    $("ov-go").addEventListener("click", () => { hideOverlay(); begin(); });
  }

  function newEpisode() {
    const seed = (Math.random() * 1e9) | 0, rng = mulberry32(seed);
    api = {
      W, H, P, level, var1: level === "var1" || level === "var1+2", var2: level === "var2" || level === "var1+2",
      seed, rng, rand: (a, b) => a + (b - a) * rng(), randInt: (a, b) => Math.floor(a + (b - a + 1) * rng()),
      pick: arr => arr[Math.floor(rng() * arr.length)], chance: p => rng() < p,
      time: 0, ended: false, sfx,
      succeed: msg => endEpisode(true, msg), fail: msg => endEpisode(false, msg)
    };
    game = def.create(api) || {};
    phase = "ready"; banner = null;
    render();
    updateHud();
  }

  function begin() { phase = "run"; start(); }

  function endEpisode(ok, msg) {
    if (api.ended) return;
    api.ended = true; phase = "ending";
    results.push(ok);
    (ok ? sfx.good : sfx.bad)();
    banner = { ok, msg: msg || (ok ? "Success" : "Failed"), t: 0 };
    updateHud();
  }

  function afterBanner() {
    episode += 1;
    if (episode < EPISODES) { newEpisode(); begin(); return; }
    stop(); phase = "done";
    const n = results.filter(Boolean).length, sr = n / EPISODES;
    saveBest(def.id, level, sr);
    const res = (DATA[def.id].res || {})[level];
    const rows = [[`You`, sr, true]].concat(res ? POLICIES.map(([k, name]) => [name, (res[k].sync[0] + res[k].async[0]) / 2, false]) : []);
    const bars = rows.map(([name, v, me]) => `<div class="cmp${me ? " me" : ""}"><span>${name}</span><span class="trk"><span style="width:${Math.round(v * 100)}%"></span></span><b>${v.toFixed(2)}</b></div>`).join("");
    const lv = levelsOf(def), nextL = lv[lv.indexOf(level) + 1];
    showOverlay(`<p class="ov-k">${level} · result</p><h3>${n} / ${EPISODES} episodes solved</h3>
      <div class="cmps">${bars}</div>
      <p class="ov-note">Policy success rates on this condition from the paper (10 seeds, averaged over sync and async).</p>
      <div class="obtns"><button class="obtn ghost" id="ov-retry">Play again</button>${nextL ? `<button class="obtn" id="ov-next">Next: ${nextL}</button>` : `<button class="obtn" id="ov-hub">More games</button>`}</div>`);
    $("ov-retry").addEventListener("click", () => selectLevel(level));
    if (nextL) $("ov-next").addEventListener("click", () => selectLevel(nextL));
    else $("ov-hub").addEventListener("click", backToHub);
  }

  function updateHud() {
    const solved = results.filter(Boolean).length;
    const cells = [{ k: "Episode", v: `${Math.min(episode + 1, EPISODES)}/${EPISODES}` }, { k: "Solved", v: String(solved) }];
    const extra = game && game.hud ? game.hud() || [] : [];
    const maxT = def.maxTime || 30;
    if (phase === "run" || phase === "ending") cells.push({ k: "Time", v: Math.max(0, maxT - api.time).toFixed(1), warn: maxT - api.time < 5 });
    $("g-hud").innerHTML = cells.concat(extra).map(c => `<span class="cell${c.warn ? " warn" : ""}"><small>${c.k}</small><b>${c.v}</b></span>`).join("");
    $("g-dots").innerHTML = Array.from({ length: EPISODES }, (_, i) => `<span class="${i < results.length ? (results[i] ? "ok" : "no") : i === episode ? "cur" : ""}"></span>`).join("");
  }

  function frame(t) {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.05, (t - last) / 1000 || 0); last = t;
    if (paused) return;
    if (phase === "run" || phase === "ending") {
      api.time += dt;
      try { game.update && game.update(dt); } catch (e) { console.error(e); }
      if (phase === "run" && api.time >= (def.maxTime || 30)) api.fail("Time up");
    }
    if (banner) { banner.t += dt; if (banner.t > 1.3) { banner = null; afterBanner(); } }
    render();
    hudTick += dt; if (hudTick > 0.1) { hudTick = 0; updateHud(); }
  }
  let hudTick = 0;
  function start() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); } }
  function stop() { if (raf) cancelAnimationFrame(raf); raf = null; }

  function render() {
    if (!game) return;
    ctx.save();
    try { game.draw ? game.draw(g) : g.scene(); } catch (e) { console.error(e); }
    ctx.restore();
    if (banner) {
      const a = Math.min(1, banner.t * 6);
      ctx.save(); ctx.globalAlpha = a;
      g.rr(W / 2 - 170, H / 2 - 34, 340, 68, 16, banner.ok ? "rgba(63,165,91,0.95)" : "rgba(224,75,60,0.95)");
      g.text(banner.ok ? "Success" : "Failed", W / 2, H / 2 - 9, { size: 22, weight: 800, color: "#fff", align: "center" });
      g.text(banner.msg, W / 2, H / 2 + 15, { size: 13, weight: 600, color: "rgba(255,255,255,0.92)", align: "center" });
      ctx.restore();
    }
  }

  function showOverlay(html) { const o = $("g-overlay"); o.innerHTML = `<div class="ov-card">${html}</div>`; o.hidden = false; }
  function hideOverlay() { $("g-overlay").hidden = true; }

  function fit() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    ctx.setTransform(cv.width / W, 0, 0, cv.height / H, 0, 0);
    render();
  }
  function toV(e) {
    const r = cv.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  }
  const live = () => game && phase === "run" && !api.ended;

  function init() {
    cv = $("g-canvas"); ctx = cv.getContext("2d"); g = makeG(ctx);
    buildHub();
    $("g-back").addEventListener("click", backToHub);
    $("sound").addEventListener("click", e => {
      soundOn = !soundOn; e.currentTarget.setAttribute("aria-pressed", String(soundOn));
      e.currentTarget.querySelector("span").textContent = soundOn ? "Sound on" : "Sound off";
    });
    window.addEventListener("resize", () => def && fit());
    cv.addEventListener("pointerdown", e => { if (live() && game.pointerDown) { e.preventDefault(); cv.setPointerCapture(e.pointerId); game.pointerDown(...toV(e)); } });
    cv.addEventListener("pointermove", e => { if (live() && game.pointerMove) game.pointerMove(...toV(e)); });
    cv.addEventListener("pointerup", e => { if (live() && game.pointerUp) game.pointerUp(...toV(e)); });
    window.addEventListener("keydown", e => {
      if (!def || $("play").hidden) return;
      if (e.key === "Escape") { backToHub(); return; }
      if ([" ", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) e.preventDefault();
      if (e.repeat) return;
      if (live() && game.keyDown) game.keyDown(e.key);
    });
    window.addEventListener("keyup", e => { if (live() && game.keyUp) game.keyUp(e.key); });
    document.addEventListener("visibilitychange", () => { paused = document.hidden; });
    // test hook: on-screen buttons call game key handlers (touch devices)
    $("g-keys").addEventListener("pointerdown", e => { const k = e.target.closest("[data-key]"); if (k && live() && game.keyDown) game.keyDown(k.dataset.key); });
    $("g-keys").addEventListener("pointerup", e => { const k = e.target.closest("[data-key]"); if (k && live() && game.keyUp) game.keyUp(k.dataset.key); });
    const id = decodeURIComponent(location.hash.slice(1));
    if (registry[id]) openGame(id);
  }

  window.Arcade = {
    register(d) { registry[d.id] = d; },
    CATS, P, W, H, shade,
    // exposed for automated tests
    _debug: {
      get api() { return api; }, get game() { return game; }, get phase() { return phase; },
      openGame, selectLevel, results: () => results.slice(),
      // Start the current episode without the animation loop (for headless tests).
      begin() { hideOverlay(); phase = "run"; },
      // Advance the simulation synchronously by n steps of dt seconds, then render.
      step(n, dt) {
        dt = dt || 1 / 60;
        for (let i = 0; i < n && phase === "run"; i++) {
          api.time += dt; game.update && game.update(dt);
          if (phase === "run" && api.time >= (def.maxTime || 30)) api.fail("Time up");
        }
        render(); updateHud();
      },
      pointerDown: (x, y) => live() && game.pointerDown && game.pointerDown(x, y),
      pointerMove: (x, y) => game && game.pointerMove && game.pointerMove(x, y),
      pointerUp: (x, y) => live() && game.pointerUp && game.pointerUp(x, y),
      keyDown: k => live() && game.keyDown && game.keyDown(k),
      keyUp: k => live() && game.keyUp && game.keyUp(k)
    }
  };
  document.addEventListener("DOMContentLoaded", init);
})();
