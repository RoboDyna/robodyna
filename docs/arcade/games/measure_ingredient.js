/* Measure Ingredient — household.
 * Side view of a measuring jar under an oil nozzle. Space (or the red valve button) toggles the nozzle.
 * Oil leaves the nozzle and needs ~0.5 s to fall into the jar, so after you shut the valve the stream
 * still in the air keeps landing. The jar rides on a small sliding tray that drifts back and forth under
 * the nozzle; oil that misses the jar mouth spills and fails. Fill to the target ring (one of
 * 25/50/75/100 %, random) within ±5 %; overfilling fails.
 */
Arcade.register({
  id: "measure_ingredient",
  maxTime: 32,
  levels: { base: "Fill to the target ring (±5 %) without spilling; the jar drifts under the nozzle." },
  controls: [["Space", "toggle the oil nozzle"], ["Click", "the valve button (same)"]],
  keys: [{ key: " ", label: "Valve on / off" }],
  create(api) {
    const { P, W, rand, pick } = api;
    const NX = 405, NY = 132, G = 900, TOL = 0.05;
    const JB = 462, JH = 230, JW = 124, MOUTH = JB - JH;            // jar inner box
    const FULL = 0.86;                                            // the 100 % ring sits at 86 % of the jar height
    const target = pick([0.25, 0.5, 0.75, 1.0]);
    const q = rand(0.16, 0.2);                                    // flow, in "100 %" units per second
    const osc = { amp: rand(30, 70), w: (2 * Math.PI) / rand(4.2, 6), ph: rand(0, 6.28) };
    const BTN = { x: 640, y: 150, r: 28 };
    let valve = false, lag = 0, level = 0, drops = [], acted = false, settle = 0, emitAcc = 0;
    const jarX = () => NX + osc.amp * Math.sin(osc.w * api.time + osc.ph);
    const yOf = f => JB - f * FULL * JH;

    function toggle() {
      valve = !valve; acted = true; lag = valve ? 0.12 : 0; api.sfx.click();
    }
    return {
      debug: () => ({ level, target, valve, drops: drops.length, osc, q, NX, JW }),   // test hook
      keyDown(k) { if (k === " ") toggle(); },
      pointerDown(x, y) { if (Math.hypot(x - BTN.x, y - BTN.y) < BTN.r + 10) toggle(); },
      update(dt) {
        if (valve) {
          lag -= dt;
          if (lag <= 0) {                                         // emit small oil packets
            emitAcc += dt;
            while (emitAcc > 1 / 90) { emitAcc -= 1 / 90; drops.push({ x: NX + rand(-0.4, 0.4), y: NY + 4, vy: 70, v: q / 90 }); }
          }
        }
        const jx = jarX(), half = JW / 2 - 4;
        for (let i = drops.length - 1; i >= 0; i--) {
          const d = drops[i], py = d.y;
          d.vy += G * dt; d.y += d.vy * dt;
          if (!d.out && py < MOUTH && d.y >= MOUTH && Math.abs(d.x - jx) > half) d.out = true;   // missed the mouth
          if (!d.out && d.y >= yOf(level)) { level += d.v; drops.splice(i, 1); continue; }
          if (d.out && d.y > 488) { drops.splice(i, 1); return api.fail("Oil spilled outside the jar"); }
        }
        if (level > target + TOL) return api.fail(`Overfilled past the ${Math.round(target * 100)} % ring`);
        if (!valve && !drops.length && Math.abs(level - target) <= TOL) {
          settle += dt; if (settle > 0.8) api.succeed(`Measured ${Math.round(level * 100)} % (target ${Math.round(target * 100)} %)`);
        } else settle = 0;
      },
      hud() { return [{ k: "Target", v: Math.round(target * 100) + " %" }, { k: "Level", v: Math.round(level * 100) + " %" }]; },
      draw(g) {
        const c = g.ctx;
        g.clear(P.wall);
        for (let y = 40; y < 480; y += 60) g.line(0, y, W, y, P.wallLine, 1);
        c.fillStyle = P.floor; c.fillRect(0, 480, W, 60); g.line(0, 480, W, 480, P.tableEdge, 3);
        // oil bottle (upside down) on a bracket, nozzle
        g.rr(NX - 120, 20, 240, 12, 4, P.metalDark);
        g.rr(NX - 34, 24, 68, 78, 14, "rgba(242,194,48,0.55)", "#c99a1a", 2);
        g.rr(NX - 12, 98, 24, 20, 4, P.metal, P.metalDark, 1.5);
        g.rr(NX - 5, 116, 10, 16, 3, P.metalDark);
        g.circle(NX, 108, 4, valve ? P.green : P.red);
        // valve button
        g.circle(BTN.x, BTN.y, BTN.r + 4, "#fff", P.line, 2);
        g.circle(BTN.x, BTN.y, BTN.r, valve ? P.green : P.red);
        g.text(valve ? "ON" : "OFF", BTN.x, BTN.y + 1, { size: 14, weight: 800, color: "#fff", align: "center" });
        g.text("valve", BTN.x, BTN.y + BTN.r + 18, { size: 12, color: P.muted, align: "center" });
        // tray rail + tray
        const jx = jarX();
        g.rr(NX - osc.amp - 100, 474, osc.amp * 2 + 200, 6, 3, P.metal);
        g.rr(jx - 84, 464, 168, 10, 4, P.metalDark);
        // jar: oil, rings, glass
        g.ellipse(jx, 466, 80, 6, P.shadow);
        const ly = yOf(level);
        if (level > 0) {
          const gr = c.createLinearGradient(0, ly, 0, JB);
          gr.addColorStop(0, "#f6cf5a"); gr.addColorStop(1, "#e0a92a");
          g.rr(jx - JW / 2, Math.min(ly, JB - 3), JW, JB - Math.min(ly, JB - 3), 6, gr);
          g.line(jx - JW / 2 + 2, ly, jx + JW / 2 - 2, ly, "rgba(255,255,255,0.6)", 2);
        }
        g.rr(jx - JW / 2 - 6, MOUTH - 6, JW + 12, JH + 10, 10, "rgba(255,255,255,0.35)", "rgba(111,124,136,0.8)", 3);
        g.rr(jx - JW / 2 - 10, MOUTH - 10, JW + 20, 10, 4, "rgba(154,166,178,0.9)");
        for (const f of [0.25, 0.5, 0.75, 1.0]) {
          const y = yOf(f), isT = f === target;
          if (isT) { c.fillStyle = "rgba(238,138,29,0.16)"; c.fillRect(jx - JW / 2, yOf(f + TOL), JW, yOf(f - TOL) - yOf(f + TOL)); }
          g.line(jx - JW / 2 - 6, y, jx - JW / 2 + (isT ? JW + 6 : 22), y, isT ? P.orange : P.metalDark, isT ? 3.5 : 2);
          g.text(Math.round(f * 100) + "%", jx - JW / 2 - 12, y, { size: 11, color: isT ? P.orange : P.muted, align: "right" });
        }
        g.chip("target", jx + JW / 2 + 40, yOf(target), P.orange);
        // oil drops / stream
        for (let i = 0; i < drops.length; i++) {               // continuous stream: join neighbouring packets
          const d = drops[i], n = drops[i + 1];
          if (n && Math.abs(n.y - d.y) < 40) g.line(d.x, d.y, n.x, n.y, "#e9b53a", 5);
          else g.circle(d.x, d.y, 3, "#e9b53a");
        }
        if (!acted) g.chip("Space (or the valve button) toggles the oil", W / 2, 510, P.navy);
      }
    };
  }
});
