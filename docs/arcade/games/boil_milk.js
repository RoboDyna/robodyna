/* Boil Milk — household (state transition).
 * Side view of a pot of milk on a gas stove. Space (or clicking the knob) toggles the burner.
 * While on, the milk heats (randomized rate); near boiling a foam head rises, accelerating as it
 * grows. Turning the burner off does not stop it at once: the burner and pot hold heat, so the foam
 * keeps climbing for a moment before it collapses.
 * Success: the foam rose above the marked line, the stove is off, and nothing overflowed for 2 s.
 * Overflow fails; never boiling fails at time-up.
 */
Arcade.register({
  id: "boil_milk",
  maxTime: 24,
  levels: { base: "Bring the milk up past the line, then shut the stove off before it boils over." },
  controls: [["Space", "toggle the stove"], ["Click", "the knob to toggle"]],
  keys: [{ key: " ", label: "stove on / off" }],
  create(api) {
    const { P, W, rand } = api;
    const POT = { x: W / 2, w: 270, top: 146, bot: 398 };
    const H = POT.bot - POT.top, BASE = 0.38, LINE = rand(0.66, 0.74), RIM = 0.97;
    const KNOB = { x: W / 2 + 200, y: 474, r: 24 };
    const heatRate = rand(15, 23), foamK = rand(0.075, 0.095);
    let on = false, b = 0, T = rand(22, 30), f = 0, reached = false, offAt = null, acted = false, spill = 0;
    const bubbles = [];

    function toggle() {
      on = !on; acted = true; api.sfx.click();
      offAt = on ? null : api.time;
    }
    return {
      keyDown(k) { if (k === " ") toggle(); },
      pointerDown(x, y) { if (Math.hypot(x - KNOB.x, y - KNOB.y) < KNOB.r + 14) toggle(); },
      update(dt) {
        // burner heat lags the knob (residual heat after shutting off)
        b += ((on ? 1 : 0) - b) * (1 - Math.exp(-dt / (on ? 0.35 : 0.75)));
        T = Math.min(100, T + (heatRate * b - 0.03 * (T - 22)) * dt);
        const hot = Math.max(0, Math.min(1, (T - 91) / 8));
        const q = hot * b;
        // foam grows faster the higher it is; collapses without heat input
        f += (foamK * q * (1 + 5.5 * f) - 0.09 * (1 - q) * f) * dt;
        f = Math.max(0, f);
        if (BASE + f >= LINE && !reached) { reached = true; api.sfx.tick(); }
        if (q > 0.05 && api.chance(dt * 30 * q)) bubbles.push({ x: rand(-POT.w / 2 + 16, POT.w / 2 - 16), r: rand(4, 9), a: 1 });
        bubbles.forEach(p => (p.a -= dt * 1.6));
        for (let i = bubbles.length - 1; i >= 0; i--) if (bubbles[i].a <= 0) bubbles.splice(i, 1);
        if (BASE + f >= RIM) { spill = 1; return api.fail("The milk boiled over"); }
        if (!on && reached && offAt != null && api.time - offAt >= 2) api.succeed("Milk rose and the stove is off, no overflow");
      },
      draw(g) {
        const c = g.ctx;
        // kitchen: wall + counter + stove front
        g.clear(P.wall);
        for (let x = 0; x < W; x += 54) g.line(x, 0, x, 430, "rgba(234,215,220,0.6)", 1);
        g.rr(-10, 430, W + 20, 120, 0, "#e9edf1", P.tableEdge, 2);
        g.rr(W / 2 - 250, 430, 500, 100, 12, "#d4dbe2", P.metal, 2);
        // burner and flames
        g.rr(W / 2 - 170, 424, 340, 10, 4, P.black);
        for (let i = -4; i <= 4; i++) {
          const fx = W / 2 + i * 26, fh = 6 + 22 * b * (0.8 + 0.2 * Math.sin(api.time * 20 + i * 1.7));
          if (b > 0.03) g.poly([[fx - 8, 424], [fx + 8, 424], [fx, 424 - fh]], `rgba(60,120,240,${0.35 + 0.5 * b})`);
        }
        // pot body (cut-away so the milk level is visible)
        const L = x => POT.x - POT.w / 2 + x, top = POT.top, bot = POT.bot;
        g.rr(L(-26), top + 30, 30, 12, 5, P.metalDark); g.rr(L(POT.w - 4), top + 30, 30, 12, 5, P.metalDark);
        g.rr(L(-8), top, POT.w + 16, H + 6, 10, P.metalDark);
        g.rr(L(0), top, POT.w, H - 2, 6, "#c9d2db");
        // milk + foam
        const lvl = y => bot - 2 - y * (H - 2);
        const milkTop = lvl(BASE), foamTop = lvl(Math.min(RIM, BASE + f));
        g.rr(L(0), milkTop, POT.w, bot - 2 - milkTop, 4, "#fbf7ee");
        if (f > 0.005) {
          g.rr(L(0), foamTop, POT.w, milkTop - foamTop + 6, 6, "#fffdf7");
          for (let x = 10; x < POT.w; x += 22) g.circle(L(x), foamTop + 2, 9 + 3 * Math.sin(x + api.time * 6), "#fffdf7");
          for (let x = 18; x < POT.w; x += 34) g.circle(L(x), foamTop + 14 + (x % 3) * 8, 5, null, "rgba(200,190,170,0.5)", 1.5);
        }
        for (const p of bubbles) g.circle(L(POT.w / 2 + p.x), foamTop + 6, p.r * (1.2 - p.a * 0.2), null, `rgba(190,180,160,${p.a})`, 1.5);
        // target line
        const ly = lvl(LINE);
        g.line(L(-14), ly, L(POT.w + 14), ly, reached ? P.green : P.teal, 3, [10, 6]);
        g.chip(reached ? "line reached" : "let it rise to here", L(POT.w + 90), ly, reached ? P.green : P.teal);
        g.line(L(0), lvl(RIM), L(POT.w), lvl(RIM), "rgba(224,75,60,0.5)", 2);
        g.text("rim", L(-30), lvl(RIM) + 2, { size: 11, color: P.red, align: "center" });
        if (spill) for (let i = 0; i < 2; i++) g.rr(L(i ? POT.w : -18), top - 4, 18, H * 0.6, 8, "#fffdf7");
        // knob
        g.circle(KNOB.x, KNOB.y, KNOB.r + 6, "#c3ccd5");
        g.circle(KNOB.x, KNOB.y, KNOB.r, on ? P.orange : P.black, P.ink, 2);
        const ka = on ? -Math.PI / 2 + 1.2 : -Math.PI / 2;
        g.line(KNOB.x, KNOB.y, KNOB.x + Math.cos(ka) * (KNOB.r - 4), KNOB.y + Math.sin(ka) * (KNOB.r - 4), "#fff", 4);
        g.text(on ? "ON" : "OFF", KNOB.x + 46, KNOB.y, { size: 13, color: on ? P.orange : P.muted });
        // thermometer
        g.rr(L(-110), 220, 16, 170, 8, "#fff", P.line, 2);
        const tf = Math.max(0, Math.min(1, (T - 20) / 80));
        g.rr(L(-106), 386 - 162 * tf, 8, 162 * tf, 4, T > 91 ? P.red : P.orange);
        g.text(Math.round(T) + "°C", L(-102), 405, { size: 12, color: P.muted, align: "center" });
        if (!acted) g.chip("Space / click the knob: stove on. Shut it off early: the foam keeps rising", W / 2, 521, P.navy);
        else if (!on && reached && offAt != null) g.chip("Holding… " + Math.max(0, 2 - (api.time - offAt)).toFixed(1) + " s", W / 2 - 150, 490, P.green);
      },
      hud() { return [{ k: "Stove", v: on ? "On" : "Off", warn: on }, { k: "Milk", v: Math.round(T) + "°C" }]; }
    };
  }
});
