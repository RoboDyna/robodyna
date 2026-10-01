/* Cook Food Timer — household (state transition).
 * Top-down: a pancake sits in a pan on the stove, with a pie timer (green -> yellow -> red) beside it.
 * The timer advances only while the pan is hot. Space (or clicking the knob) toggles the stove.
 * The pan heats up gradually, so the timer speeds up the longer the stove is on, and the pan stays hot
 * for a moment after shutting off, so the timer keeps creeping forward (rates randomized).
 * Success: stove off and the timer settles inside the target band. Passing the band fails
 * (overcooked); stopping early is allowed but the food must still reach the band before time-up.
 */
Arcade.register({
  id: "cook_food_timer",
  maxTime: 24,
  levels: { base: "Shut the stove off so the timer stops inside the target band." },
  controls: [["Space", "toggle the stove"], ["Click", "the knob to toggle"]],
  keys: [{ key: " ", label: "stove on / off" }],
  create(api) {
    const { P, W, rand } = api;
    const PAN = { x: 360, y: 285, r: 120 }, PIE = { x: 650, y: 230, r: 52 }, KNOB = { x: 360, y: 470, r: 22 };
    const lo = rand(0.5, 0.58), hi = lo + rand(0.08, 0.1);
    const rate = rand(0.11, 0.16);            // timer fraction per second at full pan heat
    const TAU_UP = rand(1.3, 2.0), TAU_DOWN = rand(0.6, 0.85);
    let on = false, h = 0, frac = 0, acted = false;
    const bub = Array.from({ length: 14 }, () => ({ a: rand(0, 6.28), r: rand(10, 62), s: rand(2, 4) }));

    function toggle() { on = !on; acted = true; api.sfx.click(); }
    function food(f) {
      const stops = [[0, [246, 232, 190]], [0.35, [236, 196, 120]], [0.6, [205, 140, 64]], [0.8, [120, 70, 34]], [1, [50, 34, 24]]];
      for (let i = 1; i < stops.length; i++) if (f <= stops[i][0]) {
        const [a, ca] = stops[i - 1], [b, cb] = stops[i], k = (f - a) / (b - a);
        return `rgb(${ca.map((v, j) => Math.round(v + (cb[j] - v) * k)).join(",")})`;
      }
      return "rgb(50,34,24)";
    }
    return {
      keyDown(k) { if (k === " ") toggle(); },
      pointerDown(x, y) { if (Math.hypot(x - KNOB.x, y - KNOB.y) < KNOB.r + 14) toggle(); },
      update(dt) {
        h += ((on ? 1 : 0) - h) * (1 - Math.exp(-dt / (on ? TAU_UP : TAU_DOWN)));
        frac += rate * h * h * dt;               // hotter pan cooks disproportionately faster
        if (frac > hi) return api.fail("Overcooked: the timer passed the target band");
        if (!on && h < 0.04 && frac >= lo) api.succeed("Cooked to the target band");
      },
      draw(g) {
        g.scene();
        const c = g.ctx;
        // stove top
        g.rr(PAN.x - 175, 120, 350, 390, 18, "#39414c", "#262c34", 2);
        g.circle(PAN.x, PAN.y, 130, `rgba(238,90,40,${0.12 + 0.55 * h})`);
        g.circle(PAN.x, PAN.y, 100, null, `rgba(255,140,60,${0.15 + 0.7 * h})`, 3);
        // pan with handle
        g.rr(PAN.x - PAN.r - 144, PAN.y - 12, 150, 24, 10, P.black);
        g.shadow(PAN.x, PAN.y + 8, PAN.r, 0);
        g.circle(PAN.x, PAN.y, PAN.r, "#5b636e", "#2b3038", 3);
        g.circle(PAN.x, PAN.y, PAN.r - 12, "#4a515b");
        // pancake
        g.circle(PAN.x, PAN.y, 74, food(frac), "rgba(90,50,20,0.35)", 3);
        g.circle(PAN.x, PAN.y, 62, food(Math.max(0, frac - 0.06)));
        if (frac < 0.45) for (const p of bub) {
          const a = Math.max(0, Math.sin(api.time * p.s + p.a)) * h;
          if (a > 0.2) g.circle(PAN.x + Math.cos(p.a) * p.r, PAN.y + Math.sin(p.a) * p.r * 0.95, 3 + 2 * a, null, `rgba(140,100,50,${a * 0.6})`, 1.5);
        }
        // knob on the stove front
        g.circle(KNOB.x, KNOB.y, KNOB.r + 6, "#c3ccd5");
        g.circle(KNOB.x, KNOB.y, KNOB.r, on ? P.orange : P.black, P.ink, 2);
        const ka = on ? -Math.PI / 2 + 1.2 : -Math.PI / 2;
        g.line(KNOB.x, KNOB.y, KNOB.x + Math.cos(ka) * (KNOB.r - 4), KNOB.y + Math.sin(ka) * (KNOB.r - 4), "#fff", 4);
        g.text(on ? "ON" : "OFF", KNOB.x + 40, KNOB.y, { size: 13, color: on ? P.orange : "#c3ccd5" });
        // pie timer with target band
        const x = PIE.x, y = PIE.y, r = PIE.r, a0 = -Math.PI / 2, A = f => a0 + Math.PI * 2 * f;
        g.shadow(x, y + 5, r, 0);
        g.circle(x, y, r, "#fff", P.line, 2);
        [[0, 1 / 3, P.green], [1 / 3, 2 / 3, P.yellow], [2 / 3, 1, P.red]].forEach(([s, e, col]) => {
          c.save(); c.globalAlpha = 0.16; c.beginPath(); c.moveTo(x, y); c.arc(x, y, r - 3, A(s), A(e)); c.closePath(); c.fillStyle = col; c.fill(); c.restore();
        });
        const fcl = Math.min(1, frac), col = fcl < 1 / 3 ? P.green : fcl < 2 / 3 ? P.yellow : P.red;
        c.beginPath(); c.moveTo(x, y); c.arc(x, y, r - 3, a0, A(fcl)); c.closePath(); c.fillStyle = col; c.fill();
        c.save(); c.globalAlpha = 0.22; c.beginPath(); c.moveTo(x, y); c.arc(x, y, r - 3, A(lo), A(hi)); c.closePath(); c.fillStyle = P.navy; c.fill(); c.restore();
        c.beginPath(); c.arc(x, y, r + 7, A(lo), A(hi)); c.strokeStyle = P.navy; c.lineWidth = 8; c.stroke();
        g.line(x, y, x + Math.cos(A(fcl)) * (r - 2), y + Math.sin(A(fcl)) * (r - 2), P.ink, 2.5);
        g.circle(x, y, 4, P.ink);
        g.text("target band", x, y + r + 26, { size: 12, color: P.muted, align: "center" });
        // pan heat gauge
        g.text("pan heat", x, 360, { size: 12, color: P.muted, align: "center" });
        g.rr(x - 70, 374, 140, 14, 7, "#eef2f6", P.line, 1.5);
        g.rr(x - 70, 374, 140 * h, 14, 7, P.orange);
        if (!acted) g.chip("Space / click the knob: stove on. The pan stays hot after you shut it off", W / 2, 521, P.navy);
      },
      hud() { return [{ k: "Stove", v: on ? "On" : "Off", warn: on }, { k: "Doneness", v: Math.round(frac * 100) + "%" }]; }
    };
  }
});
