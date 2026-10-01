/* Cook Meat Timer — state transition.
 * A steak sits in a hot pan; a pie timer next to the pan fills green -> yellow -> red as it cooks
 * (cook rate randomized per steak). A dark band on the timer rim marks the target doneness.
 * Return the steak to its cutting board so that it ends inside the band. Meat keeps cooking a little
 * after it leaves the heat (carry-over), so the move has to be made slightly early.
 * base:   one station; the steak cooks while it touches the pan. Space moves it to the board (once).
 * var1:   one station; the steak only cooks while Space is held (release pauses, residual heat lingers).
 *         Click the board (or press Down) to plate it.
 * var2:   two stations with their own pie timers (contact cook); Left / Right plate each steak.
 * var1+2: two stations; hold Left / Right to cook each steak, click a board to plate it.
 * Reaching the red zone in the pan (burnt) or plating outside the band fails.
 */
Arcade.register({
  id: "cook_meat_timer",
  maxTime: 22,
  levels: {
    base: "One pan; the timer runs while the steak is in the pan. Press Space to plate it in the band.",
    var1: "One pan; the steak only cooks while Space is held. Click the board (or Down) to plate.",
    var2: "Two pans with their own timers. Left / Right plate each steak.",
    "var1+2": "Two pans; hold Left / Right to cook each steak, click a board to plate it."
  },
  controls: [["Space", "plate (base) / hold to cook (var1)"], ["← / →", "left / right station (var2)"], ["Click", "board to plate (hold modes)"]],
  keys: lvl => ({
    base: [{ key: " ", label: "Space" }],
    var1: [{ key: " ", label: "Space" }, { key: "ArrowDown", label: "plate" }],
    var2: [{ key: "ArrowLeft", label: "◀" }, { key: "ArrowRight", label: "▶" }],
    "var1+2": [{ key: "ArrowLeft", label: "◀" }, { key: "ArrowRight", label: "▶" }]
  })[lvl],
  create(api) {
    const { P, W, rand } = api;
    const hold = api.var1, dual = api.var2;
    const BURNT = 0.76, LIFT_T = 0.4;
    const TAU_UP = 0.45, TAU_DOWN = hold ? 0.55 : 0.35;     // pan heat lag (carry-over after leaving the heat)
    const xs = dual ? [230, 580] : [405];
    const keysOf = dual ? ["ArrowLeft", "ArrowRight"] : [" "];
    const stations = xs.map((cx, i) => {
      const lo = rand(0.46, 0.55), hi = lo + rand(0.09, 0.11);
      const tCenter = hold ? rand(3.2, 5.2) : rand(3.8, 7.5) + (dual ? i * rand(-1.5, 2.5) : 0);
      return {
        cx, key: keysOf[i], lo, hi, rate: (lo + hi) / 2 / Math.max(2.8, tCenter),
        frac: 0, h: hold ? 0 : 1, held: false, where: "pan", t: 0, done: false, ok: null,
        pan: { x: cx, y: 255 }, board: { x: cx, y: 432 }, steam: []
      };
    });
    let acted = false;

    function plate(s) {
      if (s.where !== "pan" || api.ended) return;
      s.where = "lift"; s.t = 0; s.held = false; acted = true;
      api.sfx.whoosh();
    }
    function steakColor(f) {
      const stops = [[0, [217, 100, 106]], [0.4, [190, 104, 72]], [0.6, [140, 78, 44]], [0.76, [70, 46, 34]], [1, [40, 30, 26]]];
      for (let i = 1; i < stops.length; i++) {
        if (f <= stops[i][0]) {
          const [a, ca] = stops[i - 1], [b, cb] = stops[i], k = (f - a) / (b - a);
          return `rgb(${ca.map((v, j) => Math.round(v + (cb[j] - v) * k)).join(",")})`;
        }
      }
      return "rgb(40,30,26)";
    }

    return {
      keyDown(k) {
        const s = stations.find(st => st.key === k);
        if (k === "ArrowDown" && hold && !dual) { plate(stations[0]); return; }
        if (!s) return;
        if (hold) { if (s.where === "pan") { s.held = true; acted = true; api.sfx.click(); } }
        else plate(s);
      },
      keyUp(k) { const s = stations.find(st => st.key === k); if (s) s.held = false; },
      pointerDown(x, y) {
        if (!hold) return;
        for (const s of stations) if (Math.abs(x - s.board.x) < 100 && Math.abs(y - s.board.y) < 55) plate(s);
      },
      update(dt) {
        for (const s of stations) {
          const target = s.where === "pan" && (!hold || s.held) ? 1 : 0;
          s.h += (target - s.h) * (1 - Math.exp(-dt / (target > s.h ? TAU_UP : TAU_DOWN)));
          s.frac += s.rate * s.h * dt;
          if (s.h > 0.2 && api.chance(dt * 8 * s.h)) s.steam.push({ x: api.rand(-30, 30), y: 0, a: 1 });
          s.steam.forEach(p => { p.y -= 40 * dt; p.a -= dt * 1.2; });
          s.steam = s.steam.filter(p => p.a > 0);
          if (s.where === "lift") { s.t += dt / LIFT_T; if (s.t >= 1) { s.where = "board"; api.sfx.tick(); } }
          if (s.frac >= BURNT) return api.fail(dual ? `The ${s.cx < W / 2 ? "left" : "right"} steak burnt` : "The steak burnt");
          if (s.where === "board" && !s.done && s.h < 0.03) {
            s.done = true;
            const side = dual ? (s.cx < W / 2 ? "Left" : "Right") + " steak" : "Steak";
            if (s.frac < s.lo) return api.fail(`${side} undercooked (timer before the band)`);
            if (s.frac > s.hi) return api.fail(`${side} overcooked (timer past the band)`);
            s.ok = true; api.sfx.tick();
          }
        }
        if (stations.every(s => s.ok)) api.succeed(dual ? "Both steaks cooked to the band" : "Steak cooked to the band");
      },
      draw(g) {
        g.scene();
        const c = g.ctx;
        for (const s of stations) {
          const { cx } = s;
          // stove top with burner
          g.rr(cx - 120, 150, 240, 205, 16, "#39414c", "#262c34", 2);
          const glow = Math.max(0, Math.min(1, s.h));
          g.circle(s.pan.x, s.pan.y, 88, `rgba(238,90,40,${0.15 + 0.5 * glow})`);
          g.circle(s.pan.x, s.pan.y, 70, null, `rgba(255,140,60,${0.2 + 0.6 * glow})`, 3);
          // pan with handle
          const hx = dual ? (cx < W / 2 ? 1 : -1) : -1;
          g.rr(s.pan.x + hx * 76 - (hx < 0 ? 72 : 0), s.pan.y - 9, 72, 18, 8, P.black);
          g.shadow(s.pan.x, s.pan.y + 6, 80, 0);
          g.circle(s.pan.x, s.pan.y, 80, "#5b636e", "#2b3038", 3);
          g.circle(s.pan.x, s.pan.y, 70, "#4a515b");
          // cutting board
          g.shadow(s.board.x, s.board.y + 6, 92, 0);
          g.rr(s.board.x - 95, s.board.y - 42, 190, 84, 12, P.wood, P.woodDark, 2);
          g.circle(s.board.x + 80, s.board.y, 5, P.woodDark);
          // steak
          let sx = s.pan.x, sy = s.pan.y, lift = 0;
          if (s.where === "lift") {
            const k = s.t * s.t * (3 - 2 * s.t);
            sx = s.pan.x + (s.board.x - s.pan.x) * k; sy = s.pan.y + (s.board.y - s.pan.y) * k; lift = Math.sin(Math.PI * s.t) * 40;
          } else if (s.where === "board") { sx = s.board.x; sy = s.board.y; }
          if (lift) g.shadow(sx, sy + 8, 40, lift);
          const col = steakColor(s.frac);
          c.save(); c.translate(sx, sy - lift * 0.5); c.rotate(-0.25);
          g.ellipse(0, 0, 46, 30, col, "rgba(60,30,20,0.6)", 2);
          g.ellipse(-14, -6, 14, 7, "rgba(255,235,220,0.55)");        // fat cap
          if (s.frac > 0.25) for (let j = -2; j <= 2; j++) g.line(j * 14 - 10, -20, j * 14 + 10, 20, `rgba(50,25,15,${Math.min(0.6, (s.frac - 0.25) * 2)})`, 3);
          c.restore();
          for (const p of s.steam) g.circle(sx + p.x, sy - 30 + p.y, 7, `rgba(255,255,255,${0.5 * p.a})`);
          // pie timer with target band on the rim
          drawPie(g, cx + (dual ? (cx < W / 2 ? -155 : 155) : 175), 215, 38, s);
          // key label
          const lab = dual ? (s.key === "ArrowLeft" ? "◀" : "▶") : "Space";
          if (s.ok) g.chip("Done", s.board.x, s.board.y + 58, P.green);
          else g.chip(hold ? `hold ${lab} = cook` : `${lab} = plate`, s.board.x, s.board.y + 58, s.held ? P.orange : P.navy);
        }
        if (!acted) g.chip(hold ? "Hold to cook; release early (heat lingers), then click the board to plate"
          : "Plate the steak when its timer reaches the dark band (it keeps cooking a moment)", W / 2, 521, P.navy);
      },
      hud() {
        return stations.map((s, i) => ({ k: dual ? (i ? "Right" : "Left") : "Doneness", v: Math.round(s.frac * 100) + "%", warn: s.frac > s.hi }));
      }
    };

    function drawPie(g, x, y, r, s) {
      const c = g.ctx, a0 = -Math.PI / 2, A = f => a0 + Math.PI * 2 * f;
      g.shadow(x, y + 4, r, 0);
      g.circle(x, y, r, "#fff", P.line, 2);
      [[0, 1 / 3, P.green], [1 / 3, 2 / 3, P.yellow], [2 / 3, 1, P.red]].forEach(([a, b, col]) => {
        c.save(); c.globalAlpha = 0.16; c.beginPath(); c.moveTo(x, y); c.arc(x, y, r - 3, A(a), A(b)); c.closePath();
        c.fillStyle = col; c.fill(); c.restore();
      });
      const f = Math.min(1, s.frac), col = f < 1 / 3 ? P.green : f < 2 / 3 ? P.yellow : P.red;
      c.beginPath(); c.moveTo(x, y); c.arc(x, y, r - 3, a0, A(f)); c.closePath(); c.fillStyle = col; c.fill();
      // target band: shaded wedge + thick rim arc
      c.save(); c.globalAlpha = 0.22; c.beginPath(); c.moveTo(x, y); c.arc(x, y, r - 3, A(s.lo), A(s.hi)); c.closePath();
      c.fillStyle = P.navy; c.fill(); c.restore();
      c.beginPath(); c.arc(x, y, r + 6, A(s.lo), A(s.hi)); c.strokeStyle = P.navy; c.lineWidth = 7; c.stroke();
      g.text("target", x, y + r + 22, { size: 11, color: P.muted, align: "center" });
      g.line(x, y, x + Math.cos(A(f)) * (r - 2), y + Math.sin(A(f)) * (r - 2), P.ink, 2);
      g.circle(x, y, 3, P.ink);
    }
  }
});
