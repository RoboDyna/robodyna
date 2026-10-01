/* Sort Apples Belt — state transition.
 * Top-down: a belt carries ~6 apples toward a rotating gate at its end. The gate has three states:
 * left (-> red bin), right (-> green bin) and open (-> dump). Left / Right rotate it; pressing both
 * (or Down) opens it. The gate turns at finite speed, so each switch must be made before the next
 * apple reaches the end of the belt (apple gaps are randomized and can be short).
 * base:   alternating red / green, no rotten apple.
 * var1:   random color order.
 * var2:   alternating colors with one rotten (brown, spotted) apple that must go to the dump.
 * var1+2: random colors + one rotten apple.
 * Any apple in the wrong place fails.
 */
Arcade.register({
  id: "sort_apples_belt",
  maxTime: 22,
  levels: {
    base: "Alternating red and green apples.",
    var1: "Random color order.",
    var2: "Alternating colors plus one rotten apple for the dump.",
    "var1+2": "Random colors plus one rotten apple."
  },
  controls: [["←", "gate to the red bin"], ["→", "gate to the green bin"], ["← + → / ↓", "open the gate (dump)"]],
  keys: [{ key: "ArrowLeft", label: "◀ red" }, { key: "ArrowDown", label: "dump" }, { key: "ArrowRight", label: "green ▶" }],
  create(api) {
    const { P, W, rand, pick } = api;
    const BX = W / 2, BW = 116, TOP = 92, END = 318, R = 17;
    const PIV = { x: BX, y: END + 12 }, ARM = 70, TURN = 4.2;              // gate pivot, arm length, rad/s
    const ANG = { L: 0.95, O: 0, R: -0.95 };                               // arm angle from straight down (+ = toward left)
    const bins = {
      red: { x: 175, y: 440, w: 190, h: 110, col: P.red, name: "red bin" },
      green: { x: 635, y: 440, w: 190, h: 110, col: P.green, name: "green bin" },
      dump: { x: BX, y: 452, w: 130, h: 72, col: P.metalDark, name: "dump" }
    };
    const speed = rand(68, 86);
    // build the sequence
    const n = 6;
    let colors = [];
    if (!api.var1) { const first = pick(["red", "green"]); for (let i = 0; i < n; i++) colors.push(i % 2 ? (first === "red" ? "green" : "red") : first); }
    else {
      for (let i = 0; i < n; i++) colors.push(pick(["red", "green"]));
      if (colors.every(c => c === colors[0])) colors[api.randInt(1, n - 1)] = colors[0] === "red" ? "green" : "red";
    }
    if (api.var2) colors.splice(api.randInt(1, n - 2), 0, "rotten");
    const apples = [];
    let y = TOP + 6;
    colors.forEach((c, i) => {
      if (i) y -= speed * rand(0.95, 1.65);
      apples.push({ c, x: BX + rand(-14, 14), y, state: "belt", t: 0, rot: rand(0, 6) });
    });
    // gate starts pointing at the wrong bin for the first apple, so doing nothing fails
    const firstWrong = apples[0].c === "red" ? "R" : "L";
    let gate = firstWrong, ang = ANG[firstWrong], acted = false, sorted = 0;
    const held = {};

    function targetOf(a) { return a.c === "red" ? "red" : a.c === "green" ? "green" : "dump"; }
    function routeOf() {
      // nearest gate position at the moment the apple reaches the end
      const d = Object.entries(ANG).map(([k, v]) => [k, Math.abs(ang - v)]).sort((p, q) => p[1] - q[1])[0][0];
      return d === "L" ? "red" : d === "R" ? "green" : "dump";
    }
    function setGate(k) { if (gate !== k) { gate = k; api.sfx.click(); } acted = true; }

    return {
      keyDown(k) {
        held[k] = true;
        if (k === "ArrowLeft") setGate(held.ArrowRight ? "O" : "L");
        else if (k === "ArrowRight") setGate(held.ArrowLeft ? "O" : "R");
        else if (k === "ArrowDown") setGate("O");
      },
      keyUp(k) { held[k] = false; },
      update(dt) {
        const tgt = ANG[gate], step = TURN * dt;
        ang = Math.abs(tgt - ang) <= step ? tgt : ang + Math.sign(tgt - ang) * step;
        for (const a of apples) {
          a.rot += dt * speed / R * 0.3;
          if (a.state === "belt") {
            a.y += speed * dt;
            if (a.y >= END) {
              a.state = "route"; a.bin = routeOf(); a.t = 0; a.x0 = a.x; a.y0 = a.y; api.sfx.tick();
            }
          } else if (a.state === "route") {
            a.t += dt / 0.55;
            const b = bins[a.bin], k = Math.min(1, a.t);
            a.x = a.x0 + (b.x - a.x0) * k; a.y = a.y0 + (b.y - a.y0) * k;
            if (a.t >= 1) {
              a.state = "in"; a.x = b.x + rand(-b.w / 2 + 28, b.w / 2 - 28); a.y = b.y + rand(-b.h / 2 + 24, b.h / 2 - 22);
              const want = targetOf(a);
              if (a.bin !== want) {
                const what = a.c === "rotten" ? "The rotten apple" : `A ${a.c} apple`;
                return api.fail(`${what} went into the ${bins[a.bin].name}`);
              }
              sorted++; api.sfx.tick();
              if (sorted === apples.length) return api.succeed("All apples sorted");
            }
          }
        }
      },
      draw(g) {
        g.scene();
        const c = g.ctx;
        // bins
        for (const b of Object.values(bins)) {
          g.rr(b.x - b.w / 2 + 4, b.y - b.h / 2 + 6, b.w, b.h, 14, P.shadow);
          g.rr(b.x - b.w / 2, b.y - b.h / 2, b.w, b.h, 14, "#f4f6f8", b.col, 5);
          g.rr(b.x - b.w / 2 + 8, b.y - b.h / 2 + 8, b.w - 16, b.h - 16, 10, "rgba(15,34,51,0.06)");
        }
        g.text("DUMP", bins.dump.x, bins.dump.y + bins.dump.h / 2 - 12, { size: 11, color: P.muted, align: "center" });
        // belt with moving slats
        g.rr(BX - BW / 2 - 8, TOP - 26, BW + 16, END - TOP + 40, 10, P.metal, P.metalDark, 2);
        g.rr(BX - BW / 2, TOP - 20, BW, END - TOP + 28, 6, "#3c434d");
        c.save(); c.beginPath(); c.rect(BX - BW / 2, TOP - 20, BW, END - TOP + 28); c.clip();
        const off = (api.time * speed) % 24;
        for (let yy = TOP - 44 + off; yy < END + 10; yy += 24) g.line(BX - BW / 2, yy, BX + BW / 2, yy, "rgba(255,255,255,0.08)", 3);
        c.restore();
        // arrows hinting the motion
        g.text("▼", BX - BW / 2 - 22, (TOP + END) / 2, { size: 16, color: P.metal, align: "center" });
        // apples (on the belt below the feeder hood, routing, in bins)
        for (const a of apples) {
          if (a.state === "belt" && a.y < TOP - 30) continue;
          drawApple(g, a);
        }
        // feeder hood hides the queue
        g.rr(BX - BW / 2 - 14, 64, BW + 28, TOP - 50, 8, P.metalDark);
        g.rr(BX - BW / 2 - 6, 70, BW + 12, TOP - 62, 5, "#4f5964");
        // gate: arm pivoted at the end of the belt
        const ex = PIV.x + Math.sin(ang) * -ARM, ey = PIV.y + Math.cos(ang) * ARM;
        g.line(PIV.x, PIV.y, ex, ey, P.orange, 14);
        g.line(PIV.x, PIV.y, ex, ey, "#f7b267", 5);
        g.circle(PIV.x, PIV.y, 11, P.navy);
        const gl = { L: "→ red", R: "→ green", O: "→ dump" }[gate];
        g.chip("gate " + gl, PIV.x + 130, PIV.y - 6, gate === "L" ? P.red : gate === "R" ? P.green : P.metalDark);
        if (!acted) g.chip("← red bin · → green bin · ← + → (or ↓) dump rotten apples", W / 2, 521, P.navy);
      },
      dbg: { apples, gate: () => gate },
      hud() { return [{ k: "Sorted", v: `${sorted}/${apples.length}` }]; }
    };

    function drawApple(g, a) {
      const col = a.c === "red" ? P.red : a.c === "green" ? "#7dbb3f" : "#86592f";
      g.shadow(a.x + 2, a.y + 4, R, 0);
      g.ball(a.x, a.y, R, col);
      const sx = Math.cos(a.rot) * 4, sy = Math.sin(a.rot) * 4;
      g.line(a.x + sx, a.y + sy, a.x + sx * 2.2, a.y + sy * 2.2, "#5b3a1e", 3);
      g.ellipse(a.x + sx * 2.4 + 4, a.y + sy * 2.4 - 2, 6, 3, "#5f9f48", null, 0, a.rot);
      if (a.c === "rotten") for (const [dx, dy, rr] of [[-6, 4, 4.5], [7, -3, 3.5], [1, 9, 3], [-3, -8, 3]]) g.circle(a.x + dx, a.y + dy, rr, "rgba(55,32,14,0.8)");
    }
  }
});
