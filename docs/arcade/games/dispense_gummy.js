/* Dispense Gummy — static to dynamic.
 * Side view of a clear dispenser tube holding a stack of gummies in two colors. Space releases the
 * bottom gummy, which drops out of the chute and falls onto the counter below (fall time ~0.5 s).
 * A bowl rides on a rail under the chute: keep it under the chute for target-colored gummies and move it
 * away for the others (they land on the counter). Collect N target gummies with no wrong one in the bowl.
 * base: alternating color stack, bowl hops between 3 discrete stops (short move time).
 * var1: randomized stack order, discrete hops.
 * var2: alternating stack, bowl slides continuously while a key is held (with inertia).
 * var1+2: random stack + continuous motion.
 */
Arcade.register({
  id: "dispense_gummy",
  maxTime: 25,
  levels: {
    base: "Alternating stack; the bowl hops between three stops.",
    var1: "Random stack order; the bowl hops between three stops.",
    var2: "Alternating stack; the bowl slides while a key is held.",
    "var1+2": "Random stack order and sliding bowl."
  },
  controls: [["Space", "dispense one gummy"], ["← / →", "move the bowl"]],
  keys: [{ key: "ArrowLeft", label: "◀" }, { key: " ", label: "Dispense" }, { key: "ArrowRight", label: "▶" }],
  create(api) {
    const { P, W, rand, pick, chance } = api;
    const CHUTE_X = Math.round(rand(330, 480)), CHUTE_Y = 250, RIM_Y = 430, COUNTER_Y = 470;
    const GR = 13, BOWL_HW = 52, HOP = 165, HOP_T = 0.32, G = 1100, NEED = 3, COUNT = 10;
    const COLS = { red: P.red, green: P.green, yellow: P.yellow, blue: P.blue };
    const [tName, oName] = pick([["red", "green"], ["green", "red"], ["yellow", "blue"], ["blue", "yellow"]]);
    // Stack order (index 0 = bottom, dispensed first).
    let stack = [];
    if (!api.var1) {
      const first = chance(0.5);
      for (let i = 0; i < COUNT; i++) stack.push((i % 2 === 0) === first ? tName : oName);
    } else {
      do { stack = Array.from({ length: COUNT }, () => (chance(0.5) ? tName : oName)); }
      while (stack.filter(c => c === tName).length < NEED + 1 || stack.filter(c => c === oName).length < 3);
    }
    const stops = [CHUTE_X - HOP, CHUTE_X, CHUTE_X + HOP];
    const bowl = { x: CHUTE_X, v: 0, stop: 1, from: CHUTE_X, t: 1 };
    bowl.stop = pick([0, 2]); bowl.x = bowl.from = stops[bowl.stop];          // start away from the chute
    const keys = { ArrowLeft: false, ArrowRight: false };
    const falling = [], inBowl = [], onCounter = [];
    let cooldown = 0, acted = false, got = 0;

    function dispense() {
      if (!stack.length || cooldown > 0) return;
      const c = stack.shift();
      falling.push({ c, x: CHUTE_X + rand(-2, 2), y: CHUTE_Y + 6, vx: 0, vy: 40, rot: rand(-0.3, 0.3), vr: rand(-3, 3) });
      cooldown = 0.35; api.sfx.click();
    }
    function hop(d) {
      const n = Math.max(0, Math.min(2, bowl.stop + d));
      if (n === bowl.stop) return;
      bowl.from = bowl.x; bowl.stop = n; bowl.t = 0; api.sfx.whoosh();
    }
    function settle(gm) {
      if (gm.c === tName) { got += 1; api.sfx.tick(); } else return api.fail(`A ${oName} gummy landed in the bowl`);
      if (got >= NEED) api.succeed(`Collected ${NEED} ${tName} gummies`);
    }

    return {
      debug: () => ({ stack, bowl, falling, stops, tName, CHUTE_X }),   // test hook
      keyDown(k) {
        acted = true;
        if (k === " ") dispense();
        if (k in keys) { keys[k] = true; if (!api.var2) hop(k === "ArrowLeft" ? -1 : 1); }
      },
      keyUp(k) { if (k in keys) keys[k] = false; },
      update(dt) {
        cooldown -= dt;
        // bowl motion
        if (!api.var2) {
          bowl.t = Math.min(1, bowl.t + dt / HOP_T);
          const e = bowl.t * bowl.t * (3 - 2 * bowl.t);
          bowl.x = bowl.from + (stops[bowl.stop] - bowl.from) * e;
        } else {
          const dir = (keys.ArrowRight ? 1 : 0) - (keys.ArrowLeft ? 1 : 0);
          bowl.v += (dir * 300 - bowl.v) * Math.min(1, 7 * dt);       // eases toward 300 px/s
          bowl.x += bowl.v * dt;
          if (bowl.x < 90 || bowl.x > W - 90) { bowl.x = Math.max(90, Math.min(W - 90, bowl.x)); bowl.v = 0; }
        }
        // gummies
        for (let i = falling.length - 1; i >= 0; i--) {
          const gm = falling[i], py = gm.y;
          gm.vy += G * dt; gm.x += gm.vx * dt; gm.y += gm.vy * dt; gm.rot += gm.vr * dt;
          if (py < RIM_Y && gm.y >= RIM_Y) {
            const off = gm.x - bowl.x;
            if (Math.abs(off) < BOWL_HW - GR * 0.5) {
              falling.splice(i, 1); inBowl.push({ c: gm.c, dx: off * 0.7 + rand(-6, 6), dy: rand(0, 8), rot: gm.rot });
              settle(gm); continue;
            }
            if (Math.abs(off) < BOWL_HW + GR) { gm.vx = Math.sign(off) * 160; gm.vy = -180; api.sfx.tick(); }   // bounced off the rim
          }
          if (gm.y >= COUNTER_Y - GR * 0.6) {
            falling.splice(i, 1); onCounter.push({ c: gm.c, x: gm.x, rot: gm.rot }); api.sfx.tick();
          }
        }
        // out of target gummies?
        if (!api.ended && got + stack.filter(c => c === tName).length + falling.filter(f => f.c === tName).length < NEED)
          api.fail(`Not enough ${tName} gummies left`);
      },
      hud() { return [{ k: "Target", v: `${tName} ${got}/${NEED}` }, { k: "Left", v: String(stack.length) }]; },
      draw(g) {
        const c = g.ctx;
        g.clear(P.wall);
        for (let y = 40; y < COUNTER_Y; y += 60) g.line(0, y, W, y, P.wallLine, 1);
        // counter
        c.fillStyle = P.floor; c.fillRect(0, COUNTER_Y, W, 540 - COUNTER_Y);
        g.line(0, COUNTER_Y, W, COUNTER_Y, P.tableEdge, 3);
        // bowl rail and stops
        g.rr(stops[0] - 80, COUNTER_Y - 7, stops[2] - stops[0] + 160, 5, 2, P.metal);
        if (!api.var2) for (const sx of stops) g.circle(sx, COUNTER_Y - 4.5, 4, P.metalDark);
        // chute line marker on the counter
        g.line(CHUTE_X, COUNTER_Y + 6, CHUTE_X, COUNTER_Y + 18, P.muted, 2);
        // dispenser: stand, tube with stacked gummies, chute
        g.rr(CHUTE_X - 40, 0, 80, 10, 3, P.metalDark);                         // ceiling mount
        g.rr(CHUTE_X - 30, 8, 6, 30, 2, P.metal); g.rr(CHUTE_X + 24, 8, 6, 30, 2, P.metal);
        const tubeTop = 40, tubeW = 40;
        g.rr(CHUTE_X - tubeW / 2 - 4, tubeTop - 4, tubeW + 8, CHUTE_Y - tubeTop + 4, 8, "rgba(255,255,255,0.55)", P.metal, 2);
        const drawGummy = (x, y, col, rot, s) => {
          c.save(); c.translate(x, y); c.rotate(rot || 0); s = s || 1;
          g.rr(-GR * s, -GR * 0.8 * s, GR * 2 * s, GR * 1.6 * s, 6 * s, COLS[col], Arcade.shade(COLS[col], -0.25), 1.5);
          g.ellipse(-GR * 0.35 * s, -GR * 0.35 * s, GR * 0.35 * s, GR * 0.18 * s, "rgba(255,255,255,0.55)");
          c.restore();
        };
        const visible = Math.min(stack.length, 8);
        for (let i = 0; i < visible; i++) drawGummy(CHUTE_X, CHUTE_Y - 16 - i * 23, stack[i], 0);
        g.rr(CHUTE_X - 26, CHUTE_Y - 2, 52, 14, 4, P.metalDark);                // chute gate
        g.rr(CHUTE_X - 8, CHUTE_Y + 2, 16, 6, 2, cooldown > 0 ? P.yellow : P.navy);
        // counter gummies, falling gummies
        for (const gm of onCounter) drawGummy(gm.x, COUNTER_Y - GR * 0.7, gm.c, gm.rot * 0.2, 0.9);
        for (const gm of falling) drawGummy(gm.x, gm.y, gm.c, gm.rot);
        // bowl (side view) with collected gummies
        const bx = bowl.x;
        g.ellipse(bx, COUNTER_Y - 3, BOWL_HW * 0.9, 5, P.shadow);
        c.beginPath(); c.moveTo(bx - BOWL_HW, RIM_Y);
        c.quadraticCurveTo(bx - BOWL_HW + 4, COUNTER_Y - 8, bx, COUNTER_Y - 8);
        c.quadraticCurveTo(bx + BOWL_HW - 4, COUNTER_Y - 8, bx + BOWL_HW, RIM_Y);
        c.closePath(); c.fillStyle = "#e8eef4"; c.fill(); c.strokeStyle = P.metal; c.lineWidth = 2; c.stroke();
        for (const gm of inBowl) drawGummy(bx + gm.dx, RIM_Y + 12 + gm.dy, gm.c, gm.rot * 0.2, 0.85);
        g.ellipse(bx, RIM_Y, BOWL_HW, 6, "rgba(154,166,178,0.25)", P.metal, 2);
        // target card
        g.rr(30, 30, 150, 54, 10, "#fff", P.line, 1.5);
        g.text("Collect", 44, 46, { size: 12, color: P.muted });
        drawGummy(56, 66, tName, 0, 0.75);
        g.text(`${got} / ${NEED}`, 80, 66, { size: 18, weight: 800 });
        if (!acted) g.chip(api.var2 ? "Space: dispense · hold ← / → to slide the bowl" : "Space: dispense · ← / → hop the bowl", W / 2, 510, P.navy);
      }
    };
  }
});
