/* Marble Shelf Maze — static to dynamic.
 * Side view of four short shelves pivoting on the wall, staggered so that only ONE end of each shelf
 * sits above the next shelf (the other end drops the marble to the floor). The shelf the marble is on
 * is the active one (highlighted): hold Left/Right to tilt it. The marble rolls with the tilt (and keeps
 * its momentum when it lands), falls off the low end onto the next shelf, and finally into the bowl.
 * base: the marble waits on the flat top shelf until you tilt; the bowl is stationary.
 * var1: the marble is already rolling and the shelves start tilted (react quickly, counter-tilt).
 * var2: paused marble, the bowl slides back and forth below, so the last drop must be timed.
 * var1+2: rolling marble + oscillating bowl.
 */
Arcade.register({
  id: "marble_shelf_maze",
  maxTime: 25,
  levels: {
    base: "The marble waits until you tilt; the bowl stays put.",
    var1: "The marble is already rolling and the shelves start tilted.",
    var2: "The marble waits, but the bowl slides back and forth.",
    "var1+2": "Rolling marble and sliding bowl."
  },
  controls: [["← / →", "hold to tilt the highlighted shelf"]],
  keys: [{ key: "ArrowLeft", label: "◀ tilt" }, { key: "ArrowRight", label: "tilt ▶" }],
  create(api) {
    const { P, W, rand, pick, chance } = api;
    const G = 900, ROLL = 5 / 7, R = 9, THICK = 10, HALF = 118, STEP = 140, MAXT = 0.3, TILT_RATE = 0.9;
    const RIM_Y = 468, FLOOR_Y = 505, BOWL_HW = 44;
    const YS = [125, 210, 295, 380];
    // Build the staggered column: shelf k+1 sits STEP px to one side of shelf k.
    const shelves = [];
    let cx = rand(330, 480);
    for (let k = 0; k < YS.length; k++) {
      shelves.push({ cx, cy: YS[k], th: 0 });
      let side = pick([-1, 1]);
      if (k < YS.length - 1) {
        if (cx + side * STEP < 160 || cx + side * STEP > W - 160) side = -side;
        shelves[k].side = side; cx += side * STEP;
      }
    }
    const last = shelves[YS.length - 1];
    last.side = last.cx < W / 2 ? 1 : last.cx > W / 2 ? -1 : pick([-1, 1]);
    if (chance(0.35) && last.cx - last.side * (HALF + 60) > 80 && last.cx - last.side * (HALF + 60) < W - 80) last.side = -last.side;
    // var1: every shelf starts tilted toward its dead end, and the marble already rolls that way.
    if (api.var1) shelves.forEach(s => (s.th = -s.side * rand(0.07, 0.13)));
    const bowl0 = last.cx + last.side * (HALF + 52);
    const osc = api.var2 ? { amp: rand(60, 90), w: (2 * Math.PI) / rand(2.6, 3.6), ph: rand(0, 6.28) } : null;
    const bowlX = () => (osc ? bowl0 + osc.amp * Math.sin(osc.w * api.time + osc.ph) : bowl0);

    const m = { st: "shelf", sh: 0, s: api.var1 ? rand(-30, 30) : 0, v: api.var1 ? -shelves[0].side * rand(40, 70) : 0, x: 0, y: 0, vx: 0, vy: 0 };
    let started = !!api.var1, acted = false, inOff = 0;
    const keys = { ArrowLeft: false, ArrowRight: false };

    function surf(s, d, lift) {
      const c = Math.cos(s.th), sn = Math.sin(s.th);
      return { x: s.cx + d * c + sn * lift, y: s.cy + d * sn - c * lift };
    }
    function place() { const p = surf(shelves[m.sh], m.s, THICK / 2 + R); m.x = p.x; m.y = p.y; }
    place();

    return {
      debug: () => ({ m, shelves, bowlX: bowlX(), started }),   // test hook
      keyDown(k) { if (k in keys) { keys[k] = true; acted = true; started = true; } },
      keyUp(k) { if (k in keys) keys[k] = false; },
      update(dt) {
        if (m.st === "in") return;
        const dir = (keys.ArrowRight ? 1 : 0) - (keys.ArrowLeft ? 1 : 0);
        if (m.st === "shelf" || m.st === "air") {
          const act = shelves[m.sh];
          act.th = Math.max(-MAXT, Math.min(MAXT, act.th + dir * TILT_RATE * dt));
        }
        if (!started) { place(); return; }
        if (m.st === "shelf") {
          const s = shelves[m.sh];
          m.v += G * ROLL * Math.sin(s.th) * dt;
          m.v *= 1 - 0.1 * dt;
          m.s += m.v * dt;
          place();
          if (Math.abs(m.s) > HALF + 2) {                  // rolled off an end
            m.vx = m.v * Math.cos(s.th); m.vy = m.v * Math.sin(s.th); m.st = "air"; m.from = m.sh;
            api.sfx.whoosh();
          }
          return;
        }
        // airborne
        const px = m.x, py = m.y;
        m.vy += G * dt; m.x += m.vx * dt; m.y += m.vy * dt;
        for (let k = 0; k < shelves.length; k++) {
          if (k === m.from) continue;
          const s = shelves[k], c = Math.cos(s.th), sn = Math.sin(s.th);
          const d = (m.x - s.cx) * c + (m.y - s.cy) * sn;
          if (Math.abs(d) > HALF) continue;
          const top = s.cy + d * sn - THICK / 2 - R;
          if (py <= top + 2 && m.y >= top && py < s.cy) {
            m.st = "shelf"; m.sh = k; m.s = d;
            m.v = (m.vx * c + m.vy * sn) * 0.5;            // keeps part of its momentum
            place(); api.sfx.tick(); return;
          }
        }
        if (py < RIM_Y && m.y >= RIM_Y) {
          const off = m.x - bowlX();
          if (Math.abs(off) < BOWL_HW - R * 0.6) { m.st = "in"; inOff = off; api.sfx.tick(); return api.succeed("The marble is in the bowl"); }
          if (Math.abs(off) < BOWL_HW + R) { m.vx = Math.sign(off) * 150; m.vy = -170; api.sfx.tick(); }
        }
        if (m.y > FLOOR_Y - R) { m.y = FLOOR_Y - R; return api.fail("The marble fell to the floor"); }
        if (m.x < -20 || m.x > W + 20) return api.fail("The marble left the maze");
      },
      draw(g) {
        const c = g.ctx;
        g.clear(P.wall);
        for (let y = 40; y < FLOOR_Y; y += 60) g.line(0, y, W, y, P.wallLine, 1);
        c.fillStyle = P.floor; c.fillRect(0, FLOOR_Y, W, 540 - FLOOR_Y);
        g.line(0, FLOOR_Y, W, FLOOR_Y, P.tableEdge, 3);
        if (osc) g.rr(bowl0 - osc.amp - BOWL_HW, FLOOR_Y - 6, osc.amp * 2 + BOWL_HW * 2, 5, 2, P.metal);
        // shelves
        shelves.forEach((s, k) => {
          const active = k === m.sh && m.st !== "in";
          c.save(); c.translate(s.cx, s.cy); c.rotate(s.th);
          g.rr(-HALF + 3, -THICK / 2 + 5, HALF * 2, THICK, 3, P.shadow);
          if (active) g.rr(-HALF - 5, -THICK / 2 - 5, HALF * 2 + 10, THICK + 10, 8, "rgba(19,154,154,0.22)");
          g.rr(-HALF, -THICK / 2, HALF * 2, THICK, 3, P.wood, active ? P.teal : P.woodDark, active ? 2.5 : 1.5);
          c.restore();
          g.circle(s.cx, s.cy, 7, active ? P.teal : P.metal, P.metalDark, 2);
          g.circle(s.cx, s.cy, 2.5, P.metalDark);
        });
        // bowl (side view)
        const bx = bowlX();
        g.ellipse(bx, FLOOR_Y - 2, BOWL_HW * 0.9, 5, P.shadow);
        c.beginPath(); c.moveTo(bx - BOWL_HW, RIM_Y);
        c.quadraticCurveTo(bx - BOWL_HW + 4, FLOOR_Y - 9, bx, FLOOR_Y - 9);
        c.quadraticCurveTo(bx + BOWL_HW - 4, FLOOR_Y - 9, bx + BOWL_HW, RIM_Y);
        c.closePath(); c.fillStyle = P.blue; c.fill();
        g.ellipse(bx, RIM_Y, BOWL_HW, 6, "#24558f", "#1f4f8a", 2);
        if (m.st === "in") g.ball(bx + inOff * 0.6, RIM_Y + 6, R, P.red);
        else g.ball(m.x, m.y, R, P.red);
        if (!acted) g.chip(api.var1 ? "Hold ← / → to tilt the highlighted shelf — it's rolling!" : "Hold ← / → to tilt the highlighted shelf", W / 2, 524, P.navy);
      }
    };
  }
});
