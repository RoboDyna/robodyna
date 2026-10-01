/* Catch Mouse Object Drop — household.
 * Side view of a wall shelf above a table. Four objects stand on the shelf in two rows (back and
 * front), one on each side of a mouse hole. A toy mouse comes out of the hole, picks a row and a
 * direction, runs to the object on that side and pushes it off the shelf end. The object leaves the
 * edge with a horizontal speed that depends on the mouse speed and on the object (a ball rolls ahead
 * faster, a box drags slower) and falls in a parabola to the table.
 * Click once on the table to set the pillow-lined basket down there (it takes ~0.4 s to lower).
 * Success: the falling object lands in the basket. Fail: it hits the table or the basket rim.
 */
Arcade.register({
  id: "catch_mouse_object_drop",
  maxTime: 14,
  levels: { base: "A toy mouse pushes one shelf object off the edge. Catch it in the basket." },
  controls: [["Click", "set the basket down on the table (only once)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const TABLE_Y = 440, SHELF_Y = rand(175, 215), SL = rand(205, 245), SR = rand(565, 605), MID = (SL + SR) / 2;
    const ROWS = [SHELF_Y - 24, SHELF_Y - 2], DEPTH = 34, G = 1100, B_W = 116, B_H = 54, LOWER_T = 0.4;
    const KINDS = {
      can: { w: 30, h: 44, color: P.red, f: 1.0, name: "can" },
      box: { w: 42, h: 36, color: P.blue, f: 0.8, name: "box" },
      ball: { w: 32, h: 32, color: P.yellow, f: 1.45, name: "ball" },
      cup: { w: 34, h: 34, color: P.green, f: 0.95, name: "cup" }
    };
    const kinds = ["can", "box", "ball", "cup"];
    for (let i = 3; i > 0; i--) { const j = Math.floor(api.rng() * (i + 1)); [kinds[i], kinds[j]] = [kinds[j], kinds[i]]; }
    const objs = [];
    [0, 1].forEach(row => [-1, 1].forEach(side => {
      const k = KINDS[kinds[objs.length]];
      // back and front objects on the same side are kept apart so both stay visible
      const prev = objs.find(o => o.side === side);
      let x = side < 0 ? rand(SL + 30, MID - 60) : rand(MID + 60, SR - 30);
      if (prev && Math.abs(x - prev.x) < 50) x = side < 0 ? (prev.x > (SL + MID) / 2 - 15 ? prev.x - 55 : prev.x + 55) : (prev.x < (MID + SR) / 2 + 15 ? prev.x + 55 : prev.x - 55);
      objs.push(Object.assign({ row, side, x, y: ROWS[row], vx: 0, vy: 0, ang: 0, state: "rest" }, k));
    }));
    const mRow = pick([0, 1]), mDir = pick([-1, 1]);
    const target = objs.find(o => o.row === mRow && o.side === mDir);
    const mouse = { x: MID, y: ROWS[mRow], dir: mDir, v: rand(80, 150), state: "hidden", t: rand(0.6, 1.4), show: 0, gait: 0 };
    let basket = null, ghost = null;

    function drawObj(g, o, dim) {
      const ctx = g.ctx; ctx.save(); ctx.translate(o.x, o.y); ctx.rotate(o.ang); if (dim) ctx.globalAlpha = 0.9;
      const w = o.w, h = o.h;
      if (o.name === "can") { g.rr(-w / 2, -h, w, h, 4, o.color, Arcade.shade(o.color, -0.25), 1.5); g.rr(-w / 2, -h + 10, w, 8, 0, "rgba(255,255,255,0.55)"); }
      else if (o.name === "box") { g.rr(-w / 2, -h, w, h, 3, o.color, Arcade.shade(o.color, -0.25), 1.5); g.line(0, -h, 0, 0, "rgba(255,255,255,0.45)", 4); }
      else if (o.name === "ball") g.ball(0, -h / 2, w / 2, o.color);
      else {
        g.poly([[-w / 2, -h], [w / 2, -h], [w / 2 - 5, 0], [-w / 2 + 5, 0]], o.color, Arcade.shade(o.color, -0.25), 1.5);
        g.circle(w / 2 + 2, -h / 2, 8, null, Arcade.shade(o.color, -0.2), 4);
      }
      ctx.restore();
    }
    function drawMouse(g) {
      if (mouse.state === "hidden") return;
      const ctx = g.ctx; ctx.save(); ctx.translate(mouse.x, mouse.y); ctx.scale(mouse.dir, 1);
      ctx.globalAlpha = Math.min(1, mouse.show);
      const hop = Math.abs(Math.sin(mouse.gait)) * 2;
      ctx.beginPath(); ctx.moveTo(-14, -6); ctx.quadraticCurveTo(-30, -2, -36, -12); ctx.strokeStyle = "#b07a7a"; ctx.lineWidth = 2; ctx.stroke();
      g.ellipse(0, -9 - hop, 16, 9, "#9aa1a8", "#6f7880", 1.5);
      g.poly([[10, -14 - hop], [24, -7 - hop], [10, -3 - hop]], "#9aa1a8", "#6f7880", 1.5);
      g.circle(6, -18 - hop, 5, P.pink, "#6f7880", 1.2);
      g.circle(16, -10 - hop, 1.6, P.ink); g.circle(24, -7 - hop, 2, P.pink);
      ctx.restore();
    }
    function drawBasket(g, x, lift, alpha) {
      const ctx = g.ctx, y = TABLE_Y - lift; ctx.save(); ctx.globalAlpha = alpha;
      if (lift < 200) g.ellipse(x, TABLE_Y + 4, B_W / 2 * (1 - lift / 400), 6, `rgba(15,34,51,${0.18 * (1 - lift / 200)})`);
      g.ellipse(x, y - B_H, B_W / 2 - 4, 9, "#e9a3b4");                    // pillow lining seen over the rim
      g.poly([[x - B_W / 2, y - B_H], [x + B_W / 2, y - B_H], [x + B_W / 2 - 10, y], [x - B_W / 2 + 10, y]], P.wood, P.woodDark, 2);
      for (let i = 1; i < 4; i++) g.line(x - B_W / 2 + 3 + i * 1.5, y - B_H + i * 13, x + B_W / 2 - 3 - i * 1.5, y - B_H + i * 13, "rgba(120,85,45,0.45)", 1.5);
      for (let i = -2; i <= 2; i++) g.line(x + i * 20, y - B_H + 2, x + i * 17, y - 2, "rgba(120,85,45,0.35)", 1.5);
      g.rr(x - B_W / 2 - 3, y - B_H - 4, B_W + 6, 8, 4, P.woodDark);
      ctx.restore();
    }

    return {
      pointerMove(x, y) { ghost = { x: Math.max(70, Math.min(W - 70, x)) }; },
      pointerDown(x, y) {
        if (basket || y < SHELF_Y + 40) return;
        basket = { x: Math.max(70, Math.min(W - 70, x)), t: 0, down: false }; api.sfx.click();
      },
      update(dt) {
        if (basket && !basket.down) { basket.t += dt; if (basket.t >= LOWER_T) { basket.down = true; api.sfx.tick(); } }
        // mouse: wait in the hole, peek out, then run to its object and push it off the end
        const m = mouse;
        if (m.state === "hidden") { m.t -= dt; if (m.t <= 0) { m.state = "peek"; m.t = 0.55; } }
        else if (m.state === "peek") { m.show += dt * 3; m.t -= dt; if (m.t <= 0) m.state = "run"; }
        else if (m.state === "run" || m.state === "push") {
          m.x += m.dir * m.v * dt; m.gait += dt * 14;
          const front = m.x + m.dir * 24;
          if (m.state === "run" && (front - (target.x - m.dir * target.w / 2)) * m.dir >= 0) {
            // contact: a heavy object slows the mouse down; a ball rolls ahead faster
            m.state = "push"; target.state = "slide"; target.vx = m.dir * m.v * target.f; m.v = Math.min(m.v, m.v * target.f); api.sfx.tick();
          }
          if (m.dir > 0 ? m.x > SR - 26 : m.x < SL + 26) { m.state = "stop"; m.x = m.dir > 0 ? SR - 26 : SL + 26; }
        }
        for (const o of objs) {
          if (o.state === "slide") {
            o.x += o.vx * dt; if (o.name === "ball") o.ang += o.vx / (o.w / 2) * dt;
            if ((o.x - (mouse.dir > 0 ? SR : SL)) * mouse.dir > 0) { o.state = "fall"; o.vy = -10; api.sfx.whoosh(); }
          }
          if (o.state !== "fall") continue;
          const prevB = o.y;
          o.x += o.vx * dt; o.vy += G * dt; o.y += o.vy * dt; o.ang += o.vx * (o.name === "ball" ? 1 / (o.w / 2) : 0.012) * dt;
          const rimY = TABLE_Y - B_H;
          if (basket && prevB < rimY && o.y >= rimY) {
            const dx = Math.abs(o.x - basket.x);
            if (basket.down && dx < B_W / 2 - 6 - o.w * 0.3) { o.state = "caught"; o.y = rimY + 14; o.ang = 0; return api.succeed(`Caught the ${o.name} in the basket`); }
            if (dx < B_W / 2 + o.w / 2) {
              o.state = "landed";
              return api.fail(basket.down ? `The ${o.name} hit the basket rim` : "The basket was not set down in time");
            }
          }
          if (o.y >= TABLE_Y) { o.y = TABLE_Y; o.state = "landed"; return api.fail(`The ${o.name} hit the table`); }
        }
      },
      draw(g) {
        const ctx = g.ctx;
        g.clear(P.wall);
        for (let y = 50; y < TABLE_Y; y += 64) g.line(0, y, W, y, P.wallLine, 1);
        g.rr(-10, TABLE_Y, W + 20, 120, 0, P.table); g.line(0, TABLE_Y, W, TABLE_Y, P.tableEdge, 3);
        // shelf: brackets, back strip (top surface) and front edge; mouse hole in the wall
        g.poly([[SL + 30, SHELF_Y + 8], [SL + 30, SHELF_Y + 60], [SL + 40, SHELF_Y + 8]], P.metalDark);
        g.poly([[SR - 30, SHELF_Y + 8], [SR - 30, SHELF_Y + 60], [SR - 40, SHELF_Y + 8]], P.metalDark);
        g.poly([[SL, SHELF_Y - DEPTH], [SR, SHELF_Y - DEPTH], [SR, SHELF_Y], [SL, SHELF_Y]], "#d9bf98");
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, SHELF_Y - DEPTH); ctx.clip();
        g.ellipse(MID, SHELF_Y - DEPTH, 16, 15, "#3a2a20"); ctx.restore();
        g.line(SL, SHELF_Y - DEPTH, SR, SHELF_Y - DEPTH, "rgba(120,85,45,0.3)", 1.5);
        for (const o of objs) if (o.row === 0 && (o.state === "rest" || o.state === "slide")) drawObj(g, o, true);
        if (mRow === 0) drawMouse(g);
        for (const o of objs) if (o.row === 1 && (o.state === "rest" || o.state === "slide")) drawObj(g, o, false);
        if (mRow === 1) drawMouse(g);
        g.rr(SL - 4, SHELF_Y, SR - SL + 8, 12, 3, P.wood, P.woodDark, 2);
        // basket (ghost before placing), then falling / landed objects
        if (basket) { const f = Math.min(1, basket.t / LOWER_T); drawBasket(g, basket.x, (1 - f) * (1 - f) * 120, 1); }
        else if (ghost) drawBasket(g, ghost.x, 0, 0.3);
        for (const o of objs) if (o.state !== "rest" && o.state !== "slide") {
          if (o.state === "caught") { ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, TABLE_Y - B_H + 2); ctx.clip(); drawObj(g, o); ctx.restore(); }
          else drawObj(g, o);
        }
        if (!basket) g.chip("Watch the mouse, then click once on the table to set the basket down", W / 2, 505, P.navy);
      }
    };
  }
});
