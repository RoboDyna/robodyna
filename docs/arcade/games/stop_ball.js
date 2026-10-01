/* Stop Ball — household.
 * Top-down cluttered table. A table-tennis ball is flicked across the table, bounces off the clutter
 * (books, a box, a mug, a jar) and heads for a table edge. Click once to drop the gripper-shaped U bridge:
 * its opening always turns to face the ball (preview follows the pointer), so place it ahead of the ball on
 * its path. The ball must roll into the U and be stopped there before it falls off the table.
 * Dropping the bridge on or over the ball fails; it has to roll in.
 * Randomized: clutter layout, ball start, direction and speed.
 */
Arcade.register({
  id: "stop_ball",
  maxTime: 12,
  levels: { base: "The ball bounces off the clutter toward an edge; catch it in the bridge." },
  controls: [["Click", "drop the U bridge there, opening facing the ball (only once)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const TB = { x0: 34, y0: 74, x1: 776, y1: 516 }, R = 8, IW = 46, D = 40, T = 8, DROP_T = 0.25;
    const KINDS = [
      { k: "book", w: 78, h: 54, col: P.blue }, { k: "book", w: 58, h: 76, col: P.green },
      { k: "box", w: 60, h: 60, col: P.wood }, { k: "mug", r: 22, col: P.red }, { k: "jar", r: 26, col: P.yellow }
    ];

    // ---- ball physics (shared by the game and the fairness pre-check) ----
    function collide(b, o) {
      let nx, ny, d, pen;
      if (o.r) { const dx = b.x - o.x, dy = b.y - o.y; d = Math.hypot(dx, dy); pen = o.r + R - d; nx = dx / d; ny = dy / d; }
      else {
        const qx = Math.max(o.x - o.w / 2, Math.min(b.x, o.x + o.w / 2)), qy = Math.max(o.y - o.h / 2, Math.min(b.y, o.y + o.h / 2));
        const dx = b.x - qx, dy = b.y - qy; d = Math.hypot(dx, dy) || 0.001; pen = R - d; nx = dx / d; ny = dy / d;
      }
      if (pen <= 0) return false;
      b.x += nx * pen; b.y += ny * pen;
      const vn = b.vx * nx + b.vy * ny;
      if (vn < 0) { b.vx -= 1.85 * vn * nx; b.vy -= 1.85 * vn * ny; b.hits++; b.lastHit = b.t; return true; }
      return false;
    }
    function stepBall(b, dt, obst) {
      if (b.wait > 0) { b.wait -= dt; return; }
      b.t += dt;
      const f = 1 - 0.1 * dt; b.vx *= f; b.vy *= f;
      b.x += b.vx * dt; b.y += b.vy * dt;
      for (const o of obst) if (collide(b, o)) b.bump = true;
      if (b.x < TB.x0 || b.x > TB.x1 || b.y < TB.y0 || b.y > TB.y1) b.off = true;
    }

    // ---- episode generation: clutter + a flick that bounces and then runs straight to an edge ----
    let clutter, ball, tries = 0;
    for (;;) {
      tries++;
      clutter = [];
      const kinds = KINDS.slice().sort(() => api.rng() - 0.5).slice(0, 4);
      for (const kd of kinds) {
        for (let k = 0; k < 40; k++) {
          const o = Object.assign({}, kd, { x: rand(130, 680), y: rand(150, 440) });
          const rad = o.r || Math.hypot(o.w, o.h) / 2;
          if (clutter.every(q => Math.hypot(q.x - o.x, q.y - o.y) > rad + (q.r || Math.hypot(q.w, q.h) / 2) + 40)) { clutter.push(o); break; }
        }
      }
      const a = rand(0, Math.PI * 2), sp = rand(170, 240);
      ball = { x: rand(150, 660), y: rand(150, 440), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, wait: rand(0.6, 1.1), t: 0, hits: 0, lastHit: 0, off: false };
      if (tries < 1500 && clutter.some(o => Math.hypot(o.x - ball.x, o.y - ball.y) < (o.r || Math.hypot(o.w, o.h) / 2) + 45)) continue;
      const c = Object.assign({}, ball);
      for (let i = 0; i < 960 && !c.off; i++) stepBall(c, 1 / 120, clutter);
      // must fall off within 8 s, after >= 1 bounce, with a final straight run of at least 0.9 s
      const ok = c.off && (c.hits >= 1 || tries > 300) && c.t - c.lastHit > 0.9 && c.t > 1.6;
      if (ok || tries > 1500) break;
    }

    let bridge = null, ghost = null, caught = false;
    const angTo = (x, y) => Math.atan2(ball.y - y, ball.x - x);  // opening (local +u) points at the ball
    // Bridge walls as capsule segments in world coordinates.
    function walls(br) {
      const c = Math.cos(br.a), s = Math.sin(br.a), hw = IW / 2 + T / 2;
      const P2 = (u, v) => [br.x + u * c - v * s, br.y + u * s + v * c];
      return [[P2(-D / 2, -hw), P2(-D / 2, hw)], [P2(-D / 2, -hw), P2(D / 2, -hw)], [P2(-D / 2, hw), P2(D / 2, hw)]];
    }
    function segDist(px, py, [[ax, ay], [bx, by]]) {
      const vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)));
      const qx = ax + vx * t, qy = ay + vy * t;
      return [Math.hypot(px - qx, py - qy), qx, qy];
    }
    const local = (br, x, y) => { const dx = x - br.x, dy = y - br.y, c = Math.cos(br.a), s = Math.sin(br.a); return [dx * c + dy * s, -dx * s + dy * c]; };
    const inside = (br, x, y) => { const [u, v] = local(br, x, y); return u > -D / 2 && u < D / 2 + 2 && Math.abs(v) < IW / 2; };
    function validSpot(x, y) {
      if (x < TB.x0 + 40 || x > TB.x1 - 40 || y < TB.y0 + 40 || y > TB.y1 - 40) return false;
      return clutter.every(o => Math.hypot(o.x - x, o.y - y) > (o.r || Math.hypot(o.w, o.h) / 2) + 34);
    }

    return {
      _dbg: () => ({ ball, clutter, tries, bridge }),          // inspected by the test harness only
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (bridge) return;
        if (!validSpot(x, y)) { api.sfx.tick(); return; }
        bridge = { x, y, a: angTo(x, y), t: 0 };
        api.sfx.whoosh();
      },
      // Fixed 1/120 s physics steps, so play matches the fairness pre-check whatever the frame rate.
      _acc: 0,
      update(dt) {
        this._acc += dt;
        while (this._acc >= 1 / 120 - 1e-9 && !api.ended) { this._acc -= 1 / 120; this.tick(1 / 120); }
      },
      tick(dt) {
        if (bridge && bridge.t < DROP_T) {
          bridge.t += dt;
          if (bridge.t >= DROP_T) {                               // lands: must not hit or cover the ball
            api.sfx.click();
            if (walls(bridge).some(w => segDist(ball.x, ball.y, w)[0] < T / 2 + R)) return api.fail("The bridge landed on the ball");
            if (inside(bridge, ball.x, ball.y)) return api.fail("The bridge was dropped over the ball");
          }
        }
        stepBall(ball, dt, clutter);
        if (ball.bump) { ball.bump = false; api.sfx.tick(); }
        if (bridge && bridge.t >= DROP_T) {
          for (const w of walls(bridge)) {                       // bounce off the U's walls
            const [d, qx, qy] = segDist(ball.x, ball.y, w);
            if (d < T / 2 + R) {
              const nx = (ball.x - qx) / d, ny = (ball.y - qy) / d, vn = ball.vx * nx + ball.vy * ny;
              ball.x = qx + nx * (T / 2 + R); ball.y = qy + ny * (T / 2 + R);
              if (vn < 0) { ball.vx -= 1.5 * vn * nx; ball.vy -= 1.5 * vn * ny; api.sfx.tick(); }
            }
          }
          if (inside(bridge, ball.x, ball.y)) {                  // inside the U: the soft lining soaks up speed
            caught = true; const f = Math.max(0, 1 - 7 * dt); ball.vx *= f; ball.vy *= f;
            if (Math.hypot(ball.vx, ball.vy) < 6) return api.succeed("The bridge stopped the ball");
          } else caught = false;
        }
        if (ball.off) api.fail("The ball fell off the table");
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // clutter
        for (const o of clutter) {
          if (o.r) {
            g.shadow(o.x + 3, o.y + 6, o.r, 0);
            g.circle(o.x, o.y, o.r, o.col, Arcade.shade(o.col, -0.3), 2);
            g.circle(o.x, o.y, o.r * 0.68, Arcade.shade(o.col, o.k === "mug" ? -0.35 : 0.35));
            if (o.k === "mug") g.rr(o.x + o.r - 2, o.y - 5, 12, 10, 4, Arcade.shade(o.col, -0.15));
          } else {
            g.rr(o.x - o.w / 2 + 3, o.y - o.h / 2 + 5, o.w, o.h, 4, P.shadow);
            g.rr(o.x - o.w / 2, o.y - o.h / 2, o.w, o.h, 4, o.col, Arcade.shade(o.col, -0.3), 2);
            if (o.k === "book") g.rr(o.x - o.w / 2 + 5, o.y - o.h / 2, 7, o.h, 2, Arcade.shade(o.col, -0.2));
            else { g.line(o.x - o.w / 2, o.y, o.x + o.w / 2, o.y, P.woodDark, 2); g.line(o.x, o.y - o.h / 2, o.x, o.y + o.h / 2, P.woodDark, 2); }
          }
        }
        // bridge (U shape, metal with soft pink lining) — placed, falling or ghost
        const drawBridge = (br, alpha, lift, bad) => {
          c.save(); c.globalAlpha = alpha;
          c.translate(br.x, br.y - lift); c.rotate(br.a);
          const hw = IW / 2 + T, col = bad ? P.red : P.metal, dk = bad ? "#9d3026" : P.metalDark;
          c.save(); c.translate(3, 5 + lift); g.rr(-D / 2 - T / 2, -hw, D + T / 2, hw * 2, 5, lift ? "rgba(15,34,51,0.08)" : P.shadow); c.restore();
          g.rr(-D / 2, -IW / 2, D, IW, 2, bad ? "rgba(224,75,60,0.12)" : "rgba(242,167,181,0.35)");
          g.rr(-D / 2 - T / 2, -hw, T, hw * 2, 3, col, dk, 1.5);
          g.rr(-D / 2 - T / 2, -hw, D + T / 2, T, 3, col, dk, 1.5);
          g.rr(-D / 2 - T / 2, IW / 2, D + T / 2, T, 3, col, dk, 1.5);
          c.restore();
        };
        if (bridge) drawBridge(bridge, 1, Math.max(0, 1 - bridge.t / DROP_T) * 40, false);
        else if (ghost) drawBridge({ x: ghost.x, y: ghost.y, a: angTo(ghost.x, ghost.y) }, 0.4, 0, !validSpot(ghost.x, ghost.y));
        // ball (table-tennis orange); drawn above the bridge lining
        if (!ball.off) { g.shadow(ball.x + 2, ball.y + 3, R * 0.9, 0); g.ball(ball.x, ball.y, R, P.orange); }
        if (!bridge) g.chip("Click once to drop the bridge in the ball's path", W / 2, 506, P.navy);
      }
    };
  }
});
