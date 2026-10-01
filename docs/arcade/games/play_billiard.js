/* Play Billiard — static to dynamic.
 * Top-down pool table with 6 pockets and a resting red ball. Click to put the cue tip at a spot, rotate
 * the stick with Left/Right, then Space makes one stroke with fixed power along the stick. The red ball
 * then rolls with friction, rebounds off cushions and collides with other balls. One strike per episode.
 * Robot contact (cue tip already touching a ball when the stroke starts) fails, as does striking another
 * ball or missing the red one.
 * base: red ball only, any pocket.
 * var1: one pocket is nominated; red in any other pocket fails.
 * var2: distractor balls on the table (any pocket).
 * var1+2: nominated pocket + distractors.
 */
Arcade.register({
  id: "play_billiard",
  maxTime: 30,
  levels: {
    base: "Red ball only; any pocket counts.",
    var1: "Only the marked target pocket counts.",
    var2: "Other balls on the table can block or deflect; any pocket counts.",
    "var1+2": "Marked target pocket and distractor balls."
  },
  controls: [["Click", "place the cue tip"], ["← / →", "rotate the stick"], ["Space", "strike (once)"]],
  keys: [{ key: "ArrowLeft", label: "⟲" }, { key: " ", label: "Strike" }, { key: "ArrowRight", label: "⟳" }],
  create(api) {
    const { P, W, rand, randInt, pick } = api;
    const X0 = 84, X1 = 726, Y0 = 104, Y1 = 476, R = 11, V0 = 640, FRIC = 150, REACH = 60, CUE_L = 300;
    const pockets = [[X0, Y0], [(X0 + X1) / 2, Y0 - 3], [X1, Y0], [X0, Y1], [(X0 + X1) / 2, Y1 + 3], [X1, Y1]]
      .map(([x, y], i) => ({ x, y, r: i === 1 || i === 4 ? 22 : 26 }));
    const segClear = (ax, ay, bx, by, px, py, d) => {      // is point p farther than d from segment a-b?
      const vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)));
      return Math.hypot(ax + vx * t - px, ay + vy * t - py) > d;
    };
    // Lay out the table so at least one allowed pocket has a clear straight shot (and room for the cue).
    let red, others, target, tries = 0;
    for (;;) {
      tries++;
      red = { name: "red", col: P.red, x: rand(220, 590), y: rand(180, 400), vx: 0, vy: 0, on: true };
      others = [];
      if (api.var2) {
        const cols = [["yellow", P.yellow], ["blue", P.blue], ["black", P.black], ["white", "#f4f4f0"]];
        const n = randInt(3, 4);
        for (let i = 0; i < n; i++) {
          let b, k = 0;
          do { b = { name: cols[i][0], col: cols[i][1], x: rand(X0 + 40, X1 - 40), y: rand(Y0 + 35, Y1 - 35), vx: 0, vy: 0, on: true }; k++; }
          while (k < 50 && ([red, ...others].some(o => Math.hypot(o.x - b.x, o.y - b.y) < 4 * R) || pockets.some(p => Math.hypot(p.x - b.x, p.y - b.y) < 60)));
          others.push(b);
        }
      }
      const clear = pockets.filter(p => {
        const ux = (p.x - red.x), uy = (p.y - red.y), L = Math.hypot(ux, uy);
        const cx = red.x - (ux / L) * 50, cy = red.y - (uy / L) * 50;       // where the cue would sit
        return others.every(o => segClear(red.x, red.y, p.x, p.y, o.x, o.y, 2 * R + 6) && Math.hypot(o.x - cx, o.y - cy) > 2 * R + 14);
      });
      if (api.var1) { if (clear.length) { target = pick(clear); break; } }
      else if (clear.length >= 2 || (tries > 40 && clear.length)) break;
    }
    const balls = [red, ...others];
    let cue = null, ang = rand(0, Math.PI * 2), ghost = null, stroke = null, struck = false, potted = [];
    const keys = { ArrowLeft: false, ArrowRight: false };

    function strike() {
      if (!cue || stroke || struck) return;
      for (const b of balls) if (Math.hypot(b.x - cue.x, b.y - cue.y) < R + 2)
        return api.fail(`Robot contact: the cue tip was touching the ${b.name} ball`);
      stroke = { t: 0, off: 0 }; api.sfx.whoosh();
    }
    function physics(dt) {
      for (const b of balls) {
        if (!b.on) continue;
        b.x += b.vx * dt; b.y += b.vy * dt;
        const sp = Math.hypot(b.vx, b.vy);
        if (sp > 0) { const ns = Math.max(0, sp - FRIC * dt) * (1 - 0.15 * dt); b.vx *= ns / sp; b.vy *= ns / sp; }
        for (const p of pockets) if (Math.hypot(b.x - p.x, b.y - p.y) < p.r) { b.on = false; b.pk = p; potted.push(b); api.sfx.tick(); break; }
        if (!b.on) continue;
        if (b.x < X0 + R) { b.x = X0 + R; b.vx = Math.abs(b.vx) * 0.8; api.sfx.tick(); }
        if (b.x > X1 - R) { b.x = X1 - R; b.vx = -Math.abs(b.vx) * 0.8; api.sfx.tick(); }
        if (b.y < Y0 + R) { b.y = Y0 + R; b.vy = Math.abs(b.vy) * 0.8; api.sfx.tick(); }
        if (b.y > Y1 - R) { b.y = Y1 - R; b.vy = -Math.abs(b.vy) * 0.8; api.sfx.tick(); }
      }
      for (let i = 0; i < balls.length; i++) for (let j = i + 1; j < balls.length; j++) {
        const a = balls[i], b = balls[j];
        if (!a.on || !b.on) continue;
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
        if (d >= 2 * R || d === 0) continue;
        const nx = dx / d, ny = dy / d, rel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
        const push = (2 * R - d) / 2; a.x -= nx * push; a.y -= ny * push; b.x += nx * push; b.y += ny * push;
        if (rel > 0) { const k = rel * 0.97; a.vx -= k * nx; a.vy -= k * ny; b.vx += k * nx; b.vy += k * ny; api.sfx.tick(); }
      }
    }

    return {
      debug: () => ({ red, balls, pockets, target, cue, ang }),   // test hook
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (struck || stroke) return;
        if (x < X0 - 10 || x > X1 + 10 || y < Y0 - 10 || y > Y1 + 10) return;
        cue = { x, y }; api.sfx.click();
      },
      keyDown(k) {
        if (k === " ") strike();
        if (k in keys) keys[k] = true;
      },
      keyUp(k) { if (k in keys) keys[k] = false; },
      update(dt) {
        if (!stroke && !struck) ang += ((keys.ArrowRight ? 1 : 0) - (keys.ArrowLeft ? 1 : 0)) * 0.75 * dt;
        const dx = Math.cos(ang), dy = Math.sin(ang);
        if (stroke) {                                       // pull back, then thrust forward
          stroke.t += dt;
          const prev = stroke.off;
          stroke.off = stroke.t < 0.15 ? -18 * (stroke.t / 0.15) : -18 + (stroke.t - 0.15) * 750;
          if (stroke.off > prev && !struck) {
            const tx = cue.x + dx * stroke.off, ty = cue.y + dy * stroke.off;
            for (const b of balls) if (b.on && Math.hypot(b.x - tx, b.y - ty) < R + 2) {
              if (b !== red) return api.fail(`The cue struck the ${b.name} ball`);
              red.vx = dx * V0; red.vy = dy * V0; struck = true; api.sfx.click(); break;
            }
          }
          if (stroke.off >= REACH) { stroke = null; if (!struck) return api.fail("The cue missed the red ball"); }
        }
        for (let i = 0; i < 4; i++) physics(dt / 4);
        if (!red.on) {
          if (api.var1 && red.pk !== target) return api.fail("The red ball dropped into the wrong pocket");
          return api.succeed("Red ball potted");
        }
        if (struck && !stroke && balls.every(b => !b.on || Math.hypot(b.vx, b.vy) < 3)) api.fail("The red ball stopped on the table");
      },
      hud() { return [{ k: "Pocket", v: api.var1 ? "target" : "any" }]; },
      draw(g) {
        const c = g.ctx;
        g.clear(P.floor);
        g.rr(X0 - 46, Y0 - 46, X1 - X0 + 92, Y1 - Y0 + 92, 22, P.shadow);
        g.rr(X0 - 50, Y0 - 50, X1 - X0 + 100, Y1 - Y0 + 100, 22, P.wood, P.woodDark, 2);
        g.rr(X0 - 6, Y0 - 6, X1 - X0 + 12, Y1 - Y0 + 12, 6, "#3e8a5c");
        g.rr(X0, Y0, X1 - X0, Y1 - Y0, 4, "#4f9e6a");
        for (const [x, y] of [[X0 - 25, (Y0 + Y1) / 2], [X1 + 25, (Y0 + Y1) / 2]]) g.circle(x, y, 3, "#f3e3c8");
        g.line(X0 + (X1 - X0) * 0.25, Y0, X0 + (X1 - X0) * 0.25, Y1, "rgba(255,255,255,0.18)", 1.5);
        for (const p of pockets) {
          if (target === p) g.circle(p.x, p.y, p.r + 7, "rgba(242,194,48,0.35)", P.yellow, 3);
          g.circle(p.x, p.y, p.r - 3, "#1d2228");
        }
        if (target) g.chip("Target", target.x, target.y + (target.y < 300 ? -38 : 38), P.orange);
        for (const b of balls) if (b.on) { g.shadow(b.x + 2, b.y + 4, R, 0); g.ball(b.x, b.y, R, b.col); }
        // cue stick: tip at cue point (+stroke offset), body extends backwards
        const dx = Math.cos(ang), dy = Math.sin(ang);
        const drawCue = (x, y, alpha) => {
          const off = stroke ? stroke.off : 0, tx = x + dx * off, ty = y + dy * off;
          c.save(); c.globalAlpha = alpha;
          g.line(tx - dx * 4 + 3, ty - dy * 4 + 6, tx - dx * CUE_L + 3, ty - dy * CUE_L + 6, "rgba(15,34,51,0.15)", 9);
          g.line(tx - dx * 40, ty - dy * 40, tx - dx * CUE_L, ty - dy * CUE_L, P.woodDark, 9);
          g.line(tx - dx * 6, ty - dy * 6, tx - dx * 40, ty - dy * 40, "#ecdcc0", 6);
          g.line(tx, ty, tx - dx * 6, ty - dy * 6, "#5b8fc9", 6);
          c.restore();
          if (!stroke && !struck) g.line(x + dx * 8, y + dy * 8, x + dx * REACH, y + dy * REACH, "rgba(255,255,255,0.45)", 1.5, [4, 5]);
        };
        if (cue && !(struck && !stroke && Math.hypot(red.vx, red.vy) < 1)) drawCue(cue.x, cue.y, struck && !stroke ? 0.35 : 1);
        else if (!cue && ghost && ghost.x > X0 && ghost.x < X1 && ghost.y > Y0 && ghost.y < Y1) drawCue(ghost.x, ghost.y, 0.35);
        if (!cue) g.chip("Click to place the cue tip · ← / → rotate · Space strike", W / 2, 27, P.navy);
        else if (!struck && !stroke) g.chip("← / → rotate the cue · Space strikes (once)", W / 2, 27, P.navy);
      }
    };
  }
});
