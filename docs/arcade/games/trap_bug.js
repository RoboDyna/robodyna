/* Trap Bug — household.
 * Top-down table in front of a bookshelf. A bug scurries in bursts and pauses: random walk with
 * momentum, sudden turns, and it steers away from the table edges. Click once: a transparent square
 * glass box drops from just above that spot and lands ~0.3 s later. Success if the bug is fully inside
 * the glass footprint when it lands; otherwise the bug escapes. One drop only.
 */
Arcade.register({
  id: "trap_bug",
  maxTime: 12,
  levels: { base: "A bug scurries in bursts and pauses. Drop the glass once." },
  controls: [["Click", "drop the glass box there (only once)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const AREA = { x0: 75, y0: 165, x1: 735, y1: 495 }, BOX = 104, FALL_T = 0.3, BUG_L = 30, BUG_W = 17;
    const bug = { x: rand(250, 560), y: rand(250, 420), h: rand(0, Math.PI * 2), v: 0, turn: 0,
      mode: "pause", t: rand(0.3, 0.7), vt: 0, gait: 0 };
    let glass = null, ghost = null;
    const books = Array.from({ length: 30 }, (_, i) => ({ w: rand(22, 38), h: rand(0.65, 1), c: pick([P.blue, P.red, P.green, P.yellow, P.teal, P.purple, P.orange]) }));

    function steerBug(dt) {
      bug.t -= dt;
      if (bug.t <= 0) {
        if (bug.mode === "pause") {
          bug.mode = "run"; bug.t = rand(0.35, 0.95); bug.vt = rand(150, 240);
          bug.h += rand(-1.6, 1.6); bug.turn = rand(-2.5, 2.5);
        } else { bug.mode = "pause"; bug.t = rand(0.2, 0.65); bug.vt = 0; }
      }
      if (bug.mode === "run") {
        bug.turn += rand(-14, 14) * dt; bug.turn = Math.max(-3.5, Math.min(3.5, bug.turn));
        bug.h += bug.turn * dt;
        // avoid edges: turn toward the table center when getting close
        const m = 70, near = bug.x < AREA.x0 + m || bug.x > AREA.x1 - m || bug.y < AREA.y0 + m || bug.y > AREA.y1 - m;
        if (near) {
          const want = Math.atan2((AREA.y0 + AREA.y1) / 2 - bug.y, (AREA.x0 + AREA.x1) / 2 - bug.x);
          let d = want - bug.h; d = Math.atan2(Math.sin(d), Math.cos(d));
          bug.h += Math.sign(d) * Math.min(Math.abs(d), 6 * dt);
        }
      }
      bug.v += (bug.vt - bug.v) * Math.min(1, 12 * dt);
      bug.x = Math.max(AREA.x0, Math.min(AREA.x1, bug.x + Math.cos(bug.h) * bug.v * dt));
      bug.y = Math.max(AREA.y0, Math.min(AREA.y1, bug.y + Math.sin(bug.h) * bug.v * dt));
      bug.gait += bug.v * dt * 0.25;
    }
    function bugInside(cx, cy) {
      const c = Math.cos(bug.h), s = Math.sin(bug.h), half = BOX / 2 - 3;
      for (const [a, b] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const px = bug.x + c * a * BUG_L / 2 - s * b * BUG_W / 2, py = bug.y + s * a * BUG_L / 2 + c * b * BUG_W / 2;
        if (Math.abs(px - cx) > half || Math.abs(py - cy) > half) return false;
      }
      return true;
    }
    const clampC = (x, y) => ({ x: Math.max(AREA.x0 - 20, Math.min(AREA.x1 + 20, x)), y: Math.max(AREA.y0 - 10, Math.min(AREA.y1 + 10, y)) });

    function drawBug(g) {
      const ctx = g.ctx; ctx.save(); ctx.translate(bug.x, bug.y); ctx.rotate(bug.h);
      g.ellipse(2, 3, BUG_L / 2 + 2, BUG_W / 2 + 1, "rgba(15,34,51,0.15)");
      for (const sd of [-1, 1]) for (let i = -1; i <= 1; i++) {
        const sw = Math.sin(bug.gait + i * 2 + (sd > 0 ? Math.PI : 0)) * 4;
        g.line(i * 6, sd * 4, i * 7 + sw, sd * 13, "#2b1d14", 1.6);
      }
      g.line(BUG_L / 2 - 2, -2, BUG_L / 2 + 8, -7, "#2b1d14", 1.2); g.line(BUG_L / 2 - 2, 2, BUG_L / 2 + 8, 7, "#2b1d14", 1.2);
      g.ellipse(-2, 0, BUG_L / 2 - 2, BUG_W / 2, "#5a3b26", "#2b1d14", 1.2);
      g.line(-BUG_L / 2 + 2, 0, BUG_L / 2 - 6, 0, "rgba(0,0,0,0.35)", 1);
      g.circle(BUG_L / 2 - 3, 0, 5, "#2b1d14");
      ctx.restore();
    }
    function drawGlass(g, x, y, lift, alpha) {
      const ctx = g.ctx, k = 1 + lift * 0.35, s = BOX * k;
      ctx.save(); ctx.globalAlpha = alpha;
      g.rr(x - BOX / 2 + 4, y - BOX / 2 + 6, BOX, BOX, 6, `rgba(15,34,51,${0.1 * (1 - lift)})`);
      g.rr(x - s / 2, y - s / 2 - lift * 40, s, s, 6, "rgba(170,215,240,0.28)", "rgba(60,130,175,0.85)", 3);
      g.rr(x - s / 2 + 7, y - s / 2 + 7 - lift * 40, s - 14, s - 14, 4, null, "rgba(255,255,255,0.7)", 1.5);
      g.line(x - s / 2 + 14, y - s / 2 + 26 - lift * 40, x - s / 2 + 34, y - s / 2 + 12 - lift * 40, "rgba(255,255,255,0.9)", 3);
      ctx.restore();
    }

    return {
      pointerMove(x, y) { ghost = clampC(x, y); },
      pointerDown(x, y) {
        if (glass || y < AREA.y0 - 40) return;
        const c = clampC(x, y); glass = { x: c.x, y: c.y, t: 0, down: false }; api.sfx.whoosh();
      },
      update(dt) {
        if (glass && glass.down) return;
        steerBug(dt);
        if (glass) {
          glass.t += dt;
          if (glass.t >= FALL_T) {
            glass.down = true; api.sfx.tick();
            if (bugInside(glass.x, glass.y)) api.succeed("The bug is trapped under the glass");
            else api.fail("The bug was not fully under the glass");
          }
        }
      },
      draw(g) {
        g.scene();
        // bookshelf at the back of the table
        g.rr(60, 76, W - 120, 62, 6, P.woodDark);
        let bx = 72;
        for (const b of books) { if (bx + b.w > W - 72) break; g.rr(bx, 84 + (1 - b.h) * 46, b.w - 3, 46 * b.h, 3, b.c); bx += b.w; }
        g.rr(60, 130, W - 120, 10, 3, P.wood);
        // flat decor: notebook and pen (the bug runs over them)
        g.rr(560, 360, 130, 96, 6, "#f7f3e8", P.line, 2);
        for (let i = 1; i < 6; i++) g.line(572, 360 + i * 15, 678, 360 + i * 15, "rgba(47,109,181,0.25)", 1);
        g.line(130, 440, 222, 412, P.navy, 6); g.line(222, 412, 236, 408, P.metal, 4);
        g.rr(AREA.x0 - 10, AREA.y0 - 10, AREA.x1 - AREA.x0 + 20, AREA.y1 - AREA.y0 + 20, 12, null, "rgba(19,154,154,0.12)", 2);
        if (glass && glass.down) { drawBug(g); drawGlass(g, glass.x, glass.y, 0, 1); }
        else {
          drawBug(g);
          if (glass) { const f = glass.t / FALL_T; drawGlass(g, glass.x, glass.y, 1 - f * f, 1); }
          else if (ghost) {
            g.ctx.save(); g.ctx.setLineDash([6, 6]);
            g.rr(ghost.x - BOX / 2, ghost.y - BOX / 2, BOX, BOX, 6, "rgba(170,215,240,0.12)", "rgba(60,130,175,0.6)", 2);
            g.ctx.restore();
          }
        }
        if (!glass) g.chip("Click once: the glass drops there a moment later", W / 2, 515, P.navy);
      }
    };
  }
});
