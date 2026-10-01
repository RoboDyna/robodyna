/* Save Goal — trajectory prediction.
 * Top-down mini soccer field on the table. After a short wait the ball is kicked from mid-field toward the
 * goal at an angle and rolls (with a little friction). Click once inside the green zone to place the
 * goalkeeper bar; placing is only allowed until the ball crosses the red line, so the save must be predicted.
 * The episode succeeds if the ball hits the keeper before crossing the goal line.
 * var1: 1–3 field players stand in the way; the ball glances off them, changing direction.
 * var2: an opaque mid-field cover hides the ball for part of its run.
 */
Arcade.register({
  id: "save_goal",
  maxTime: 10,
  levels: {
    base: "Direct shot at the goal.",
    var1: "The shot deflects off field players on the way.",
    var2: "A mid-field cover hides the ball for part of its run.",
    "var1+2": "Deflecting field players and the mid-field cover."
  },
  controls: [["Click", "place the goalkeeper in the green zone (once, before the ball crosses the red line)"]],
  create(api) {
    const { P, W, rand, randInt } = api;
    const FX0 = 64, FX1 = 746, FY0 = 82, FY1 = 488, GX = W / 2, GW = 92, BALL_R = 8, PL_R = 15;
    const RED_Y = FY1 - 122, ZONE = { x0: GX - 135, x1: GX + 135, y0: FY1 - 76, y1: FY1 - 8 }, KW = 66, KH = 12;
    const COVER = api.var2 ? { y0: 170, y1: 262 } : null;

    // Advance the ball by dt, bouncing off field players. Shared by the game and the fairness pre-check.
    function stepBall(b, dt, players) {
      if (b.wait > 0) { b.wait -= dt; return; }
      const f = 1 - 0.06 * dt;
      b.vx *= f; b.vy *= f; b.x += b.vx * dt; b.y += b.vy * dt;
      for (const p of players) {
        const dx = b.x - p.x, dy = b.y - p.y, d = Math.hypot(dx, dy);
        if (d < PL_R + BALL_R) {
          const nx = dx / d, ny = dy / d, vn = b.vx * nx + b.vy * ny;
          if (vn < 0) { b.vx -= 1.8 * vn * nx; b.vy -= 1.8 * vn * ny; b.lastHit = b.y; b.hits++; b.bump = true; }
          b.x = p.x + nx * (PL_R + BALL_R + 0.5); b.y = p.y + ny * (PL_R + BALL_R + 0.5);
        }
      }
    }
    function newBall() {
      const x = rand(190, 620), y = rand(118, 148), tx = GX + rand(-GW + 20, GW - 20), sp = rand(115, 155);
      const d = Math.hypot(tx - x, FY1 - y);
      return { x, y, vx: (tx - x) / d * sp, vy: (FY1 - y) / d * sp, wait: rand(0.6, 1.1), hits: 0, lastHit: -1 };
    }
    // Pre-roll until the ball reaches the goal line (or leaves the field).
    function preview(b, players) {
      const c = Object.assign({}, b);
      for (let i = 0; i < 1200; i++) {
        stepBall(c, 1 / 120, players);
        if (c.y >= FY1 || c.x < FX0 || c.x > FX1 || c.y < FY0) break;
      }
      return c;
    }

    // Fair episode: the shot (after any deflections) goes cleanly into the goal mouth, and in var1 it hits at
    // least one player, with the last deflection well before the red line so the save stays predictable.
    let ball, players = [], tries = 0;
    for (;;) {
      tries++;
      ball = newBall(); players = [];
      if (api.var1) {
        const n = randInt(1, 3);
        // the first player stands on the shot line, slightly off-centre, so the ball glances off it
        const py = rand(250, 315), t = (py - ball.y) / ball.vy, sx = ball.x + ball.vx * t, off = rand(6, 16) * (api.chance(0.5) ? 1 : -1);
        players.push({ x: sx + off, y: py });
        for (let i = 1; i < n; i++) players.push({ x: rand(FX0 + 40, FX1 - 40), y: rand(210, 320) });
      }
      const e = preview(ball, players);
      let ok = e.y >= FY1 && Math.abs(e.x - GX) < GW - 14;
      if (ok && api.var1) ok = e.hits >= 1 && e.lastHit < RED_Y - 40 && players.every((p, i) => players.slice(0, i).every(q => Math.hypot(p.x - q.x, p.y - q.y) > 50));
      if (ok || tries > 3000) break;
    }
    const kick = { x: ball.x - ball.vx / Math.hypot(ball.vx, ball.vy) * 22, y: ball.y - ball.vy / Math.hypot(ball.vx, ball.vy) * 22 };

    let keeper = null, ghost = null, late = false;
    const inZone = (x, y) => x >= ZONE.x0 && x <= ZONE.x1 && y >= ZONE.y0 && y <= ZONE.y1;

    return {
      _dbg: () => ({ ball, players, tries, keeper }),            // inspected by the test harness only
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (keeper) return;
        if (ball.y > RED_Y) { late = true; api.sfx.tick(); return; }
        if (!inZone(x, y)) { api.sfx.tick(); return; }
        keeper = { x, y, t: 0 };
        api.sfx.click();
      },
      // Fixed 1/120 s physics steps, so play matches the fairness pre-check whatever the frame rate.
      _acc: 0,
      update(dt) {
        this._acc += dt;
        while (this._acc >= 1 / 120 - 1e-9 && !api.ended) { this._acc -= 1 / 120; this.tick(1 / 120); }
      },
      tick(dt) {
        if (keeper) keeper.t += dt;
        const wasWaiting = ball.wait > 0;
        stepBall(ball, dt, players);
        if (wasWaiting && ball.wait <= 0) api.sfx.whoosh();
        if (ball.bump) { ball.bump = false; api.sfx.tick(); }
        if (keeper) {                                             // ball vs keeper bar (circle vs rectangle)
          const qx = Math.max(keeper.x - KW / 2, Math.min(ball.x, keeper.x + KW / 2));
          const qy = Math.max(keeper.y - KH / 2, Math.min(ball.y, keeper.y + KH / 2));
          if (Math.hypot(ball.x - qx, ball.y - qy) < BALL_R) {
            ball.vy = -Math.abs(ball.vy) * 0.6; ball.vx *= 0.6; return api.succeed("Saved by the goalkeeper");
          }
        }
        if (ball.y >= FY1 + BALL_R) {
          if (Math.abs(ball.x - GX) < GW) return api.fail(keeper ? "Goal! The keeper was not in the ball's path" : "Goal! No keeper was placed");
          return api.succeed("The shot went wide");
        }
        if (ball.x < FX0 - 20 || ball.x > FX1 + 20) api.succeed("The shot went wide");
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // pitch: half field with stripes, halfway line on top, goal at the bottom
        g.rr(FX0 - 6, FY0 - 6, FX1 - FX0 + 12, FY1 - FY0 + 34, 10, "#cfe6d3");
        for (let i = 0; i < 7; i++) c.fillStyle = i % 2 ? "#d7ecda" : "#cfe6d3", c.fillRect(FX0, FY0 + i * (FY1 - FY0) / 7, FX1 - FX0, (FY1 - FY0) / 7);
        c.strokeStyle = "rgba(255,255,255,0.9)"; c.lineWidth = 3;
        c.strokeRect(FX0, FY0, FX1 - FX0, FY1 - FY0);
        c.beginPath(); c.arc(GX, FY0, 60, 0, Math.PI); c.stroke();
        c.strokeRect(GX - 150, FY1 - 92, 300, 92);
        // green keeper zone and the red placement line
        g.rr(ZONE.x0, ZONE.y0, ZONE.x1 - ZONE.x0, ZONE.y1 - ZONE.y0, 6, "rgba(63,165,91,0.28)", P.green, 2);
        g.line(FX0, RED_Y, FX1, RED_Y, P.red, 3);
        // goal: net behind the line, posts
        g.rr(GX - GW, FY1, GW * 2, 24, 3, "#eef2f6", P.metalDark, 2);
        for (let x = GX - GW + 10; x < GX + GW; x += 10) g.line(x, FY1 + 2, x, FY1 + 22, "rgba(111,124,136,0.35)", 1);
        g.line(GX - GW, FY1 + 12, GX + GW, FY1 + 12, "rgba(111,124,136,0.35)", 1);
        g.line(GX - GW, FY1, GX + GW, FY1, "#fff", 4);
        g.circle(GX - GW, FY1, 5, P.white, P.metalDark, 2); g.circle(GX + GW, FY1, 5, P.white, P.metalDark, 2);
        // striker spot and field players
        g.shadow(kick.x, kick.y + 3, 12, 0); g.circle(kick.x, kick.y, 12, P.orange, "#b7650f", 2);
        for (const p of players) { g.shadow(p.x, p.y + 4, PL_R, 0); g.circle(p.x, p.y, PL_R, P.blue, "#1f4f8a", 2); g.circle(p.x, p.y, 5, "#fff"); }
        // the ball (hidden while under the cover)
        const hidden = COVER && ball.y > COVER.y0 && ball.y < COVER.y1;
        if (!hidden) { g.shadow(ball.x + 1, ball.y + 3, BALL_R, 0); g.ball(ball.x, ball.y, BALL_R, "#f4f6f8"); g.circle(ball.x, ball.y, BALL_R, null, P.ink, 1.2); g.circle(ball.x + 1, ball.y + 1, 2.5, P.ink); }
        if (COVER) {
          g.rr(FX0 - 14, COVER.y0 + 6, FX1 - FX0 + 28, COVER.y1 - COVER.y0, 8, P.shadow);
          g.rr(FX0 - 14, COVER.y0, FX1 - FX0 + 28, COVER.y1 - COVER.y0, 8, "#c5ced8", P.metalDark, 2);
          for (let x = FX0 + 20; x < FX1; x += 46) g.line(x, COVER.y0 + 8, x + 18, COVER.y1 - 8, "rgba(255,255,255,0.35)", 3);
        }
        // keeper: gloves bar + body
        const drawKeeper = (x, y, alpha, bad) => {
          c.save(); c.globalAlpha = alpha;
          g.rr(x - KW / 2 + 2, y - KH / 2 + 4, KW, KH, 6, P.shadow);
          g.rr(x - KW / 2, y - KH / 2, KW, KH, 6, bad ? P.red : P.yellow, bad ? "#9d3026" : "#b88c12", 2);
          g.circle(x, y + 1, 9, bad ? "#f3b6ad" : P.navy);
          c.restore();
        };
        if (keeper) drawKeeper(keeper.x, keeper.y, Math.min(1, 0.4 + keeper.t * 4), false);
        else if (ghost && ghost.y > RED_Y && ball.y <= RED_Y) drawKeeper(ghost.x, ghost.y, 0.4, !inZone(ghost.x, ghost.y));
        if (!keeper) g.chip(late || ball.y > RED_Y ? "Too late: the ball crossed the red line" : "Click once in the green zone to place the keeper", W / 2, 70, late || ball.y > RED_Y ? P.red : P.navy);
      }
    };
  }
});
