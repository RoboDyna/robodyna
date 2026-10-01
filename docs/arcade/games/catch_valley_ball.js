/* Catch Valley Ball — trajectory prediction.
 * Top-down view. A red ball is released from one side of a curved valley (half-pipe) ramp, so it swings
 * across the valley while it rolls down. Wherever it is in its swing when it reaches the exit sets the angle
 * at which it rolls out onto the table; it then keeps rolling (with friction) past a red line and stops.
 * Click once to push the blue box (open side facing the ramp) to a spot behind the red line; the ball is
 * caught if it rolls into the open side. Too far and the ball stops short; off the path and it rolls by.
 * var1: the ball leaves toward a metal side rail and rebounds off it.
 * var2: a black distractor ball comes down the valley at another time / swing; it must not end in the box.
 */
Arcade.register({
  id: "catch_valley_ball",
  maxTime: 12,
  levels: {
    base: "Straight exit from the valley, no distractor.",
    var1: "The red ball rebounds off a side rail after leaving the valley.",
    var2: "A black distractor ball rolls down too. Catch only the red one.",
    "var1+2": "Side-rail rebound and a black distractor."
  },
  controls: [["Click", "push the box there (only once, behind the red line)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const TOP_Y = 76, EXIT_Y = 214, L = EXIT_Y - TOP_Y, RW = 124, ACC = 210, BALL_R = 9;
    const LINE_Y = 300, BW = 86, BD = 56, WT = 7, BOX_SPEED = 1500;
    const TBL = { x0: 34, x1: 776, y1: 514 };
    const cx = rand(330, 480);
    const railSide = pick([-1, 1]);
    const rail = api.var1 ? { x: cx + railSide * rand(74, 100), y0: EXIT_Y + 14, y1: 470 } : null;

    // A ball's whole state; `side` is where it is released across the valley, `w` its swing rate.
    function makeBall(color, side, delay) {
      const amp = api.var1 && color === P.red ? rand(30, 46) : rand(14, 40);
      return {
        color, delay, A: side * amp, w: (2 * Math.PI) / rand(0.95, 1.45), tau: 0, state: "wait",
        x: cx + side * amp, y: TOP_Y + 8, vx: 0, vy: 0, dec: 0, dStop: rand(200, 290), hitRail: false
      };
    }
    // Advance one ball by dt; `box` (or null) is the placed box. Shared by the game and the pre-check.
    function stepBall(b, dt, box) {
      if (b.state === "wait") { b.delay -= dt; if (b.delay <= 0) b.state = "ramp"; return; }
      if (b.state === "ramp") {
        b.tau += dt;
        const s = 0.5 * ACC * b.tau * b.tau;
        b.x = cx + b.A * Math.cos(b.w * b.tau); b.y = TOP_Y + 8 + s;
        if (b.y >= EXIT_Y) {
          b.state = "roll"; b.y = EXIT_Y;
          b.vx = -b.A * b.w * Math.sin(b.w * b.tau); b.vy = ACC * b.tau;
          b.dec = (b.vx * b.vx + b.vy * b.vy) / (2 * b.dStop);  // rolling friction from the planned stop distance
        }
        return;
      }
      if (b.state !== "roll" && b.state !== "inbox") return;
      const sp = Math.hypot(b.vx, b.vy), dec = b.state === "inbox" ? b.dec * 6 + 300 : b.dec;
      const ns = Math.max(0, sp - dec * dt);
      if (ns <= 0.5) { b.vx = b.vy = 0; b.state = b.state === "inbox" ? "caught" : "stopped"; return; }
      b.vx *= ns / sp; b.vy *= ns / sp;
      const px = b.x, py = b.y;
      b.x += b.vx * dt; b.y += b.vy * dt;
      if (rail && !b.hitRail && Math.sign(b.x - rail.x) !== Math.sign(px - rail.x) && b.y > rail.y0 && b.y < rail.y1) {
        b.x = rail.x - Math.sign(b.vx) * 1; b.vx = -b.vx * 0.78; b.vy *= 0.92; b.hitRail = true; b.bump = true;
      }
      if (box && box.down) boxContact(b, px, py, box);
      if (b.x < TBL.x0 || b.x > TBL.x1 || b.y > TBL.y1) b.state = "off";
    }
    // Box walls: left, right and back (bottom); the top side is open toward the ramp.
    function boxContact(b, px, py, box) {
      const l = box.x - BW / 2, r = box.x + BW / 2, t = box.y - BD / 2, bt = box.y + BD / 2;
      if (b.state === "inbox") {                              // keep it inside, bouncing softly off the walls
        if (b.x < l + WT + BALL_R) { b.x = l + WT + BALL_R; b.vx = Math.abs(b.vx) * 0.4; }
        if (b.x > r - WT - BALL_R) { b.x = r - WT - BALL_R; b.vx = -Math.abs(b.vx) * 0.4; }
        if (b.y > bt - WT - BALL_R) { b.y = bt - WT - BALL_R; b.vy = -Math.abs(b.vy) * 0.3; }
        if (b.y < t + BALL_R) { b.y = t + BALL_R; b.vy = Math.abs(b.vy) * 0.3; }
        return;
      }
      if (py < t && b.y >= t && b.x > l - BALL_R && b.x < r + BALL_R) {   // reaches the open side
        if (b.x > l + WT + BALL_R * 0.4 && b.x < r - WT - BALL_R * 0.4) { b.state = "inbox"; b.bump = true; }
        else { b.y = t - 1; b.vy = -Math.abs(b.vy) * 0.4; b.bump = true; }  // clipped a wall end
        return;
      }
      if (b.y > t && b.y < bt + BALL_R) {                      // outer side walls
        if (px <= l - BALL_R && b.x > l - BALL_R) { b.x = l - BALL_R - 1; b.vx = -Math.abs(b.vx) * 0.6; b.bump = true; }
        if (px >= r + BALL_R && b.x < r + BALL_R) { b.x = r + BALL_R + 1; b.vx = Math.abs(b.vx) * 0.6; b.bump = true; }
      }
    }
    // Pre-roll a ball without any box (fixed 1/120 s) to check the episode is fair.
    function preview(b) {
      const c = Object.assign({}, b), pts = [];
      for (let i = 0; i < 2400 && (c.state === "wait" || c.state === "ramp" || c.state === "roll"); i++) {
        stepBall(c, 1 / 120, null);
        if (c.state === "roll") pts.push([c.x, c.y]);
      }
      return { pts, end: c };
    }

    // Draw episodes until one is fair: red stops well behind the line, on the table, (var1) after the rail,
    // and (var2) some stretch of red's path is far from anywhere the black ball goes.
    let red, black = null, tries = 0;
    for (;;) {
      tries++;
      const side = api.var1 ? -railSide : pick([-1, 1]);       // var1: released opposite the rail so it swings over
      red = makeBall(P.red, side, rand(0.6, 1.2));
      const pr = preview(red), e = pr.end;
      let ok = e.state === "stopped" && e.y > LINE_Y + 75 && e.y < TBL.y1 - 30 && e.x > 70 && e.x < W - 70;
      if (ok && api.var1) ok = e.hitRail && pr.pts.some(p => p[1] < LINE_Y + 50 && Math.abs(p[0] - rail.x) < 3);
      if (ok && !api.var1) ok = Math.abs(e.x - cx) < 210;
      if (ok && api.var2) {
        ok = false;
        for (let k = 0; k < 25 && !ok; k++) {                // several distractor draws for this red ball
          black = makeBall(P.black, pick([-1, 1]), red.delay + pick([-1, 1]) * rand(0.75, 1.1));
          if (black.delay < 0.3) black.delay += 1.9;
          const pb = preview(black);
          ok = pb.end.state === "stopped" && pr.pts.some(p => p[1] > LINE_Y + 40 && p[1] < e.y - 5 &&
            pb.pts.every(q => Math.hypot(q[0] - p[0], q[1] - p[1]) > 85));
        }
      }
      if (ok || tries > 3000) break;
    }
    const balls = black ? [red, black] : [red];

    let box = null, ghost = null;
    const park = { x: cx + (rail ? -railSide : pick([-1, 1])) * 150, y: 150 };   // box waits beside the ramp
    const validBox = (x, y) => y - BD / 2 > LINE_Y + 2 && y + BD / 2 < TBL.y1 && x - BW / 2 > TBL.x0 && x + BW / 2 < TBL.x1;

    return {
      _dbg: () => ({ red, black, tries, cx, rail }),        // inspected by the test harness only
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (box) return;
        if (!validBox(x, y)) { api.sfx.tick(); return; }
        box = { x: park.x, y: park.y, tx: x, ty: y, down: false };
        api.sfx.whoosh();
      },
      // Fixed 1/120 s physics steps, so play matches the fairness pre-check whatever the frame rate.
      _acc: 0,
      update(dt) {
        this._acc += dt;
        while (this._acc >= 1 / 120 - 1e-9 && !api.ended) { this._acc -= 1 / 120; this.tick(1 / 120); }
      },
      tick(dt) {
        if (box && !box.down) {                                 // the box slides over to the clicked spot
          const dx = box.tx - box.x, dy = box.ty - box.y, d = Math.hypot(dx, dy), st = BOX_SPEED * dt;
          if (d <= st) {
            box.x = box.tx; box.y = box.ty; box.down = true; api.sfx.click();
            for (const b of balls) {                            // pushed onto a ball that is already there
              if ((b.state === "roll" || b.state === "stopped") && Math.abs(b.x - box.x) < BW / 2 + BALL_R && Math.abs(b.y - box.y) < BD / 2 + BALL_R) {
                if (b === red) return api.fail("The box was pushed onto the red ball");
                b.y = box.y + BD / 2 + BALL_R + 2; b.vx = b.vy = 0; b.state = "stopped";
              }
            }
          } else { box.x += dx / d * st; box.y += dy / d * st; }
        }
        for (const b of balls) { stepBall(b, dt, box); if (b.bump) { b.bump = false; api.sfx.tick(); } }
        if (black && black.state === "inbox") return api.fail("The black distractor rolled into the box");
        if (red.state === "off") return api.fail("The red ball rolled off the table");
        if (red.state === "stopped") return api.fail(red.y < LINE_Y + 2 ? "The red ball stopped short" : "The red ball missed the box");
        if (red.state === "caught" && (!black || black.state === "stopped" || black.state === "off")) api.succeed("Caught the red ball");
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // area behind the red line where the box may go, and the line itself
        g.rr(TBL.x0 + 6, LINE_Y + 2, TBL.x1 - TBL.x0 - 12, TBL.y1 - LINE_Y - 6, 10, "rgba(47,109,181,0.05)");
        g.line(TBL.x0 + 4, LINE_Y, TBL.x1 - 4, LINE_Y, P.red, 3);
        // valley ramp: a half-pipe, darker along its trough, with raised wooden lips
        g.rr(cx - RW / 2 - 4, TOP_Y - 4, RW + 8, L + 10, 10, P.shadow);
        const gr = c.createLinearGradient(cx - RW / 2, 0, cx + RW / 2, 0);
        gr.addColorStop(0, "#efe2cf"); gr.addColorStop(0.5, "#c7a77e"); gr.addColorStop(1, "#efe2cf");
        g.rr(cx - RW / 2, TOP_Y, RW, L, 8, gr, P.woodDark, 2);
        for (let i = 1; i < 5; i++) g.line(cx - RW / 2 + 8, TOP_Y + i * L / 5, cx + RW / 2 - 8, TOP_Y + i * L / 5, "rgba(0,0,0,0.05)", 1);
        g.rr(cx - RW / 2 - 6, TOP_Y, 8, L, 3, P.woodDark);
        g.rr(cx + RW / 2 - 2, TOP_Y, 8, L, 3, P.woodDark);
        if (rail) { g.rr(rail.x - 6, rail.y0 + 3, 12, rail.y1 - rail.y0, 4, P.shadow); g.rr(rail.x - 6, rail.y0, 12, rail.y1 - rail.y0, 4, P.metal, P.metalDark); }
        // the open box: floor, then three walls (open side faces the ramp)
        const drawBox = (x, y, alpha, bad) => {
          c.save(); c.globalAlpha = alpha;
          const l = x - BW / 2, t = y - BD / 2, col = bad ? P.red : P.blue, dk = bad ? "#9d3026" : "#1f4f8a";
          g.rr(l + 2, t + 5, BW, BD, 5, P.shadow);
          g.rr(l, t, BW, BD, 5, bad ? "#f7d5d0" : "#d6e3f3");
          g.rr(l, t, WT, BD, 3, col, dk); g.rr(l + BW - WT, t, WT, BD, 3, col, dk);
          g.rr(l, t + BD - WT, BW, WT, 3, col, dk);
          c.restore();
        };
        if (box) drawBox(box.x, box.y, 1, false);
        else if (ghost && ghost.y > LINE_Y - 40) drawBox(ghost.x, ghost.y, 0.35, !validBox(ghost.x, ghost.y));
        if (!box) drawBox(park.x, park.y, 1, false);
        for (const b of balls) {
          if (b.state === "off") continue;
          g.shadow(b.x + 2, b.y + 3, BALL_R * 0.9, 0);
          g.ball(b.x, b.y, BALL_R, b.color);
        }
        if (!box) g.chip("Click once behind the red line to push the box there", W / 2, 500, P.navy);
      }
    };
  }
});
