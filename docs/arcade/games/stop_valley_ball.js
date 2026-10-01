/* Stop Valley Ball — trajectory prediction.
 * Top-down view with height. A red ball is released from one side of a raised valley (half-pipe) ramp,
 * swings across the valley while rolling down, and flies off the ramp's edge, dropping toward the table
 * (its shadow shows where it is over the table; the ball is drawn lifted by its height).
 * Click to send the ping-pong bat there: its round head is held mid-air and glides at finite speed, so
 * late clicks arrive too late. The red ball must hit the bat head before it reaches the table.
 * You can re-click as often as you like.
 * var1: a tall side rail stands beside the ramp; the ball rebounds off it in mid-air.
 * var2: a black distractor ball also comes down the valley. It may hit the bat harmlessly; only red counts.
 */
Arcade.register({
  id: "stop_valley_ball",
  maxTime: 10,
  levels: {
    base: "Straight exit from the valley, no distractor.",
    var1: "The red ball rebounds off a side rail in mid-air.",
    var2: "A black distractor ball comes down too. Only the red ball counts.",
    "var1+2": "Mid-air rail rebound and a black distractor."
  },
  controls: [["Click", "move the bat head there (glides; re-click any time)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const K = 0.55;                                            // screen y offset per unit of height
    const TOP_Y = 74, EXIT_Y = 196, ACC = 200, EXIT_Z = 100, GZ = 470, BALL_R = 9, RW = 120;
    const BAT_Z = 38, BAT_R = 27, BAT_SPEED = 300;
    const EXIT_G = EXIT_Y + EXIT_Z * K;                        // ground y under the ramp edge
    const cx = rand(320, 490);
    let rail = null;                                           // var1: placed per episode on the side the ball heads to
    const bat = { x: W / 2, y: 488, tx: W / 2, ty: 488 };      // ground position under the bat head
    const start = { x: bat.x, y: bat.y };

    function makeBall(color, side, delay) {
      const amp = api.var1 && color === P.red ? rand(32, 48) : rand(12, 40);
      return { color, delay, A: side * amp, w: (2 * Math.PI) / rand(0.95, 1.45), tau: 0, state: "wait",
        x: cx + side * amp, y: TOP_Y + 8, z: 0, vx: 0, vy: 0, vz: 0, hitRail: false, railZ: 0, cross: null };
    }
    // One step for a ball; `useBat` false during the fairness pre-check.
    function stepBall(b, dt, useBat) {
      if (b.state === "wait") { b.delay -= dt; if (b.delay <= 0) b.state = "ramp"; return; }
      if (b.state === "ramp") {
        b.tau += dt;
        b.x = cx + b.A * Math.cos(b.w * b.tau); b.y = TOP_Y + 8 + 0.5 * ACC * b.tau * b.tau;   // screen y on the ramp
        if (b.y >= EXIT_Y) {
          b.state = "air"; b.y = EXIT_G; b.z = EXIT_Z;
          b.vx = -b.A * b.w * Math.sin(b.w * b.tau); b.vy = ACC * b.tau; b.vz = 0;
        }
        return;
      }
      if (b.state === "air") {
        const px = b.x;
        b.x += b.vx * dt; b.y += b.vy * dt; b.vz -= GZ * dt; b.z += b.vz * dt;
        if (rail && Math.sign(b.x - rail.x) !== Math.sign(px - rail.x) && b.y > rail.y0 && b.y < rail.y1 && b.z < rail.h) {
          b.x = rail.x - Math.sign(b.vx); b.vx = -b.vx * 0.8; b.hitRail = true; b.railZ = b.z; b.bump = true;
        }
        if (!b.cross && b.z <= BAT_Z + BALL_R) b.cross = { x: b.x, y: b.y };   // where a bat head would meet it
        if (useBat && b.vz < 0 && b.z <= BAT_Z + BALL_R + 4 && b.z >= BAT_Z - 6 && Math.hypot(b.x - bat.x, b.y - bat.y) < BAT_R + BALL_R * 0.6) {
          b.z = BAT_Z + BALL_R; b.vz = 260; b.vx *= 0.4; b.vy = -Math.abs(b.vy) * 0.5; b.batHit = true; b.bump = true;
        }
        if (b.z <= 0) { b.z = 0; b.state = "landed"; b.vz = 120; b.vx *= 0.5; b.vy *= 0.5; b.bump = true; }
        return;
      }
      if (b.state === "landed") {                               // small bounce, then rolls out
        b.x += b.vx * dt; b.y += b.vy * dt; b.vx *= 1 - 1.6 * dt; b.vy *= 1 - 1.6 * dt;
        if (b.z > 0 || b.vz > 0) { b.vz -= GZ * dt; b.z = Math.max(0, b.z + b.vz * dt); }
      }
    }
    function preview(b) {
      const c = Object.assign({}, b);
      for (let i = 0; i < 2400 && c.state !== "landed"; i++) stepBall(c, 1 / 120, false);
      return c;
    }

    // Fair episode: lands on the table, not right where the bat starts, and (var1) hits the rail high up.
    let red, black = null, tries = 0;
    for (;;) {
      tries++;
      red = makeBall(P.red, pick([-1, 1]), rand(0.6, 1.2));
      if (api.var1) {
        // exit offset/velocity across the valley decide which side the rail goes and how far out
        const T = Math.sqrt(2 * (EXIT_Y - TOP_Y - 8) / ACC), u = red.A * Math.cos(red.w * T), vx = -red.A * red.w * Math.sin(red.w * T);
        const s = Math.sign(vx) || 1;
        rail = { x: cx + s * Math.max(RW / 2 + 12, u * s + rand(30, 60)), y0: EXIT_G - 6, y1: EXIT_G + 200, h: 130 };
        if (Math.abs(vx) < 110) continue;
      }
      const e = preview(red);
      let ok = e.x > 70 && e.x < W - 70 && e.y < 470 && e.cross && Math.hypot(e.cross.x - start.x, e.cross.y - start.y) > 90;
      if (ok && api.var1) ok = e.hitRail && e.railZ > BAT_Z + 25;
      if (ok && !api.var1) ok = Math.abs(e.x - cx) < 200;
      if (ok || tries > 3000) break;
    }
    if (api.var2) {
      black = makeBall(P.black, pick([-1, 1]), red.delay + pick([-1, 1]) * rand(0.5, 0.9));
      if (black.delay < 0.25) black.delay += 1.5;
    }
    const balls = black ? [red, black] : [red];
    let clicked = false;

    return {
      _dbg: () => ({ red, black, tries, cx, rail, bat }),      // inspected by the test harness only
      pointerDown(x, y) {
        // the click marks where the head should appear; the bat's ground point is below it
        bat.tx = Math.max(50, Math.min(W - 50, x)); bat.ty = Math.max(EXIT_G + 10, Math.min(505, y + BAT_Z * K));
        clicked = true; api.sfx.click();
      },
      // Fixed 1/120 s physics steps, so play matches the fairness pre-check whatever the frame rate.
      _acc: 0,
      update(dt) {
        this._acc += dt;
        while (this._acc >= 1 / 120 - 1e-9 && !api.ended) { this._acc -= 1 / 120; this.tick(1 / 120); }
      },
      tick(dt) {
        const dx = bat.tx - bat.x, dy = bat.ty - bat.y, d = Math.hypot(dx, dy), st = BAT_SPEED * dt;
        if (d <= st) { bat.x = bat.tx; bat.y = bat.ty; } else { bat.x += dx / d * st; bat.y += dy / d * st; }
        for (const b of balls) { stepBall(b, dt, true); if (b.bump) { b.bump = false; api.sfx.tick(); } }
        if (red.batHit) return api.succeed("The red ball hit the bat");
        if (red.state === "landed") api.fail("The red ball hit the table");
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // raised valley ramp: long shadow on the table, two legs at the edge, then the half-pipe
        g.poly([[cx - RW / 2, TOP_Y + 70], [cx + RW / 2, TOP_Y + 70], [cx + RW / 2 + 4, EXIT_G], [cx - RW / 2 - 4, EXIT_G]], "rgba(15,34,51,0.08)");
        for (const sx of [-1, 1]) g.rr(cx + sx * (RW / 2 - 8) - 4, EXIT_Y, 8, EXIT_G - EXIT_Y, 2, P.metalDark);
        const gr = c.createLinearGradient(cx - RW / 2, 0, cx + RW / 2, 0);
        gr.addColorStop(0, "#efe2cf"); gr.addColorStop(0.5, "#c7a77e"); gr.addColorStop(1, "#efe2cf");
        g.rr(cx - RW / 2, TOP_Y, RW, EXIT_Y - TOP_Y, 8, gr, P.woodDark, 2);
        for (let i = 1; i < 4; i++) g.line(cx - RW / 2 + 8, TOP_Y + i * 30, cx + RW / 2 - 8, TOP_Y + i * 30, "rgba(0,0,0,0.05)", 1);
        g.rr(cx - RW / 2 - 6, TOP_Y, 8, EXIT_Y - TOP_Y, 3, P.woodDark);
        g.rr(cx + RW / 2 - 2, TOP_Y, 8, EXIT_Y - TOP_Y, 3, P.woodDark);
        g.line(cx - RW / 2, EXIT_Y, cx + RW / 2, EXIT_Y, P.woodDark, 4);
        // tall rail: its foot on the table and its top face lifted by its height
        if (rail) {
          const top = rail.h * K;
          g.rr(rail.x - 5, rail.y0 - top, 10, rail.y1 - rail.y0 + top, 3, "rgba(154,166,178,0.55)");
          g.rr(rail.x - 5, rail.y0 - top, 10, rail.y1 - rail.y0, 3, P.metal, P.metalDark);
        }
        // shadows first (ground), then the bat, then balls by height
        for (const b of balls) if (b.state === "air" || b.state === "landed") g.shadow(b.x, b.y, BALL_R, b.z);
        g.shadow(bat.x, bat.y, BAT_R * 0.8, BAT_Z);
        if (clicked && Math.hypot(bat.tx - bat.x, bat.ty - bat.y) > 2) g.circle(bat.tx, bat.ty - BAT_Z * K, BAT_R, null, "rgba(11,42,69,0.35)", 2);
        const hx = bat.x, hy = bat.y - BAT_Z * K;
        g.line(hx + 10, hy + 18, hx + 26, hy + 60, P.wood, 10);  // handle toward the robot side
        g.line(hx + 10, hy + 18, hx + 26, hy + 60, P.woodDark, 2);
        g.circle(hx, hy, BAT_R, P.red, "#9d3026", 3);
        g.circle(hx - 7, hy - 8, BAT_R * 0.35, "rgba(255,255,255,0.18)");
        const drawn = balls.slice().sort((a, b) => a.z - b.z);
        for (const b of drawn) {
          if (b.state === "wait" || b.state === "ramp") { g.ball(b.x, b.y, BALL_R, b.color); continue; }
          g.ball(b.x, b.y - b.z * K, BALL_R * (1 + b.z / 300), b.color);
        }
        if (!clicked) g.chip("Click where the bat head should meet the red ball", W / 2, 506, P.navy);
      }
    };
  }
});
