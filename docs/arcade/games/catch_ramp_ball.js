/* Catch Ramp Ball — trajectory prediction.
 * A red ball rolls down a tilted ramp, leaves the exit and flies onto the table.
 * Place the cup once, where the ball will land.
 * var1: the ball leaves at an angle and rebounds off a side wall mid-air.
 * var2: a blue distractor ball rolls down another lane; it must not land in the cup.
 */
Arcade.register({
  id: "catch_ramp_ball",
  maxTime: 14,
  levels: {
    base: "Straight ramp exit, no distractor.",
    var1: "The ball rebounds off a side wall before landing.",
    var2: "A blue distractor ball rolls down too. Catch only the red one.",
    "var1+2": "Wall rebound and a blue distractor."
  },
  controls: [["Click", "place the cup (only once)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const RAMP_LEN = 170, RAMP_W = 120, ACC = 300, EXIT_Z = 70, GZ = 900, BALL_R = 9, CUP_R = 30;
    const wallSide = pick([-1, 1]);
    // Ramp is anchored at its exit point and points "down" the table, tilted by `ang`.
    const ang = api.var1 ? wallSide * rand(0.32, 0.45) : rand(-0.22, 0.22);
    const exit = { x: rand(300, 510), y: 250 };
    const dir = { x: Math.sin(ang), y: Math.cos(ang) };
    const nrm = { x: dir.y, y: -dir.x };                  // across the ramp
    const top = { x: exit.x - dir.x * RAMP_LEN, y: exit.y - dir.y * RAMP_LEN };
    const wall = api.var1 ? { x: exit.x + wallSide * rand(95, 120), y0: exit.y + 10, y1: exit.y + 230 } : null;

    function makeBall(color, lane, delay) {
      return { color, lane, delay, s: 14, v: 0, state: "wait", x: 0, y: 0, z: EXIT_Z, vx: 0, vy: 0, vz: 0, inCup: false, fade: 1 };
    }
    const lanes = api.var2 ? (api.chance(0.5) ? [-30, 30] : [30, -30]) : [rand(-20, 20)];
    const red = makeBall(P.red, lanes[0], rand(0.7, 1.4));
    const balls = [red];
    if (api.var2) balls.push(makeBall(P.blue, lanes[1], red.delay + pick([-1, 1]) * rand(0.45, 0.8)));

    let cup = null, ghost = null;

    function rampPos(b) {
      return { x: top.x + dir.x * b.s + nrm.x * b.lane, y: top.y + dir.y * b.s + nrm.y * b.lane };
    }
    function land(b) {
      b.state = "landed"; b.z = 0;
      if (cup && cup.down && Math.hypot(b.x - cup.x, b.y - cup.y) < CUP_R - BALL_R + 3) {
        b.inCup = true; b.x = cup.x + (b.x - cup.x) * 0.5; b.y = cup.y + (b.y - cup.y) * 0.5;
      } else {
        b.vx *= 0.55; b.vy *= 0.55; b.vz = 160;               // small bounce, then rolls away
      }
      api.sfx.tick();
    }

    return {
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (cup || y < exit.y + 25) return;                   // cup goes on the table beyond the ramp
        cup = { x, y, t: 0, down: false };
        api.sfx.click();
      },
      update(dt) {
        if (cup && !cup.down) { cup.t += dt; if (cup.t > 0.25) cup.down = true; }
        for (const b of balls) {
          if (b.state === "wait") { b.delay -= dt; if (b.delay <= 0) b.state = "ramp"; }
          if (b.state === "ramp") {
            b.v += ACC * dt; b.s += b.v * dt;
            const p = rampPos(b); b.x = p.x; b.y = p.y;
            if (b.s >= RAMP_LEN) { b.state = "air"; b.vx = dir.x * b.v; b.vy = dir.y * b.v; b.vz = 0; b.z = EXIT_Z; }
          } else if (b.state === "air") {
            b.x += b.vx * dt; b.y += b.vy * dt; b.vz -= GZ * dt; b.z += b.vz * dt;
            if (wall && Math.sign(b.x - wall.x) !== Math.sign(b.x - b.vx * dt - wall.x) && b.y > wall.y0 && b.y < wall.y1) {
              b.x = wall.x - Math.sign(b.vx) * 1; b.vx = -b.vx * 0.8; api.sfx.tick();
            }
            if (b.z <= 0) land(b);
          } else if (b.state === "landed" && !b.inCup) {
            b.x += b.vx * dt; b.y += b.vy * dt;
            if (b.z > 0 || b.vz > 0) { b.vz -= GZ * dt; b.z = Math.max(0, b.z + b.vz * dt); }
            b.vx *= 1 - 1.5 * dt; b.vy *= 1 - 1.5 * dt;
          }
        }
        // Decide once every ball has landed.
        if (red.state === "landed" && !api.ended) {
          if (!red.inCup) return api.fail("The red ball missed the cup");
          if (balls.every(b => b.state === "landed")) {
            if (balls.some(b => b !== red && b.inCup)) return api.fail("The blue distractor landed in the cup");
            api.succeed("Caught the red ball");
          }
        }
      },
      draw(g) {
        g.scene();
        // landing zone hint
        g.rr(40, exit.y + 25, W - 80, 230, 12, "rgba(19,154,154,0.05)");
        // ramp: wooden board with rails, drawn as a rotated rectangle
        const c = g.ctx;
        c.save(); c.translate(exit.x, exit.y); c.rotate(-ang);
        g.rr(-RAMP_W / 2 - 3, -RAMP_LEN - 6, RAMP_W + 6, RAMP_LEN + 12, 6, P.shadow);
        g.rr(-RAMP_W / 2, -RAMP_LEN, RAMP_W, RAMP_LEN, 5, P.wood, P.woodDark, 2);
        for (let i = 1; i < 6; i++) g.line(-RAMP_W / 2 + 6, -RAMP_LEN + i * 28, RAMP_W / 2 - 6, -RAMP_LEN + i * 28, "rgba(0,0,0,0.06)", 1);
        g.rr(-RAMP_W / 2 - 6, -RAMP_LEN, 7, RAMP_LEN, 3, P.woodDark);
        g.rr(RAMP_W / 2 - 1, -RAMP_LEN, 7, RAMP_LEN, 3, P.woodDark);
        g.line(-RAMP_W / 2, 0, RAMP_W / 2, 0, P.red, 3);       // exit edge
        c.restore();
        if (wall) { g.rr(wall.x - 7, wall.y0, 14, wall.y1 - wall.y0, 4, P.metal, P.metalDark); }
        // cup (top-down): outer rim, inner well
        const drawCup = (x, y, alpha, lift) => {
          c.save(); c.globalAlpha = alpha;
          g.shadow(x, y + 4, CUP_R, lift);
          g.circle(x, y - lift * 0.5, CUP_R, P.blue, "#1f4f8a", 3);
          g.circle(x, y - lift * 0.5, CUP_R - 7, "#24558f");
          c.restore();
        };
        if (cup) drawCup(cup.x, cup.y, 1, cup.down ? 0 : (1 - cup.t / 0.25) * 60);
        else if (ghost && ghost.y > exit.y + 25) drawCup(ghost.x, ghost.y, 0.3, 0);
        // balls: shadow on the table, ball lifted by height
        for (const b of balls) {
          if (b.state === "wait") { const p = rampPos(b); g.ball(p.x, p.y, BALL_R, b.color); continue; }
          if (b.inCup) { g.ball(b.x, b.y, BALL_R * 0.9, b.color); continue; }
          if (b.state === "ramp") { g.ball(b.x, b.y, BALL_R, b.color); continue; }
          g.shadow(b.x, b.y, BALL_R, b.z);
          g.ball(b.x, b.y - b.z * 0.55, BALL_R * (1 + b.z / 260), b.color);
        }
        if (!cup) g.chip("Click once to place the cup", W / 2, 520, P.navy);
      }
    };
  }
});
