/* Drop Ball Hole — periodic pattern.
 * Top-down shape sorter: a disc rotates around its center at a randomized speed and direction,
 * carrying a target hole (green ring) around periodically. Click on the disc to drop the ball from
 * above at that fixed world spot; it falls for ~0.55 s (shadow shrinks) before touching the disc,
 * so the drop has to lead the hole. ONE drop per episode.
 * Regular surface: a ball that lands near the hole slips relative to the turning disc (friction pulls
 * it up to the disc speed), so it can still roll into the hole if it lands just in front of it.
 * Sticky surface: the ball sticks where it lands and simply turns with the disc — only a direct hit works.
 * Success: the ball falls through the target hole into the container below.
 * base: regular surface, target hole only.   var1: sticky surface.
 * var2: regular surface + an identical dummy hole (no ring); the ball in the dummy fails.
 * var1+2: sticky surface + dummy hole.
 */
Arcade.register({
  id: "drop_ball_hole",
  maxTime: 12,
  levels: {
    base: "Regular surface: a near miss can still roll in. Target hole only.",
    var1: "Sticky surface: the ball sticks where it lands. Hit the hole directly.",
    var2: "Regular surface with a dummy hole. Only the ringed hole counts.",
    "var1+2": "Sticky surface and a dummy hole."
  },
  controls: [["Click", "drop the ball on the disc (only once)"]],
  create(api) {
    const { P, rand, pick } = api;
    const C = { x: 405, y: 300 }, R = 190, HOLE_R = 22, BALL_R = 12, FALL_T = 0.55, Z0 = 170, SLIP_K = 3.2;
    const sticky = api.var1;
    const w = pick([-1, 1]) * rand(0.6, 1.15);              // rad/s
    let th = rand(0, Math.PI * 2);                          // disc rotation angle
    const target = { r: rand(75, 140), a: rand(0, Math.PI * 2), dummy: false };
    const holes = [target];
    if (api.var2) holes.push({ r: rand(75, 140), a: target.a + pick([-1, 1]) * rand(1.7, 2.6), dummy: true });
    // flat printed decals that make the rotation easy to read (not holes)
    const decals = Array.from({ length: 5 }, (_, i) => ({ r: rand(40, 165), a: i * 1.257 + rand(-0.3, 0.3),
      shape: i % 3, color: [P.yellow, P.blue, P.orange, P.purple, P.teal][i] }))
      .filter(d => holes.every(h => Math.hypot(d.r * Math.cos(d.a) - h.r * Math.cos(h.a), d.r * Math.sin(d.a) - h.r * Math.sin(h.a)) > 55));

    let ball = null, ghost = null, rest = 0;
    const holePos = h => ({ x: C.x + h.r * Math.cos(h.a + th), y: C.y + h.r * Math.sin(h.a + th) });
    const surfV = (x, y) => ({ x: -w * (y - C.y), y: w * (x - C.x) });

    return {
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (ball || Math.hypot(x - C.x, y - C.y) > R - BALL_R) return;
        ball = { x, y, vx: 0, vy: 0, z: Z0, t: 0, state: "fall", sink: 0, hole: null };
        api.sfx.whoosh();
      },
      update(dt) {
        th += w * dt;
        if (!ball) return;
        const b = ball;
        if (b.state === "fall") {
          b.t += dt; b.z = Z0 * Math.max(0, 1 - (b.t / FALL_T) ** 2);
          if (b.t >= FALL_T) { b.state = "disc"; b.z = 0; api.sfx.tick();
            if (sticky) { const s = surfV(b.x, b.y); b.vx = s.x; b.vy = s.y; }   // grabs the disc speed at once
          }
        } else if (b.state === "disc") {
          const s = surfV(b.x, b.y);
          if (sticky) { b.vx = s.x; b.vy = s.y; }
          else { const k = Math.min(1, SLIP_K * dt); b.vx += (s.x - b.vx) * k; b.vy += (s.y - b.vy) * k; }
          if (sticky) {        // move exactly with the disc (rotate about C)
            const dx = b.x - C.x, dy = b.y - C.y, ca = Math.cos(w * dt), sa = Math.sin(w * dt);
            b.x = C.x + dx * ca - dy * sa; b.y = C.y + dx * sa + dy * ca;
          } else { b.x += b.vx * dt; b.y += b.vy * dt; }
          b.t += dt;
          for (const h of holes) {
            const p = holePos(h);
            if (Math.hypot(b.x - p.x, b.y - p.y) < HOLE_R - 1) {
              b.state = "sink"; b.hole = h; api.sfx.whoosh();
              if (h.dummy) return api.fail("The ball dropped through the dummy hole");
              return;
            }
          }
          if (Math.hypot(b.x - C.x, b.y - C.y) > R) { b.state = "off"; return api.fail("The ball rolled off the disc"); }
          const rel = Math.hypot(b.vx - s.x, b.vy - s.y);
          rest = rel < 6 ? rest + dt : 0;
          if (sticky && b.t > FALL_T + 0.45) return api.fail("The ball stuck to the disc, away from the hole");
          if (!sticky && rest > 0.35) return api.fail("The ball came to rest on the disc, not in the hole");
        } else if (b.state === "sink") {
          const p = holePos(b.hole); b.x += (p.x - b.x) * Math.min(1, dt * 12); b.y += (p.y - b.y) * Math.min(1, dt * 12);
          b.sink += dt;
          if (b.sink > 0.3) api.succeed("The ball fell through the target hole");
        }
      },
      draw(g) {
        g.scene();
        const ctx = g.ctx;
        // support stand + disc shadow
        g.ellipse(C.x + 8, C.y + 14, R + 6, R * 0.98, P.shadow);
        g.circle(C.x, C.y, R + 8, P.metal, P.metalDark, 2);
        const top = sticky ? "#b9a6c9" : "#e4c99f", edge = sticky ? "#8b75a0" : P.woodDark;
        g.circle(C.x, C.y, R, top, edge, 3);
        ctx.save(); ctx.translate(C.x, C.y); ctx.rotate(th);
        // rotation cues: spokes, decals, sticky texture
        for (let i = 0; i < 12; i++) {
          const a = i * Math.PI / 6;
          g.line(Math.cos(a) * 26, Math.sin(a) * 26, Math.cos(a) * (R - 6), Math.sin(a) * (R - 6), "rgba(0,0,0,0.05)", 2);
        }
        if (sticky) for (let i = 0; i < 70; i++) {
          const a = i * 2.39996, r = 18 + (R - 26) * Math.sqrt((i + 0.5) / 70);
          g.circle(Math.cos(a) * r, Math.sin(a) * r, 2.2, "rgba(80,50,110,0.22)");
        }
        for (const d of decals) {
          const x = d.r * Math.cos(d.a), y = d.r * Math.sin(d.a), k = 11;
          if (d.shape === 0) g.poly([[x, y - k], [x + k, y + k * 0.8], [x - k, y + k * 0.8]], d.color);
          else if (d.shape === 1) g.rr(x - k * 0.9, y - k * 0.9, k * 1.8, k * 1.8, 3, d.color);
          else g.poly([[x, y - k], [x + k, y], [x, y + k], [x - k, y]], d.color);
        }
        for (const h of holes) {
          const x = h.r * Math.cos(h.a), y = h.r * Math.sin(h.a);
          if (!h.dummy) g.circle(x, y, HOLE_R + 7, null, P.green, 4);
          g.circle(x, y, HOLE_R, "#1b222b", "rgba(0,0,0,0.5)", 2);
          g.circle(x + 3, y + 4, HOLE_R - 6, "#0e1319");
        }
        ctx.restore();
        g.circle(C.x, C.y, 16, P.metal, P.metalDark, 2);       // axle cap
        // target label rides with the hole
        const tp = holePos(target);
        g.chip("target", tp.x, tp.y - HOLE_R - 20, P.green);
        // ghost aim marker
        if (!ball && ghost && Math.hypot(ghost.x - C.x, ghost.y - C.y) < R - BALL_R) {
          g.circle(ghost.x, ghost.y, BALL_R, "rgba(224,75,60,0.18)", "rgba(224,75,60,0.7)", 1.5);
          g.line(ghost.x - 18, ghost.y, ghost.x + 18, ghost.y, "rgba(224,75,60,0.5)", 1);
          g.line(ghost.x, ghost.y - 18, ghost.x, ghost.y + 18, "rgba(224,75,60,0.5)", 1);
        }
        // ball: falling (shadow shrinks toward the landing spot), on the disc, or sinking
        if (ball && ball.state !== "off") {
          const b = ball;
          if (b.state === "fall") {
            const f = b.z / Z0;
            g.ellipse(b.x, b.y, BALL_R * (0.8 + f * 1.4), BALL_R * (0.6 + f), `rgba(15,34,51,${0.3 - 0.18 * f})`);
            g.ball(b.x, b.y - b.z * 0.6, BALL_R * (1 + f * 0.7), P.red);
          } else if (b.state === "sink") {
            const k = Math.max(0, 1 - b.sink / 0.3);
            ctx.save(); ctx.globalAlpha = 0.3 + 0.7 * k; g.ball(b.x, b.y, BALL_R * (0.5 + 0.5 * k), P.red); ctx.restore();
          } else { g.shadow(b.x + 2, b.y + 3, BALL_R * 0.9, 0); g.ball(b.x, b.y, BALL_R, P.red); }
        }
        g.chip(sticky ? "Sticky surface" : "Regular surface", 110, 100, sticky ? "#7b5ea7" : P.woodDark);
        g.chip(w > 0 ? "Turning clockwise" : "Turning counter-clockwise", 110, 126, P.muted);
        if (!ball) g.chip("Click once on the disc to drop the ball (it takes a moment to fall)", 405, 515, P.navy);
      },
      hud() { return [{ k: "Surface", v: sticky ? "Sticky" : "Regular" }, { k: "Drop", v: ball ? "used" : "ready" }]; }
    };
  }
});
