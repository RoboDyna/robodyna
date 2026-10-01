/* Pick Ripe Apple — state transition.
 * Side view: two branches (left / right) hang over a basket. The good apple ripens on its branch
 * (green -> red), stays ripe for a few seconds, then spoils (brown, spotted). Pick it while it is red:
 * Left / Right sends the gripper to that branch and carries the apple above the basket; Space releases
 * it and it falls (fall time ~0.5 s) into the basket.
 * base:   one good apple (random side), static basket.
 * var1:   the other branch holds a spoiled apple that must not be picked.
 * var2:   one good apple; the basket swings left/right, so the release has to lead it.
 * var1+2: spoiled distractor + swinging basket.
 * Fails: picking an unripe or spoiled apple, letting the good apple spoil, missing the basket.
 */
Arcade.register({
  id: "pick_ripe_apple",
  maxTime: 18,
  levels: {
    base: "One apple ripens on a branch; pick it while red, drop it in the basket.",
    var1: "A spoiled apple hangs on the other branch. Pick only the good one.",
    var2: "One apple; the basket swings, so time the release.",
    "var1+2": "Spoiled distractor and a swinging basket."
  },
  controls: [["← / →", "pick the apple on that side"], ["Space", "release over the basket"]],
  keys: [{ key: "ArrowLeft", label: "◀ pick left" }, { key: " ", label: "release" }, { key: "ArrowRight", label: "pick right ▶" }],
  create(api) {
    const { P, W, rand, pick } = api;
    const R = 19, G = 1300, GROUND = 500, CARRY_T = 0.6;
    const goodSide = pick([-1, 1]);
    const branchX = { "-1": 215, "1": 595 }, APPLE_Y = 215;
    // ripening timeline of the good apple (seconds)
    const tRipe = rand(1.8, 4.2), tTurn = rand(1.2, 1.7), tWindow = rand(2.3, 3.4), tRot = 1.3;
    const good = { side: goodSide, x: branchX[goodSide], y: APPLE_Y, good: true, state: "branch" };
    const apples = [good];
    if (api.var1) apples.push({ side: -goodSide, x: branchX[-goodSide], y: APPLE_Y, good: false, state: "branch", tint: rand(0.75, 1) });
    // basket
    const hover = { x: W / 2 + (api.var2 ? rand(-45, 45) : 0), y: 290 };
    const bask = { x0: api.var2 ? W / 2 : hover.x, A: api.var2 ? rand(95, 135) : 0, w: (Math.PI * 2) / rand(2.6, 3.6), ph: rand(0, 6.28), half: 66, rim: 432 };
    const basketX = t => bask.x0 + bask.A * Math.sin(bask.w * t + bask.ph);
    let held = null, carryT = 0, falling = null, acted = false;

    // ripeness 0..1 (green -> red) and spoil 0..1 (red -> brown) of the good apple at time t
    function ripe(t) { return Math.max(0, Math.min(1, (t - tRipe) / tTurn)); }
    function spoil(t) { return Math.max(0, Math.min(1, (t - tRipe - tTurn - tWindow) / tRot)); }
    function look(a, t) {
      if (!a.good) return { r: 1, s: a.tint };
      if (a.frozen) return a.frozen;
      return { r: ripe(t), s: spoil(t) };
    }
    function mix(c1, c2, k) { return c1.map((v, i) => Math.round(v + (c2[i] - v) * k)); }
    function colorOf(L) {
      const c = mix(mix([132, 190, 64], [224, 75, 60], L.r), [128, 84, 44], L.s);
      return "#" + c.map(v => v.toString(16).padStart(2, "0")).join("");
    }

    return {
      keyDown(k) {
        if ((k === "ArrowLeft" || k === "ArrowRight") && !held && !falling) {
          const side = k === "ArrowLeft" ? -1 : 1, a = apples.find(p => p.side === side && p.state === "branch");
          if (!a) return;
          acted = true; api.sfx.click();
          const L = look(a, api.time);
          a.frozen = L; a.state = "carry"; held = a; carryT = 0;
          if (!a.good) return api.fail("Picked the spoiled apple");
          if (L.s > 0.25) return api.fail("Picked it too late: the apple had spoiled");
          if (L.r < 0.85) return api.fail("Picked it too early: the apple was not ripe yet");
        }
        if (k === " " && held && carryT >= 1) {
          falling = held; held = null; falling.state = "fall"; falling.vy = 0; api.sfx.whoosh();
        }
      },
      update(dt) {
        const t = api.time;
        if (!held && !falling && good.state === "branch" && spoil(t) > 0.6) return api.fail("The apple spoiled on the branch");
        if (held) {
          carryT = Math.min(1, carryT + dt / CARRY_T);
          const k = carryT * carryT * (3 - 2 * carryT), sx = branchX[held.side];
          held.x = sx + (hover.x - sx) * k; held.y = APPLE_Y + (hover.y - APPLE_Y) * k - Math.sin(Math.PI * k) * 50;
        }
        if (falling) {
          const a = falling, prevY = a.y;
          a.vy += G * dt; a.y += a.vy * dt;
          const bx = basketX(t);
          if (prevY < bask.rim && a.y >= bask.rim) {
            if (Math.abs(a.x - bx) < bask.half - R + 4) { a.state = "in"; a.dx = a.x - bx; falling = null; return api.succeed("Ripe apple in the basket"); }
            if (Math.abs(a.x - bx) < bask.half + R) { a.vy = -a.vy * 0.3; a.x += Math.sign(a.x - bx) * 6; api.sfx.tick(); }
          }
          if (a.y >= GROUND - R) { a.y = GROUND - R; falling = null; return api.fail("The apple missed the basket"); }
        }
      },
      draw(g) {
        const c = g.ctx, t = api.time;
        // side view: soft wall, floor strip
        g.clear(P.wall);
        c.fillStyle = "#f9f1f3"; c.fillRect(0, 0, W, 60);
        g.rr(-10, GROUND, W + 20, 60, 0, P.floor, P.tableEdge, 2);
        // two small trees at the sides with branches reaching inwards
        for (const side of [-1, 1]) {
          const tx = side < 0 ? 70 : W - 70, bx = branchX[side];
          g.rr(tx - 16, 150, 32, GROUND - 150, 8, P.woodDark);
          g.circle(tx, 120, 70, "#6fae5a"); g.circle(tx + side * -40, 95, 50, "#7dbb63"); g.circle(tx + side * 30, 160, 45, "#64a24f");
          c.save(); c.lineCap = "round"; c.strokeStyle = P.woodDark; c.lineWidth = 12;
          c.beginPath(); c.moveTo(tx, 190); c.quadraticCurveTo((tx + bx) / 2, 160, bx + side * 25, 175); c.stroke(); c.restore();
          g.ellipse(bx + side * 5, 168, 18, 8, "#5f9f48", null, 0, side * 0.5);
          g.ellipse(bx + side * 40, 172, 16, 7, "#6fae5a", null, 0, -side * 0.4);
        }
        // basket (side view): woven tub
        const bx = basketX(t);
        g.ellipse(bx, GROUND, bask.half + 6, 7, P.shadow);
        g.poly([[bx - bask.half, bask.rim], [bx + bask.half, bask.rim], [bx + bask.half - 12, GROUND - 2], [bx - bask.half + 12, GROUND - 2]], P.wood, P.woodDark, 2);
        for (let i = 1; i < 4; i++) g.line(bx - bask.half + 6 + i * 2, bask.rim + i * 17, bx + bask.half - 6 - i * 2, bask.rim + i * 17, "rgba(0,0,0,0.12)", 2);
        g.rr(bx - bask.half - 4, bask.rim - 6, bask.half * 2 + 8, 12, 6, P.woodDark);
        if (api.var2) g.line(bask.x0 - bask.A - bask.half, GROUND + 9, bask.x0 + bask.A + bask.half, GROUND + 9, P.metal, 4);
        // apples
        for (const a of apples) {
          let x = a.x, y = a.y;
          if (a.state === "in") { x = bx + a.dx; y = bask.rim + 6; }
          const L = look(a, t), col = colorOf(L);
          if (a.state === "branch") g.line(x, 176, x, y - R + 2, P.woodDark, 3);
          if (a.state === "in") { c.save(); c.beginPath(); c.rect(bx - bask.half, 0, bask.half * 2, bask.rim + 2); c.clip(); }
          g.ball(x, y, R, col);
          g.line(x, y - R + 2, x + 3, y - R - 6, "#5b3a1e", 3);
          if (L.s > 0.05) for (const [dx, dy, rr] of [[-7, 3, 4], [6, -5, 3], [3, 8, 3.5], [-4, -9, 2.5]]) g.circle(x + dx, y + dy, rr * Math.min(1, L.s * 1.4), "rgba(60,36,18,0.75)");
          if (a.state === "in") c.restore();
        }
        if (api.var2 || falling) g.line(hover.x, hover.y + R + 6, hover.x, hover.y + R + 30, "rgba(15,34,51,0.18)", 2, [3, 5]);
        // gripper: rod from the top to the apple it carries (or parked at the hover point)
        const gx = held ? held.x : hover.x, gy = held ? held.y : hover.y - 60;
        g.line(gx, 0, gx, gy - R - 10, P.metalDark, 6);
        g.rr(gx - 22, gy - R - 14, 44, 10, 4, P.metal, P.metalDark);
        const open = held ? 0 : 6;
        g.rr(gx - 22 - open, gy - R - 6, 7, 20, 3, P.metalDark); g.rr(gx + 15 + open, gy - R - 6, 7, 20, 3, P.metalDark);
        // key labels under the branches
        for (const side of [-1, 1]) if (apples.some(a => a.side === side && a.state === "branch")) g.chip(side < 0 ? "◀ pick" : "pick ▶", branchX[side], 262, P.navy);
        if (!acted) g.chip("Pick the apple when it turns red (not brown), then Space over the basket", W / 2, 528, P.navy);
        else if (held && carryT >= 1) g.chip("Space = release", W / 2, 528, P.orange);
      }
    };
  }
});
