/* Pack Fruits — state transition.
 * Top-down: two parallel belts (different speeds) carry apples toward the end of the table, where
 * unpacked fruit falls off. Click a fruit to select it (it keeps riding the belt), then click a basket:
 * the gripper carries it there (short travel, one fruit at a time). Arrival times are spaced so every
 * apple can be packed, but fruit on the fast belt overtakes fruit on the slow one.
 * base:   red apples only, one basket, apples on either belt.
 * var1:   red and green apples (one at a time), red and green baskets.
 * var2:   red apples plus black distractor apples that must be left on the belt.
 * var1+2: two colors + black distractors.
 * Fails: a colored apple falls off the belt, an apple in the wrong basket, a black apple in any basket.
 */
Arcade.register({
  id: "pack_fruits",
  maxTime: 24,
  levels: {
    base: "Red apples on both belts, one basket.",
    var1: "Red and green apples arrive one at a time; match the baskets.",
    var2: "Red apples plus black distractors. Leave the black ones.",
    "var1+2": "Two colors and black distractors."
  },
  controls: [["Click", "a fruit to select it"], ["Click", "a basket to pack the selected fruit"]],
  create(api) {
    const { P, W, rand, pick, randInt } = api;
    const two = api.var1, R = 17, X0 = 46, XEND = 630, FEED = 92, TRAVEL = 0.45;
    const belts = [{ y: 165, v: rand(68, 92) }, { y: 285, v: rand(108, 145) }];
    if (api.chance(0.5)) [belts[0].v, belts[1].v] = [belts[1].v, belts[0].v];
    const baskets = two
      ? [{ x: 255, y: 425, c: "red" }, { x: 555, y: 425, c: "green" }]
      : [{ x: 405, y: 425, c: "red" }];
    // Build fruit with spaced arrival times at the belt end (so each one can be packed in time).
    const nCol = randInt(4, 5), fruits = [];
    let T = rand(3.6, 4.6);
    const cols = [];
    for (let i = 0; i < nCol; i++) cols.push(two ? pick(["red", "green"]) : "red");
    if (two && cols.every(c => c === cols[0])) cols[randInt(1, nCol - 1)] = cols[0] === "red" ? "green" : "red";
    for (let i = 0; i < nCol; i++) {
      const b = api.randInt(0, 1);
      fruits.push({ c: cols[i], b, x: XEND - belts[b].v * T, y: belts[b].y + rand(-10, 10), state: "belt" });
      T += two ? rand(1.8, 2.4) : rand(1.1, 1.8);
    }
    if (api.var2) {
      const nBlack = randInt(1, 2);
      for (let i = 0; i < nBlack; i++) {
        let b, Tb, tries = 0;
        do { b = api.randInt(0, 1); Tb = rand(3.2, T - 1); tries++; }
        while (tries < 40 && fruits.some(f => f.b === b && Math.abs((XEND - f.x) / belts[b].v - Tb) * belts[b].v < 52));
        fruits.push({ c: "black", b, x: XEND - belts[b].v * Tb, y: belts[b].y + rand(-10, 10), state: "belt" });
      }
    }
    let sel = null, moving = null, hover = null, packed = 0, acted = false;
    const need = fruits.filter(f => f.c !== "black").length;

    function visible(f) { return f.state === "belt" && f.x > FEED - 4; }
    return {
      pointerMove(x, y) { hover = { x, y }; },
      pointerDown(x, y) {
        // basket click with a selection -> pack
        const bk = baskets.find(b => Math.abs(x - b.x) < 80 && Math.abs(y - b.y) < 55);
        if (bk) {
          if (sel && !moving && sel.state === "belt") {
            moving = sel; sel.state = "move"; sel.t = 0; sel.from = { x: sel.x, y: sel.y }; sel.to = bk; sel = null; acted = true;
            api.sfx.whoosh();
          }
          return;
        }
        let best = null, bd = 30;
        for (const f of fruits) if (visible(f) && f !== moving) { const d = Math.hypot(f.x - x, f.y - y); if (d < bd) { bd = d; best = f; } }
        sel = best; if (best) { api.sfx.click(); acted = true; }
      },
      update(dt) {
        for (const f of fruits) {
          if (f.state === "belt") {
            f.x += belts[f.b].v * dt;
            if (f.x > XEND + 6) { f.state = "drop"; f.t = 0; if (sel === f) sel = null; if (f.c !== "black") return api.fail(`A ${f.c} apple fell off the belt`); }
          } else if (f.state === "drop") { f.t += dt; f.x += belts[f.b].v * dt; }
          else if (f.state === "move") {
            f.t += dt / TRAVEL;
            const k = Math.min(1, f.t), e = k * k * (3 - 2 * k);
            f.x = f.from.x + (f.to.x - f.from.x) * e; f.y = f.from.y + (f.to.y - f.from.y) * e; f.z = Math.sin(Math.PI * k) * 50;
            if (f.t >= 1) {
              f.state = "in"; f.z = 0; moving = null; api.sfx.tick();
              f.x = f.to.x + rand(-40, 40); f.y = f.to.y + rand(-14, 14);
              if (f.c === "black") return api.fail("A black distractor went into a basket");
              if (f.c !== f.to.c) return api.fail(`A ${f.c} apple went into the ${f.to.c} basket`);
              packed++;
              if (packed === need) return api.succeed("All apples packed");
            }
          }
        }
      },
      draw(g) {
        g.scene();
        const c = g.ctx;
        // belts
        for (const b of belts) {
          g.rr(X0 - 6, b.y - 42, XEND - X0 + 12, 84, 10, P.metal, P.metalDark, 2);
          g.rr(X0, b.y - 36, XEND - X0, 72, 6, "#3c434d");
          c.save(); c.beginPath(); c.rect(X0, b.y - 36, XEND - X0, 72); c.clip();
          const off = (api.time * b.v) % 26;
          for (let x = X0 - 26 + off; x < XEND; x += 26) g.line(x, b.y - 36, x, b.y + 36, "rgba(255,255,255,0.08)", 3);
          c.restore();
          g.text("▶", XEND + 22, b.y, { size: 16, color: P.metal, align: "center" });
        }
        // table end: drop-off zone
        g.rr(XEND + 40, 90, 110, 260, 10, P.badZone);
        g.text("falls off", XEND + 95, 112, { size: 11, color: P.red, align: "center" });
        // baskets
        for (const b of baskets) {
          const hot = sel && hover && Math.abs(hover.x - b.x) < 80 && Math.abs(hover.y - b.y) < 55;
          g.rr(b.x - 76, b.y - 46, 160, 100, 18, P.shadow);
          g.rr(b.x - 80, b.y - 50, 160, 100, 18, P.wood, hot ? P.navy : P.woodDark, hot ? 4 : 3);
          g.rr(b.x - 68, b.y - 38, 136, 76, 12, "#b48f62");
          for (let i = -2; i <= 2; i++) g.line(b.x + i * 26, b.y - 36, b.x + i * 26, b.y + 36, "rgba(0,0,0,0.08)", 2);
          g.chip(b.c === "red" ? "red apples" : "green apples", b.x, b.y + 62, b.c === "red" ? P.red : P.green);
        }
        // fruit
        const order = fruits.filter(f => f.state !== "move").concat(fruits.filter(f => f.state === "move"));
        for (const f of order) {
          if (f.state === "belt" && !visible(f)) continue;
          let r = R, a = 1;
          if (f.state === "drop") { r = R * Math.max(0.4, 1 - f.t * 0.8); a = Math.max(0, 1 - f.t * 1.5); if (a <= 0) continue; }
          c.save(); c.globalAlpha = a;
          const z = f.z || 0;
          g.shadow(f.x + 2, f.y + 4, r, z);
          const col = f.c === "red" ? P.red : f.c === "green" ? "#7dbb3f" : P.black;
          g.ball(f.x, f.y - z * 0.5, r * (1 + z / 300), col);
          g.line(f.x + 1, f.y - z * 0.5 - 3, f.x + 4, f.y - z * 0.5 - 9, "#5b3a1e", 3);
          if (f === sel) g.circle(f.x, f.y, r + 7, null, P.yellow, 4);
          c.restore();
        }
        // feeder hood at the belt start
        g.rr(X0 - 10, 110, FEED - X0 + 14, 230, 10, P.metalDark);
        g.rr(X0 - 4, 118, FEED - X0 + 2, 214, 6, "#4f5964");
        if (!acted) g.chip("Click an apple, then click its basket before it falls off", W / 2, 521, P.navy);
        else if (sel) g.chip("Now click a basket", W / 2, 521, P.orange);
      },
      dbg: { fruits, baskets },
      hud() { return [{ k: "Packed", v: `${packed}/${need}` }]; }
    };
  }
});
