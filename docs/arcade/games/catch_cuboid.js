/* Catch Cuboid — periodic pattern.
 * Side view of a box whose top board has three slots. Cuboid(s) inside the box rise through a slot on
 * a periodic schedule (rise, short hold, sink, wait), with randomized period and phase per cuboid.
 * Click a cuboid while it sticks out of the board to pull it out (once per cuboid). Clicking a slot
 * whose cuboid is still inside fails (the gripper jams on the board). Empty slots are harmless.
 * Success: every cuboid pulled out during one of its pop-up windows.
 * base: one cuboid, transparent front board (you can see it moving inside).
 * var1: two cuboids with different periods/phases, transparent board.
 * var2: one cuboid, opaque board — invisible until it emerges, so learn the rhythm.
 * var1+2: two cuboids, opaque board.
 */
Arcade.register({
  id: "catch_cuboid",
  maxTime: 16,
  levels: {
    base: "One cuboid. The front board is transparent.",
    var1: "Two cuboids pop up with different rhythms. Transparent board.",
    var2: "One cuboid behind an opaque board: learn its rhythm.",
    "var1+2": "Two cuboids, opaque board."
  },
  controls: [["Click", "pull out a cuboid while it sticks out (once each)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const SLOTS = [265, 405, 545], SLOT_Y = 232, CW = 50, CH = 150, EXT = 142, LOW = 262, GRAB_H = 0.35;
    const BOX = { x: 170, y: 256, w: 470, h: 154 }, TABLE_Y = 410;
    const smooth = u => u * u * (3 - 2 * u);

    function makeCuboid(slot, color, name, period0) {
      const R = rand(0.32, 0.42), Hd = rand(0.22, 0.36);
      let Wt = rand(1.3, 2.3);
      if (period0) {               // second cuboid: make its period clearly different from the first
        const T = 2 * R + Hd + Wt;
        if (Math.abs(T - period0) < 0.45) Wt += 0.6;
      }
      return { slot, x: SLOTS[slot], color, name, R, Hd, Wt, T: 2 * R + Hd + Wt, t0: rand(0.8, 2.6), h: 0,
        pulled: false, py: 0, pt: 0 };
    }
    function heightAt(c, t) {
      if (t < c.t0) return 0;
      const u = (t - c.t0) % c.T;
      if (u < c.R) return smooth(u / c.R);
      if (u < c.R + c.Hd) return 1;
      if (u < 2 * c.R + c.Hd) return smooth(1 - (u - c.R - c.Hd) / c.R);
      return 0;
    }
    const slotIdx = [0, 1, 2];
    const s1 = pick(slotIdx);
    const cubs = [makeCuboid(s1, P.red, "red")];
    if (api.var1) {
      const s2 = pick(slotIdx.filter(s => s !== s1));
      cubs.push(makeCuboid(s2, P.blue, "blue", cubs[0].T));
      if (Math.abs(cubs[1].t0 - cubs[0].t0) < 0.5) cubs[1].t0 += 0.7;
    }
    const opaque = api.var2;
    let acted = false, hover = null, jam = null;

    function drawCuboid(g, c, top, alpha) {
      const ctx = g.ctx; ctx.save(); ctx.globalAlpha = alpha;
      g.poly([[c.x + CW / 2, top], [c.x + CW / 2 + 10, top - 8], [c.x + CW / 2 + 10, top + CH - 8], [c.x + CW / 2, top + CH]], Arcade.shade(c.color, -0.3));
      g.poly([[c.x - CW / 2, top], [c.x - CW / 2 + 10, top - 8], [c.x + CW / 2 + 10, top - 8], [c.x + CW / 2, top]], Arcade.shade(c.color, 0.3));
      g.rr(c.x - CW / 2, top, CW, CH, 3, c.color, Arcade.shade(c.color, -0.25), 1.5);
      ctx.restore();
    }

    return {
      pointerMove(x, y) { hover = { x, y }; },
      pointerDown(x, y) {
        if (y < 60 || y > TABLE_Y) return;
        const s = SLOTS.findIndex(sx => Math.abs(x - sx) < 48);
        if (s < 0) return;
        acted = true;
        const c = cubs.find(k => k.slot === s);
        if (!c) { api.sfx.tick(); return; }                 // empty slot: harmless
        if (c.pulled) return;
        if (c.h >= GRAB_H) { c.pulled = true; c.py = LOW - c.h * EXT; api.sfx.click(); return; }
        jam = { x: SLOTS[s], t: 0 };
        api.fail(`Grabbed the ${c.name} cuboid while it was inside the box`);
      },
      update(dt) {
        if (jam) jam.t += dt;
        for (const c of cubs) {
          if (c.pulled) { c.pt += dt; c.py -= dt * 520; continue; }
          c.h = heightAt(c, api.time);
        }
        if (cubs.every(c => c.pulled && c.pt > 0.35)) api.succeed(cubs.length > 1 ? "Pulled out both cuboids" : "Pulled out the cuboid");
      },
      draw(g) {
        const ctx = g.ctx;
        // side view: wall + table
        g.clear(P.wall);
        for (let y = 60; y < TABLE_Y; y += 70) g.line(0, y, W, y, P.wallLine, 1);
        g.rr(-10, TABLE_Y, W + 20, 160, 0, P.table);
        g.line(0, TABLE_Y, W, TABLE_Y, P.tableEdge, 3);
        g.ellipse(BOX.x + BOX.w / 2, TABLE_Y + 6, BOX.w / 2 + 30, 12, P.shadow);
        // box interior (back wall) + inside cuboids seen through the glass
        g.rr(BOX.x, BOX.y, BOX.w, BOX.h, 4, "#d9c3a3");
        if (!opaque) {
          ctx.save(); ctx.beginPath(); ctx.rect(BOX.x, BOX.y, BOX.w, BOX.h); ctx.clip();
          for (const c of cubs) if (!c.pulled) drawCuboid(g, c, LOW - c.h * EXT, 0.5);
          ctx.restore();
        }
        // front board: glass (transparent) or wood (opaque)
        if (opaque) {
          g.rr(BOX.x, BOX.y, BOX.w, BOX.h, 4, P.wood, P.woodDark, 2.5);
          for (let i = 1; i < 5; i++) g.line(BOX.x + 6, BOX.y + i * 30, BOX.x + BOX.w - 6, BOX.y + i * 30, "rgba(0,0,0,0.06)", 1.5);
        } else {
          g.rr(BOX.x, BOX.y, BOX.w, BOX.h, 4, "rgba(170,210,240,0.28)", P.woodDark, 2.5);
          g.line(BOX.x + 30, BOX.y + 20, BOX.x + 90, BOX.y + 120, "rgba(255,255,255,0.6)", 4);
          g.line(BOX.x + 50, BOX.y + 20, BOX.x + 110, BOX.y + 120, "rgba(255,255,255,0.4)", 2);
        }
        // top board (slight 3/4 view) with slots
        g.poly([[BOX.x - 8, BOX.y], [BOX.x + 14, BOX.y - 30], [BOX.x + BOX.w + 30, BOX.y - 30], [BOX.x + BOX.w + 8, BOX.y]], "#e2cfae", P.woodDark, 2);
        g.rr(BOX.x - 8, BOX.y - 2, BOX.w + 16, 12, 3, P.woodDark);
        for (let i = 0; i < 3; i++) {
          const hov = hover && Math.abs(hover.x - SLOTS[i]) < 48 && hover.y > 60 && hover.y < TABLE_Y;
          g.rr(SLOTS[i] - 31, SLOT_Y - 6, 72, 14, 3, hov ? "#3a2a1a" : "#4b3826");
          if (hov && !api.ended) g.rr(SLOTS[i] - 44, 70, 98, TABLE_Y - 78, 10, null, "rgba(19,154,154,0.35)", 2);
        }
        // emerged parts (above the slot line)
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, SLOT_Y + 1); ctx.clip();
        for (const c of cubs) if (!c.pulled) drawCuboid(g, c, LOW - c.h * EXT, 1);
        ctx.restore();
        // pulled cuboids fly up, then lie on the table at the right
        cubs.forEach((c, i) => {
          if (!c.pulled) return;
          if (c.py > -CH) {
            ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, SLOT_Y + 1); ctx.clip();
            drawCuboid(g, c, c.py, Math.max(0, 1 - c.pt * 1.5)); ctx.restore();
          }
          const lx = 665 + i * 18, ly = TABLE_Y + 28 + i * 34;
          ctx.save(); ctx.globalAlpha = Math.min(1, c.pt * 3);
          g.ellipse(lx + 60, ly + 22, 80, 8, P.shadow);
          g.rr(lx - 15, ly, CH * 0.8, 24, 3, c.color, Arcade.shade(c.color, -0.25), 1.5);
          ctx.restore();
        });
        if (jam) g.text("✕", jam.x + 5, SLOT_Y - 30, { size: 34, weight: 900, color: P.red, align: "center" });
        g.chip(opaque ? "Opaque board" : "Transparent board", BOX.x + BOX.w / 2, BOX.y + BOX.h - 18, opaque ? P.woodDark : P.blue);
        if (!acted) g.chip("Click a cuboid while it sticks out of a slot", W / 2, 505, P.navy);
      },
      hud() { return [{ k: "Pulled", v: `${cubs.filter(c => c.pulled).length}/${cubs.length}` }]; }
    };
  }
});
