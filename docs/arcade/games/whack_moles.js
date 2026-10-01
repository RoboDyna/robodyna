/* Whack Moles — periodic pattern.
 * A board with a 3×3 grid of holes. Two moles bob up and down periodically (wait, rise, peak, sink),
 * each with its own randomized speed and rhythm. Click a hole to swing the hammer: it lands a moment
 * later (0.1 s) and needs ~0.55 s to recover, so a mistimed swing can cost the peak. A hit only counts
 * when the mole is well above the rim; striking an empty or sunken hole just misses.
 * Success: both moles whacked. Fail: the hammer lands on a rabbit that is showing.
 * base: two moles in fixed holes.            var1: fixed-hole moles + a rabbit distractor.
 * var2: an unhit mole reappears in a different hole after each dive (no rabbit).
 * var1+2: relocating moles and a (relocating) rabbit.
 */
Arcade.register({
  id: "whack_moles",
  maxTime: 20,
  levels: {
    base: "Two moles bob in fixed holes.",
    var1: "Fixed-hole moles plus a rabbit. Never hit the rabbit.",
    var2: "Unhit moles pop up in a different hole each time.",
    "var1+2": "Relocating moles and a rabbit."
  },
  controls: [["Click", "swing the hammer at a hole"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const HX = [255, 405, 555], HY = [205, 318, 431], HRX = 50, HRY = 17;
    const holes = []; for (const y of HY) for (const x of HX) holes.push({ x, y });
    const STRIKE = 0.1, COOL = 0.55, HIT_H = 0.55, RABBIT_H = 0.25;
    const relocate = api.var2;

    const used = new Set();
    function freeHole(except) {
      const opts = holes.map((_, i) => i).filter(i => !used.has(i) && i !== except);
      const i = pick(opts); used.add(i); return i;
    }
    function makeCritter(kind) {
      return { kind, hole: freeHole(-1), h: 0, phase: "wait", t: rand(0.4, 1.9),
        rise: rand(0.27, 0.4), hold: rand(0.22, 0.42), wait: rand(0.9, 1.9), hit: false, down: 0 };
    }
    const moles = [makeCritter("mole"), makeCritter("mole")];
    const critters = moles.slice();
    if (api.var1) critters.push(makeCritter("rabbit"));

    let hammer = null, cool = 0, acted = false, ptr = null;

    function stepCritter(c, dt) {
      if (c.hit) { c.down += dt; if (c.down > 0.35) c.h = Math.max(0, c.h - dt * 2.5); return; }
      c.t -= dt;
      if (c.phase === "wait") { c.h = 0; if (c.t <= 0) { c.phase = "rise"; c.t = c.rise; } }
      else if (c.phase === "rise") { c.h = 1 - Math.max(0, c.t) / c.rise; if (c.t <= 0) { c.phase = "hold"; c.t = c.hold; c.h = 1; } }
      else if (c.phase === "hold") { c.h = 1; if (c.t <= 0) { c.phase = "sink"; c.t = c.rise; } }
      else if (c.phase === "sink") {
        c.h = Math.max(0, c.t) / c.rise;
        if (c.t <= 0) {
          c.phase = "wait"; c.h = 0; c.t = c.wait * rand(0.9, 1.1);
          if (relocate) { used.delete(c.hole); c.hole = freeHole(c.hole); }
        }
      }
    }
    function land(hi) {
      const c = critters.find(k => k.hole === hi && !k.hit);
      if (c && c.kind === "rabbit" && c.h >= RABBIT_H) return api.fail("The hammer hit the rabbit");
      if (c && c.kind === "mole" && c.h >= HIT_H) {
        c.hit = true; api.sfx.good();
        if (moles.every(m => m.hit)) api.succeed("Whacked both moles");
        return;
      }
      api.sfx.tick();                                       // missed: empty or sunken hole
    }

    function drawCritter(g, c) {
      const ctx = g.ctx, hl = holes[c.hole], x = hl.x, y = hl.y;
      if (c.h <= 0.01) return;
      const top = y - c.h * 66;
      ctx.save(); ctx.beginPath(); ctx.rect(x - 70, y - 170, 140, 170); ctx.ellipse(x, y, HRX - 3, HRY - 2, 0, 0, Math.PI * 2); ctx.clip();
      if (c.kind === "mole") {
        g.rr(x - 27, top, 54, 90, 26, "#8b5e3c", "#6b4529", 2);
        g.ellipse(x, top + 40, 16, 18, "#b98a63");
        if (c.hit) { g.text("✕", x - 10, top + 20, { size: 13, color: P.ink, align: "center" }); g.text("✕", x + 10, top + 20, { size: 13, color: P.ink, align: "center" }); }
        else { g.circle(x - 10, top + 20, 3.5, P.ink); g.circle(x + 10, top + 20, 3.5, P.ink); }
        g.ellipse(x, top + 30, 7, 5, P.pink, "#c46f80", 1);
        g.line(x - 8, top + 32, x - 24, top + 28, "rgba(0,0,0,0.35)", 1); g.line(x + 8, top + 32, x + 24, top + 28, "rgba(0,0,0,0.35)", 1);
      } else {
        g.ellipse(x - 12, top - 22, 8, 26, "#f4f1ec", "#c9c3bb", 1.5, -0.15); g.ellipse(x - 12, top - 20, 4, 18, P.pink, null, 0, -0.15);
        g.ellipse(x + 12, top - 22, 8, 26, "#f4f1ec", "#c9c3bb", 1.5, 0.15); g.ellipse(x + 12, top - 20, 4, 18, P.pink, null, 0, 0.15);
        g.rr(x - 27, top, 54, 90, 26, "#f4f1ec", "#c9c3bb", 2);
        g.circle(x - 10, top + 20, 3.5, "#c0392b"); g.circle(x + 10, top + 20, 3.5, "#c0392b");
        g.poly([[x - 5, top + 28], [x + 5, top + 28], [x, top + 33]], P.pink);
      }
      ctx.restore();
    }

    return {
      pointerMove(x, y) { ptr = { x, y }; },
      pointerDown(x, y) {
        if (cool > 0) return;
        const hi = holes.findIndex(h => Math.abs(x - h.x) < 62 && y > h.y - 90 && y < h.y + 30);
        if (hi < 0) return;
        acted = true; cool = COOL; hammer = { hole: hi, t: 0 }; api.sfx.whoosh();
      },
      update(dt) {
        cool = Math.max(0, cool - dt);
        critters.forEach(c => stepCritter(c, dt));
        if (hammer) {
          const before = hammer.t; hammer.t += dt;
          if (before < STRIKE && hammer.t >= STRIKE) land(hammer.hole);
          if (hammer.t > STRIKE + 0.2) hammer = null;
        }
      },
      draw(g) {
        g.scene();
        const ctx = g.ctx;
        g.rr(156, 136, 510, 372, 22, P.shadow);
        g.rr(150, 126, 510, 372, 22, "#7fb069", "#5d8a4c", 3);
        g.rr(166, 142, 478, 340, 16, "#8fc077");
        for (let i = 0; i < holes.length; i++) {
          const h = holes[i];
          g.ellipse(h.x, h.y + 3, HRX + 6, HRY + 5, "#6a9a57");
          g.ellipse(h.x, h.y, HRX, HRY, "#2a1d14");
        }
        for (const c of critters) drawCritter(g, c);
        // front lip of each hole drawn over the bodies for depth
        for (const h of holes) { ctx.beginPath(); ctx.ellipse(h.x, h.y, HRX, HRY, 0, 0, Math.PI); ctx.strokeStyle = "#5c3f2a"; ctx.lineWidth = 3; ctx.stroke(); }
        for (const m of moles) if (m.hit) {
          const h = holes[m.hole];
          if (m.down < 1.2) g.chip("hit!", h.x, h.y - 80 - m.down * 20, P.green);
          else g.text("✓", h.x, h.y, { size: 18, weight: 900, color: P.green, align: "center" });
        }
        // hammer: ghost at the pointer, swing at the target hole
        if (hammer) {
          const h = holes[hammer.hole], k = Math.min(1, hammer.t / STRIKE);
          const hy = h.y - 70 + 55 * k;
          g.rr(h.x + 20, hy - 6, 70, 12, 5, P.wood, P.woodDark, 1.5);
          g.rr(h.x - 26, hy - 18, 52, 36, 8, P.metal, P.metalDark, 2);
        } else if (ptr && !api.ended) {
          ctx.save(); ctx.globalAlpha = cool > 0 ? 0.25 : 0.55;
          g.rr(ptr.x + 18, ptr.y - 50, 60, 10, 4, P.wood);
          g.rr(ptr.x - 22, ptr.y - 62, 44, 32, 7, P.metal, P.metalDark, 1.5);
          ctx.restore();
        }
        if (!acted) g.chip(api.var1 ? "Click a mole when it is up. Never the rabbit." : "Click a mole when it is up", W / 2, 520, P.navy);
      },
      hud() {
        return [{ k: "Moles", v: `${moles.filter(m => m.hit).length}/2` }, { k: "Hammer", v: cool > 0 ? "busy" : "ready", warn: cool > 0 }];
      }
    };
  }
});
