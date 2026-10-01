/* Pour Beer — household.
 * Side view of a glass mug under a beer tap. Holding Space (or holding the tap handle) pours: the beer
 * level rises steadily, but the foam head builds faster and faster the longer you keep pouring. After
 * you release, the foam slowly settles (part of it turns back into beer). Foam over the rim fails.
 * Ring the bell to finish: the top of the drink must be inside the target band with only a modest,
 * settled head of foam.
 */
Arcade.register({
  id: "pour_beer",
  maxTime: 35,
  levels: { base: "Pour in bursts, let the foam settle, then ring the bell." },
  controls: [["Space (hold)", "pour"], ["Click", "the bell to finish"]],
  keys: [{ key: " ", label: "Hold to pour" }],
  create(api) {
    const { P, W, rand } = api;
    const MX = 330, MW = 170, MB = 468, MH = 250, MT = MB - MH;   // mug inner box (side view)
    const BELL = { x: 640, y: 440, r: 34 };
    const TAP = { x: MX, y: 92 };
    const qL = rand(0.075, 0.095), k = rand(0.42, 0.55), conv = 0.3, maxHead = 0.12;
    const lo = rand(0.58, 0.72), hi = lo + 0.1;
    let L = 0, F = 0, hold = 0, pouring = false, keyHeld = false, mouseHeld = false, acted = false, stream = 0;
    const bubbles = Array.from({ length: 26 }, () => ({ x: rand(-1, 1), y: rand(0, 1), r: rand(1.5, 3.2), s: rand(0.1, 0.3) }));

    function finish() {
      if (api.ended) return;
      api.sfx.click();
      const top = L + F;
      if (L < 0.05) return api.fail("The mug is empty");
      if (top < lo) return api.fail("Not full enough: below the target band");
      if (top > hi) return api.fail("Too full: above the target band");
      if (F > maxHead) return api.fail("Too much foam: let the head settle first");
      api.succeed("A well-poured beer");
    }
    const lvlY = f => MB - f * MH;

    return {
      debug: () => ({ L, F, lo, hi }),       // test hook
      keyDown(key) { if (key === " ") { keyHeld = true; acted = true; } },
      keyUp(key) { if (key === " ") keyHeld = false; },
      pointerDown(x, y) {
        if (Math.hypot(x - BELL.x, y - BELL.y) < BELL.r + 12) return finish();
        if (x > TAP.x - 90 && x < TAP.x + 40 && y < TAP.y + 40) { mouseHeld = true; acted = true; }
      },
      pointerUp() { mouseHeld = false; },
      update(dt) {
        pouring = keyHeld || mouseHeld;
        if (pouring) {
          hold += dt;
          const fr = qL * (0.6 + 2.4 * Math.min(hold / 2, 1));     // foam ramps up while held
          L += qL * dt; F += fr * dt;
        } else hold = Math.max(0, hold - 1.5 * dt);
        const s = k * F * dt; F -= s; L += conv * s;                  // foam settles into beer
        stream += ((pouring ? 1 : 0) - stream) * Math.min(1, 12 * dt);
        for (const b of bubbles) { b.y += b.s * dt; if (b.y > 1) b.y -= 1; }
        if (L + F > 1) { F = 1 - L; return api.fail("Foam overflowed the rim"); }
      },
      hud() { return [{ k: "Head", v: F > maxHead ? "foamy" : "settled", warn: F > maxHead }]; },
      draw(g) {
        const c = g.ctx;
        g.clear(P.wall);
        for (let y = 40; y < 480; y += 60) g.line(0, y, W, y, P.wallLine, 1);
        c.fillStyle = P.floor; c.fillRect(0, 480, W, 60); g.line(0, 480, W, 480, P.tableEdge, 3);
        // tap: wall bar, body, spout, handle (tilts while pouring)
        g.rr(TAP.x - 140, 40, 220, 18, 6, P.metalDark);
        g.rr(TAP.x - 18, 56, 36, 32, 6, P.metal, P.metalDark, 1.5);
        g.rr(TAP.x - 6, 84, 12, 16, 3, P.metalDark);
        c.save(); c.translate(TAP.x - 4, 60); c.rotate(pouring ? -0.55 : 0);
        g.rr(-6, -62, 12, 62, 5, P.black); g.rr(-8, -70, 16, 16, 6, P.orange);
        c.restore();
        // stream
        if (stream > 0.05) {
          const top = TAP.y + 8, bot = lvlY(L + F) + 2;
          c.save(); c.globalAlpha = 0.85 * stream;
          g.rr(TAP.x - 3 * stream, top, 6 * stream, Math.max(0, bot - top), 3, "#e2a33b");
          c.restore();
        }
        // mug: shadow, glass, beer, foam, glass outline, handle
        g.ellipse(MX, MB + 14, MW * 0.62, 8, P.shadow);
        g.rr(MX - MW / 2 - 8, MT - 6, MW + 16, MH + 20, 14, "rgba(255,255,255,0.55)");
        const ly = lvlY(L), fy = lvlY(L + F);
        if (L > 0) {
          const gr = c.createLinearGradient(0, ly, 0, MB);
          gr.addColorStop(0, "#f2b84b"); gr.addColorStop(1, "#d9861f");
          g.rr(MX - MW / 2, ly, MW, MB - ly, 8, gr);
          for (const b of bubbles) { const by = MB - b.y * (MB - ly); if (by > ly + 4) g.circle(MX + b.x * (MW / 2 - 8), by, b.r * 0.6, "rgba(255,255,255,0.35)"); }
        }
        if (F > 0.003) {
          g.rr(MX - MW / 2, fy, MW, ly - fy + 4, 6, "#fffaf0");
          for (let i = 0; i < 9; i++) g.circle(MX - MW / 2 + 10 + i * (MW - 20) / 8, fy + 2, 9, "#fffaf0");
        }
        // target band and rim
        c.fillStyle = "rgba(63,165,91,0.13)"; c.fillRect(MX - MW / 2, lvlY(hi), MW, lvlY(lo) - lvlY(hi));
        g.line(MX - MW / 2 - 14, lvlY(lo), MX + MW / 2 + 4, lvlY(lo), P.green, 2, [6, 4]);
        g.line(MX - MW / 2 - 14, lvlY(hi), MX + MW / 2 + 4, lvlY(hi), P.green, 2, [6, 4]);
        g.text("target", MX - MW / 2 - 18, (lvlY(lo) + lvlY(hi)) / 2, { size: 12, color: P.green, align: "right" });
        g.line(MX - MW / 2 - 14, MT, MX - MW / 2 - 2, MT, P.red, 2);
        g.text("rim", MX - MW / 2 - 18, MT, { size: 12, color: P.red, align: "right" });
        g.rr(MX - MW / 2 - 8, MT - 6, MW + 16, MH + 20, 14, null, "rgba(111,124,136,0.75)", 3);
        c.lineWidth = 16; c.strokeStyle = "rgba(154,166,178,0.8)";
        c.beginPath(); c.moveTo(MX + MW / 2 + 8, MT + 40);
        c.bezierCurveTo(MX + MW / 2 + 85, MT + 40, MX + MW / 2 + 85, MB - 50, MX + MW / 2 + 8, MB - 50); c.stroke();
        // bell
        g.ellipse(BELL.x, BELL.y + 26, 46, 8, P.shadow);
        g.rr(BELL.x - 44, BELL.y + 14, 88, 12, 5, P.woodDark);
        c.beginPath(); c.arc(BELL.x, BELL.y + 14, BELL.r, Math.PI, 0); c.closePath();
        const bg = c.createLinearGradient(BELL.x - BELL.r, 0, BELL.x + BELL.r, 0);
        bg.addColorStop(0, "#f7d774"); bg.addColorStop(0.5, P.yellow); bg.addColorStop(1, "#c99a1a");
        c.fillStyle = bg; c.fill();
        g.circle(BELL.x, BELL.y - BELL.r + 10, 6, P.metalDark);
        g.chip("Click to finish", BELL.x, BELL.y + 50, P.navy);
        if (!acted) g.chip("Hold Space (or hold the tap handle) to pour", W / 2, 510, P.navy);
      }
    };
  }
});
