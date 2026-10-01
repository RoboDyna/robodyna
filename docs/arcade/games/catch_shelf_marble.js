/* Catch Shelf Marble — static to dynamic.
 * Side view of three tilted wooden shelves mounted zig-zag on a wall. A red marble rolls down the top
 * shelf, drops onto the next one, bounces off the raised lip at its high end, rolls back, and so on,
 * until it leaves the lowest shelf and falls toward the floor. A bowl slides on a floor rail while
 * Left/Right are held (finite speed); catch the marble in the bowl.
 * base: the marble is held at the top until the first bowl-key press.
 * var1: the marble starts rolling as soon as the episode starts.
 * var2: one lower shelf rocks back and forth (no lips), so the side it leaves from changes.
 * var1+2: immediate motion + rocking shelf.
 */
Arcade.register({
  id: "catch_shelf_marble",
  maxTime: 22,
  levels: {
    base: "The marble waits until you first move the bowl.",
    var1: "The marble starts rolling immediately.",
    var2: "One lower shelf rocks back and forth, changing where the marble exits.",
    "var1+2": "Immediate rolling and a rocking shelf."
  },
  controls: [["← / →", "hold to slide the bowl"]],
  keys: [{ key: "ArrowLeft", label: "◀" }, { key: "ArrowRight", label: "▶" }],
  create(api) {
    const { P, W, rand, pick } = api;
    const G = 900, ROLL = 5 / 7, R = 9, THICK = 10;
    const RIM_Y = 462, FLOOR_Y = 500, BOWL_HW = 46, BOWL_SPEED = 270;
    const mir = pick([1, -1]);                           // mirror the whole layout left/right
    const X = x => (mir > 0 ? x : W - x);
    const jx = rand(-40, 40);
    // Shelves: high end (with a lip) and low end. Each lower shelf reaches past the one above.
    const raw = [
      { hi: 150, lo: 470, y: 135 },
      { hi: 610, lo: 250, y: 240 },
      { hi: 130, lo: rand(420, 520), y: 345 }
    ];
    const shelves = raw.map((s, i) => {
      const a = X(s.hi + jx), b = X(s.lo + jx);
      const cx = (a + b) / 2, half = Math.abs(a - b) / 2;
      const down = Math.sign(b - a);                      // +1: low end is on the right
      return { cx, cy: s.y, half, th: down * rand(0.16, 0.2), lip: true, rock: null, i };
    });
    if (api.var2) {                                       // a non-top shelf rocks around its centre
      const s = shelves[pick([1, 2])];
      s.lip = false;
      s.rock = { bias: Math.sign(s.th) * 0.03, amp: rand(0.15, 0.2), w: (2 * Math.PI) / rand(2.4, 3.4), ph: rand(0, 6.28) };
    }
    const m = { st: "hold", sh: shelves[0], s: 0, v: 0, x: 0, y: 0, vx: 0, vy: 0 };
    m.s = -Math.sign(shelves[0].th) * (shelves[0].half - 14);
    let started = !!api.var1, startDelay = 0.4, moved = false;
    const bowl = { x: shelves[2].cx + rand(-40, 40) };   // starts under the middle of the last shelf, away from its ends
    const keys = { ArrowLeft: false, ArrowRight: false };

    function tilt(s) { return s.rock ? s.rock.bias + s.rock.amp * Math.sin(s.rock.w * api.time + s.rock.ph) : s.th; }
    function onShelf(s, d, lift) {                        // point at arc length d along shelf, lifted by `lift`
      const t = tilt(s), c = Math.cos(t), sn = Math.sin(t);
      return { x: s.cx + d * c + sn * lift, y: s.cy + d * sn - c * lift };
    }
    function place() { const p = onShelf(m.sh, m.s, THICK / 2 + R); m.x = p.x; m.y = p.y; }
    place();

    function land(s) {
      const t = tilt(s);
      const c = Math.cos(t), sn = Math.sin(t);
      m.st = "shelf"; m.sh = s;
      m.s = (m.x - s.cx) * c + (m.y - s.cy) * sn;
      m.v = (m.vx * c + m.vy * sn) * 0.5;               // keep part of the along-shelf speed
      place(); api.sfx.tick();
    }

    return {
      debug: () => ({ m, bowl }),            // test hook
      keyDown(k) {
        if (k in keys) { keys[k] = true; moved = true; if (!started) { started = true; startDelay = 0.25; } }
      },
      keyUp(k) { if (k in keys) keys[k] = false; },
      update(dt) {
        const dir = (keys.ArrowRight ? 1 : 0) - (keys.ArrowLeft ? 1 : 0);
        if (m.st !== "in") bowl.x = Math.max(70, Math.min(W - 70, bowl.x + dir * BOWL_SPEED * dt));
        if (m.st === "hold") { if (started && (startDelay -= dt) <= 0) m.st = "shelf"; else { place(); return; } }
        if (m.st === "shelf") {
          const s = m.sh, t = tilt(s);
          m.v += G * ROLL * Math.sin(t) * dt;
          m.v *= 1 - 0.08 * dt;
          m.s += m.v * dt;
          const high = -Math.sign(t);                      // side of the raised lip
          if (s.lip && m.s * high > s.half - R && m.v * high > 0) {
            m.s = high * (s.half - R); m.v = -m.v * 0.45; api.sfx.tick();
          }
          if (Math.abs(m.s) > s.half + 2) {                // rolled off the end
            const c = Math.cos(t), sn = Math.sin(t);
            place(); m.vx = m.v * c; m.vy = m.v * sn; m.st = "air"; m.from = s;
            return;
          }
          place();
        } else if (m.st === "air") {
          const px = m.x, py = m.y;
          m.vy += G * dt; m.x += m.vx * dt; m.y += m.vy * dt;
          for (const s of shelves) {                       // land on a lower shelf crossing its top surface
            if (s === m.from) continue;
            const t = tilt(s), c = Math.cos(t), sn = Math.sin(t);
            const d = (m.x - s.cx) * c + (m.y - s.cy) * sn;
            if (Math.abs(d) > s.half) continue;
            const top = s.cy + d * sn - THICK / 2 - R;
            const topPrev = s.cy + ((px - s.cx) * c + (py - s.cy) * sn) * sn - THICK / 2 - R;
            if (py <= topPrev + 1 && m.y >= top) { m.y = top; land(s); return; }
          }
          if (py < RIM_Y && m.y >= RIM_Y) {               // crossing the bowl's rim height
            const off = Math.abs(m.x - bowl.x);
            if (off < BOWL_HW - R * 0.6) { m.st = "in"; m.off = m.x - bowl.x; api.sfx.tick(); return api.succeed("Caught the red marble"); }
            if (off < BOWL_HW + R) { m.vx = Math.sign(m.x - bowl.x) * 140; m.vy = -160; api.sfx.tick(); }
          }
          if (m.y > FLOOR_Y - R) { m.y = FLOOR_Y - R; m.st = "floor"; return api.fail("The marble hit the floor"); }
          if (m.x < -20 || m.x > W + 20) return api.fail("The marble rolled away");
        }
      },
      draw(g) {
        const c = g.ctx;
        g.clear(P.wall);
        // wall tiles and floor
        for (let y = 40; y < FLOOR_Y; y += 60) g.line(0, y, W, y, P.wallLine, 1);
        c.fillStyle = P.floor; c.fillRect(0, FLOOR_Y, W, 540 - FLOOR_Y);
        g.line(0, FLOOR_Y, W, FLOOR_Y, P.tableEdge, 3);
        // bowl rail
        g.rr(50, FLOOR_Y - 8, W - 100, 6, 3, P.metal);
        // shelves
        for (const s of shelves) {
          const t = tilt(s);
          c.save(); c.translate(s.cx, s.cy); c.rotate(t);
          g.rr(-s.half + 3, -THICK / 2 + 5, s.half * 2, THICK, 3, P.shadow);
          g.rr(-s.half, -THICK / 2, s.half * 2, THICK, 3, P.wood, P.woodDark, 1.5);
          if (s.lip) { const hx = -Math.sign(t) * (s.half - 3); g.rr(hx - 3, -THICK / 2 - 12, 6, 14, 2, P.woodDark); }
          c.restore();
          // bracket to the wall (or the pivot of a rocking shelf)
          if (s.rock) { g.circle(s.cx, s.cy, 7, P.metal, P.metalDark, 2); g.circle(s.cx, s.cy, 2.5, P.metalDark); }
          else { g.poly([[s.cx - 8, s.cy + 5], [s.cx + 8, s.cy + 5], [s.cx, s.cy + 30]], P.metal); }
        }
        // start gate (base): a small peg holding the marble
        if (m.st === "hold" && !started) {
          const p = onShelf(shelves[0], m.s + Math.sign(shelves[0].th) * 14, THICK / 2 + 8);
          g.rr(p.x - 3, p.y - 10, 6, 18, 3, P.metalDark);
        }
        // bowl (side view)
        const bx = bowl.x;
        g.ellipse(bx, FLOOR_Y - 2, BOWL_HW * 0.9, 5, P.shadow);
        c.beginPath(); c.moveTo(bx - BOWL_HW, RIM_Y);
        c.quadraticCurveTo(bx - BOWL_HW + 4, FLOOR_Y - 10, bx, FLOOR_Y - 10);
        c.quadraticCurveTo(bx + BOWL_HW - 4, FLOOR_Y - 10, bx + BOWL_HW, RIM_Y);
        c.closePath(); c.fillStyle = P.blue; c.fill();
        g.ellipse(bx, RIM_Y, BOWL_HW, 6, "#24558f", "#1f4f8a", 2);
        // marble
        if (m.st === "in") g.ball(bx + m.off * 0.6, RIM_Y + 6, R, P.red);
        else g.ball(m.x, m.y, R, P.red);
        if (!moved) g.chip(api.var1 ? "Hold ← / → to move the bowl — the marble is rolling!" : "Hold ← / → to move the bowl (the marble starts when you do)", W / 2, 522, P.navy);
      }
    };
  }
});
