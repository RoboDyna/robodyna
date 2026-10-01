/* Make Soup — household, dynamic avoidance (side view).
 * The robot holds a chopping board by its lip (right end) with chopped vegetables on it. Click to move the
 * lip over that spot (finite glide speed); hold Right to tip the board toward the pot, Left to tip it back.
 * Pieces stick until the board is steep enough for their own friction, then slide off the lip and fall.
 * The board pivots about the lip, so its far end swings up at speed omega * r: tipping fast and then
 * stopping tosses the far pieces into the air, over the pot and past the rim. Tap the key to tip gently.
 * Success: every piece lands inside the pot. Any piece on the rim, the stove or the counter fails.
 */
Arcade.register({
  id: "make_soup",
  maxTime: 25,
  levels: { base: "Tip the vegetables off the board into the pot without spilling any." },
  controls: [["Click", "move the board's lip over that spot"], ["→ / ←", "hold to tip toward the pot / back"]],
  keys: [{ key: "ArrowLeft", label: "◀ back" }, { key: "ArrowRight", label: "tip ▶" }],
  create(api) {
    const { P, W, rand, pick } = api;
    const G = 900, BL = 250, BT = 12, PIV_Y = 205, GLIDE_V = 320;
    const ALPHA = 2.2, OMAX = 1.7, BETA = 4.5, TH_MIN = -0.12, TH_MAX = 1.05;
    const COUNTER_Y = 450, RIM_Y = 322, WALL = 10;
    const pot = { x: rand(540, 660), w: rand(150, 180) };
    const board = { x: 300, tx: 300, th: 0, om: 0 };
    let keyL = false, keyR = false, acted = false, ghost = null;

    // Vegetable pieces: r = distance from the lip along the board.
    const kinds = [
      { color: P.orange, shape: "disc" }, { color: P.green, shape: "cube" }, { color: P.red, shape: "disc" },
      { color: "#f5eecd", shape: "cube" }, { color: P.green, shape: "disc" }, { color: P.orange, shape: "cube" }
    ];
    const n = api.randInt(4, 5), pieces = [];
    for (let k = 0; k < n; k++) {
      const kd = kinds[(k + api.randInt(0, 5)) % kinds.length];
      pieces.push({ ...kd, rad: rand(9, 12), mu: rand(0.38, 0.6), r: 40 + k * (170 / (n - 1)) + rand(-6, 6),
        rd: 0, vn: 0, state: "board", x: 0, y: 0, vx: 0, vy: 0, spin: rand(0, 6) });
    }
    const dirv = th => ({ x: -Math.cos(th), y: -Math.sin(th) });     // lip -> far end
    const nrm = th => ({ x: Math.sin(th), y: -Math.cos(th) });       // board's upper normal
    function onBoardPos(p) {
      const d = dirv(board.th), nn = nrm(board.th), o = p.rad + BT / 2;
      return { x: board.x + d.x * p.r + nn.x * o, y: PIV_Y + d.y * p.r + nn.y * o };
    }
    function launch(p, glideVx) {
      const d = dirv(board.th), nn = nrm(board.th), q = onBoardPos(p);
      p.x = q.x; p.y = q.y; p.vx = d.x * p.rd + nn.x * p.vn + glideVx; p.vy = d.y * p.rd + nn.y * p.vn;
      p.state = "air";
    }

    function update(dt) {
      // board glide and tilt (angular velocity ramps up while a key is held, eases out on release)
      const dx = board.tx - board.x, gv = Math.sign(dx) * Math.min(Math.abs(dx), GLIDE_V * dt);
      board.x += gv;
      const want = keyR && !keyL ? 1 : keyL && !keyR ? -1 : 0;
      if (want) board.om = Math.max(-OMAX, Math.min(OMAX, board.om + want * ALPHA * dt));
      else board.om = Math.sign(board.om) * Math.max(0, Math.abs(board.om) - BETA * dt);
      let decel = !want ? BETA : 0;
      board.th += board.om * dt;
      if (board.th > TH_MAX || board.th < TH_MIN) { board.th = Math.max(TH_MIN, Math.min(TH_MAX, board.th)); board.om = 0; decel = 1e4; }
      const s = Math.sin(board.th), c = Math.cos(board.th);
      // pieces on the board
      const onb = pieces.filter(p => p.state === "board").sort((a, b) => a.r - b.r);
      for (const p of onb) {
        // separation: the far end decelerates faster than gravity can pull the piece down with it
        const surf = board.om * p.r;
        if (p.vn - G * c * dt > surf + 1 && p.vn > 60 && decel * p.r > G * c) { launch(p, gv / dt); api.sfx.whoosh(); continue; }
        p.vn = surf;
        const drive = -G * s + board.om * board.om * p.r;     // along +r (away from the lip)
        if (Math.abs(p.rd) < 3 && Math.abs(drive) <= p.mu * G * c) { p.rd = 0; continue; }
        const fr = p.mu * 0.8 * G * c * (p.rd !== 0 ? -Math.sign(p.rd) : -Math.sign(drive));
        p.rd += (drive + fr) * dt;
        p.r += p.rd * dt;
        if (p.r < 0 || p.r > BL) { launch(p, gv / dt); api.sfx.tick(); }
      }
      // keep pieces from passing through each other on the board
      for (let i = 1; i < onb.length; i++) {
        const a = onb[i - 1], b = onb[i];
        if (a.state === "board" && b.state === "board" && b.r - a.r < a.rad + b.rad) {
          b.r = a.r + a.rad + b.rad; const v = Math.min(a.rd, b.rd); a.rd = b.rd = v;
        }
      }
      // flying pieces
      for (const p of pieces) {
        if (p.state === "air") {
          const py = p.y;
          p.vy += G * dt; p.x += p.vx * dt; p.y += p.vy * dt; p.spin += dt * 6;
          // land back on the board?
          const d = dirv(board.th), nn = nrm(board.th), rx = p.x - board.x, ry = p.y - PIV_Y;
          const rr = rx * d.x + ry * d.y, hh = rx * nn.x + ry * nn.y - p.rad - BT / 2;
          if (rr > 0 && rr < BL && hh <= 0 && hh > -18 && p.vx * nn.x + p.vy * nn.y < board.om * rr) {
            p.state = "board"; p.r = rr; p.rd = p.vx * d.x + p.vy * d.y; p.vn = board.om * rr; continue;
          }
          if (py < RIM_Y && p.y >= RIM_Y) {
            const off = Math.abs(p.x - pot.x), inner = pot.w / 2 - WALL;
            if (off < inner - p.rad * 0.4) { p.state = "pot"; p.vx *= 0.3; api.sfx.tick(); }
            else if (off < pot.w / 2 + p.rad) { p.state = "spill"; p.vy = -180; p.vx = Math.sign(p.x - pot.x) * 120; if (!api.ended) api.fail("A piece hit the pot's rim and bounced out"); }
          }
          if (p.y > COUNTER_Y - p.rad) { p.y = COUNTER_Y - p.rad; p.state = "spilled"; if (!api.ended) api.fail("A piece missed the pot and fell on the stove"); }
        } else if (p.state === "pot" || p.state === "spill") {
          p.vy += G * dt; p.x += p.vx * dt; p.y += p.vy * dt;
          const floor = p.state === "pot" ? COUNTER_Y - 60 - p.rad : COUNTER_Y - p.rad;
          if (p.state === "pot") p.x = Math.max(pot.x - pot.w / 2 + WALL + p.rad, Math.min(pot.x + pot.w / 2 - WALL - p.rad, p.x));
          if (p.y > floor) { p.y = floor; p.vy = 0; p.vx = 0; if (p.state === "pot") p.state = "in"; else p.state = "spilled"; }
        }
      }
      if (!api.ended && pieces.every(p => p.state === "in")) api.succeed("Every piece made it into the soup");
    }

    function drawPiece(g, x, y, p, ang) {
      const c = g.ctx;
      c.save(); c.translate(x, y); c.rotate(ang);
      if (p.shape === "disc") { g.ellipse(0, 0, p.rad, p.rad * 0.75, p.color, Arcade.shade(p.color, -0.25), 2); g.ellipse(0, -1, p.rad * 0.45, p.rad * 0.3, "rgba(255,255,255,0.35)"); }
      else g.rr(-p.rad, -p.rad, 2 * p.rad, 2 * p.rad, 3, p.color, Arcade.shade(p.color, -0.25), 2);
      c.restore();
    }

    return {
      _test: { pot, pieces, board },                                          // for the automated tests
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x) { board.tx = Math.max(270, Math.min(W - 40, x)); acted = true; api.sfx.click(); },
      keyDown(k) { if (k === "ArrowLeft") keyL = true; if (k === "ArrowRight") { keyR = true; acted = true; } },
      keyUp(k) { if (k === "ArrowLeft") keyL = false; if (k === "ArrowRight") keyR = false; },
      update(dt) { const m = Math.ceil(dt / (1 / 240)); for (let i = 0; i < m; i++) update(dt / m); },
      draw(g) {
        const c = g.ctx;
        // kitchen side view: back wall, tiles, counter with a stove
        g.clear(P.wall);
        for (let x = 0; x < W; x += 54) g.line(x, 0, x, COUNTER_Y, P.wallLine, 1);
        for (let y = 0; y < COUNTER_Y; y += 54) g.line(0, y, W, y, P.wallLine, 1);
        g.rr(0, COUNTER_Y, W, 16, 0, P.table, P.tableEdge);
        g.rr(0, COUNTER_Y + 16, W, 100, 0, P.floor);
        g.rr(pot.x - pot.w / 2 - 30, COUNTER_Y - 10, pot.w + 60, 12, 4, P.black);
        // pot (back half, soup, then front walls)
        const px0 = pot.x - pot.w / 2, px1 = pot.x + pot.w / 2;
        g.rr(px0, RIM_Y, pot.w, COUNTER_Y - 10 - RIM_Y, 10, "#7d8a96");
        g.rr(px0 + WALL, COUNTER_Y - 80, pot.w - 2 * WALL, 66, 6, "#e9a23b");
        for (const p of pieces) if (p.state === "in" || p.state === "pot") drawPiece(g, p.x, p.y, p, p.spin);
        g.rr(px0, RIM_Y, WALL, COUNTER_Y - 10 - RIM_Y, 4, P.metal, P.metalDark, 2);
        g.rr(px1 - WALL, RIM_Y, WALL, COUNTER_Y - 10 - RIM_Y, 4, P.metal, P.metalDark, 2);
        g.rr(px0 - 6, RIM_Y - 4, WALL + 8, 8, 3, P.metalDark);
        g.rr(px1 - WALL - 2, RIM_Y - 4, WALL + 8, 8, 3, P.metalDark);
        g.rr(px0 + WALL, COUNTER_Y - 30, pot.w - 2 * WALL, 18, 4, "rgba(111,124,136,0.55)");
        // steam
        for (let k = 0; k < 3; k++) {
          const t = (api.time * 0.6 + k / 3) % 1;
          g.circle(pot.x - 30 + k * 30, RIM_Y - 10 - t * 60, 8 + t * 10, `rgba(255,255,255,${(0.6 * (1 - t)).toFixed(2)})`);
        }
        // ghost lip position
        if (ghost && !keyL && !keyR) g.line(Math.max(270, Math.min(W - 40, ghost.x)), PIV_Y + 20, Math.max(270, Math.min(W - 40, ghost.x)), RIM_Y - 10, "rgba(11,42,69,0.25)", 2, [5, 6]);
        // board (rotated about the lip) + robot gripper at the lip
        c.save(); c.translate(board.x, PIV_Y); c.rotate(board.th);
        g.rr(-BL, -BT / 2, BL, BT, 4, P.wood, P.woodDark, 2);
        g.rr(-BL + 4, -BT / 2 + 2, BL - 8, 3, 2, "rgba(255,255,255,0.25)");
        c.restore();
        g.line(board.x + 6, 40, board.x + 6, PIV_Y - 12, P.navy, 10);
        g.rr(board.x - 6, PIV_Y - 16, 24, 32, 6, P.navy);
        for (const p of pieces) {
          if (p.state === "board") { const q = onBoardPos(p); drawPiece(g, q.x, q.y, p, board.th); }
          else if (p.state !== "in" && p.state !== "pot") drawPiece(g, p.x, p.y, p, p.spin);
        }
        // tilt gauge
        g.text(`tilt ${Math.round((board.th * 180) / Math.PI)}°`, 22, 24, { size: 13, color: P.muted });
        if (!acted) g.chip("Click to move the board over the pot, then hold → (tap for gentle tipping)", W / 2, 500, P.navy);
      },
      hud() { return [{ k: "In pot", v: `${pieces.filter(p => p.state === "in" || p.state === "pot").length}/${pieces.length}` }]; }
    };
  }
});
