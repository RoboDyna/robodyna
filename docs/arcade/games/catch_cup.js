/* Catch Cup — household.
 * Side view of a kitchen counter with a wall shelf above it. A mug slowly slides toward the shelf edge; once
 * its centre of mass passes the edge it tips over (rotation accelerating about the edge), comes off and
 * falls with the sideways speed it picked up. Click once to drop the pillow on the counter where the mug
 * will land; it must land on the pillow, not on the bare counter.
 * Randomized: tipping side, shelf height and edge position, slide speed and tipping strength.
 */
Arcade.register({
  id: "catch_cup",
  maxTime: 10,
  levels: { base: "A mug tips off the shelf edge; place the pillow where it will land." },
  controls: [["Click", "drop the pillow on the counter there (only once)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const TABLE_Y = 452, MW = 44, MH = 52, G = 1100, PW = 106, PH = 24, DROP_T = 0.4;
    const dir = pick([-1, 1]);                                 // +1: the shelf comes from the left and the mug tips right
    const edgeX = dir > 0 ? rand(200, 380) : W - rand(200, 380);
    const shelfY = rand(170, 270);
    const slide = rand(10, 150), tipK = rand(0.85, 1.3), mugCol = pick([P.blue, P.teal, P.orange, P.purple]);
    // Physics runs in "local" coordinates: x measured from the shelf edge along the tipping direction,
    // y in screen pixels. `o` is how far the mug bottom overhangs the edge.
    const m = { state: "slide", o: rand(MW / 2 - 34, MW / 2 - 14), th: 0, w: 0, cx: 0, cy: 0, vx: 0, vy: 0, wait: rand(0.8, 1.5) };
    const COMr = () => [m.o - MW / 2, -MH / 2];               // centre of mass relative to the pivot, unrotated
    const rot = (x, y, a) => [x * Math.cos(a) - y * Math.sin(a), x * Math.sin(a) + y * Math.cos(a)];
    let pillow = null, ghostX = null, landed = null;
    const toScreen = lx => edgeX + dir * lx;

    // Lowest point of the mug body (local frame, while falling).
    function bottomY() {
      let mx = -1e9;
      for (const [x, y] of [[-MW / 2, -MH / 2], [MW / 2, -MH / 2], [-MW / 2, MH / 2], [MW / 2, MH / 2]]) mx = Math.max(mx, m.cy + rot(x, y, m.th)[1]);
      return mx;
    }

    return {
      _dbg: () => ({ m, edgeX, dir, slide, tipK, shelfY }),      // inspected by the test harness only
      pointerMove(x) { ghostX = x; },
      pointerDown(x) {
        if (pillow) return;
        pillow = { x: Math.max(30 + PW / 2, Math.min(W - 30 - PW / 2, x)), t: 0 };
        api.sfx.whoosh();
      },
      update(dt) {
        if (pillow && pillow.t < DROP_T) { pillow.t += dt; if (pillow.t >= DROP_T) api.sfx.click(); }
        if (m.state === "slide") {                               // pause, then slide toward the edge
          if (m.wait > 0) { m.wait -= dt; return; }
          m.o += slide * dt;
          if (m.o >= MW / 2) { m.state = "tip"; m.w = slide / (MH / 2); }
        } else if (m.state === "tip") {                          // rotate about the edge; gravity torque grows with angle
          const [rx, ry] = rot(...COMr(), m.th);
          m.w += tipK * 0.9 * rx * dt * 1.0; m.th += m.w * dt;
          if (m.th > 0.95) {                                     // comes off the edge with the rim's momentum
            const [cx, cy] = rot(...COMr(), m.th);
            m.cx = cx; m.cy = shelfY + cy; m.vx = -m.w * cy + slide * 1.6; m.vy = m.w * cx; m.state = "fall"; m.w *= 0.45;   // tumbles slower once free
            api.sfx.tick();
          }
        } else if (m.state === "fall") {
          m.vy += G * dt; m.cx += m.vx * dt; m.cy += m.vy * dt; m.th += m.w * dt;
          const sx = toScreen(m.cx);
          const onPillow = pillow && pillow.t >= DROP_T && Math.abs(sx - pillow.x) < PW / 2 - 4;
          const surf = onPillow ? TABLE_Y - PH + 6 : TABLE_Y;
          if (bottomY() >= surf) {
            m.state = "done"; landed = { soft: onPillow };
            m.cy -= bottomY() - surf;
            if (onPillow) api.succeed("The mug landed softly on the pillow");
            else api.fail(pillow ? "The mug missed the pillow and hit the counter" : "The mug hit the counter");
          }
        }
      },
      draw(g) {
        const c = g.ctx;
        // wall with faint tiles, counter top and front
        g.clear(P.wall);
        for (let y = 40; y < TABLE_Y; y += 46) g.line(0, y, W, y, "rgba(234,215,220,0.8)", 1);
        g.rr(0, TABLE_Y, W, 16, 0, P.table, P.tableEdge, 2);
        g.rr(0, TABLE_Y + 16, W, 540 - TABLE_Y - 16, 0, "#e6dccd");
        for (let x = 0; x < W; x += 135) g.line(x, TABLE_Y + 22, x, 540, "rgba(168,134,92,0.25)", 2);
        // shelf plank with a bracket, coming from the side wall
        const sx0 = dir > 0 ? 0 : edgeX, sx1 = dir > 0 ? edgeX : W;
        g.rr(sx0, shelfY + 14, sx1 - sx0, 8, 0, "rgba(15,34,51,0.08)");
        g.rr(sx0 - 4, shelfY, sx1 - sx0 + 8, 14, 3, P.wood, P.woodDark, 2);
        const bx = edgeX - dir * 70;
        g.poly([[bx, shelfY + 14], [bx - dir * 50, shelfY + 14], [bx - dir * 50, shelfY + 64]], P.metal, P.metalDark);
        // pillow (falls onto the counter) or ghost
        const drawPillow = (x, y, alpha) => {
          c.save(); c.globalAlpha = alpha;
          g.ellipse(x, TABLE_Y + 2, PW / 2, 5, "rgba(15,34,51,0.12)");
          g.rr(x - PW / 2, y - PH, PW, PH, 11, "#f1e9f7", "#b7a5d0", 2);
          g.line(x - PW / 2 + 14, y - PH / 2, x + PW / 2 - 14, y - PH / 2, "rgba(123,94,167,0.25)", 2, [5, 5]);
          c.restore();
        };
        if (pillow) { const k = Math.min(1, pillow.t / DROP_T); drawPillow(pillow.x, TABLE_Y - (1 - k * k) * 160, 1); }
        else if (ghostX != null) drawPillow(Math.max(30 + PW / 2, Math.min(W - 30 - PW / 2, ghostX)), TABLE_Y, 0.35);
        // mug: body, rim, handle on the trailing side
        c.save();
        if (m.state === "slide" || m.state === "tip") {
          c.translate(edgeX, shelfY); c.scale(dir, 1); c.rotate(m.th); c.translate(m.o - MW / 2, -MH / 2);
        } else { c.translate(toScreen(m.cx), m.cy); c.scale(dir, 1); c.rotate(m.th); }
        c.beginPath(); c.arc(-MW / 2, 0, 13, Math.PI / 2, Math.PI * 1.5); c.lineWidth = 6; c.strokeStyle = Arcade.shade(mugCol, -0.25); c.stroke();
        g.rr(-MW / 2, -MH / 2, MW, MH, 6, mugCol, Arcade.shade(mugCol, -0.3), 2);
        g.rr(-MW / 2 + 5, -MH / 2 + 6, 6, MH - 14, 3, "rgba(255,255,255,0.3)");
        g.line(-MW / 2 + 2, -MH / 2 + 2, MW / 2 - 2, -MH / 2 + 2, Arcade.shade(mugCol, -0.4), 3);
        c.restore();
        if (landed && !landed.soft) g.chip("crash!", toScreen(m.cx), TABLE_Y + 34, P.red);
        if (!pillow) g.chip("Click once on the counter to drop the pillow", W / 2, 510, P.navy);
      }
    };
  }
});
