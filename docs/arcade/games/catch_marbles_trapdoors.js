/* Catch Marbles Trapdoors — periodic pattern.
 * Top-down tray: a long channel with four colored trapdoors (red, green, blue, yellow, shuffled).
 * Marbles roll back and forth along the channel, bouncing off the end bumpers with slight speed
 * changes, so their motion is periodic but not exactly regular. Keys 1–4 (or the colored buttons)
 * open the matching door for a moment; any marble over an open door falls into the box below.
 * Success: the target marble falls through the door of ITS OWN color.
 * Fail: the target falls through another door, the distractor falls through any door, or the
 * target door has no opens left.
 * base: 3 opens per door, one marble.        var1: each door opens only once.
 * var2: 3 opens per door + a distractor marble rolling in the other lane at another speed.
 * var1+2: one-use doors + distractor.
 */
Arcade.register({
  id: "catch_marbles_trapdoors",
  maxTime: 20,
  levels: {
    base: "Each door opens 3 times. Only the target marble rolls.",
    var1: "Each door opens only once. One shot.",
    var2: "Doors open 3 times, but a distractor marble rolls too. It must not fall.",
    "var1+2": "One-use doors and a distractor marble."
  },
  controls: [["1–4", "open that trapdoor briefly"], ["Click", "the colored buttons do the same"]],
  keys: [{ key: "1", label: "1" }, { key: "2", label: "2" }, { key: "3", label: "3" }, { key: "4", label: "4" }],
  create(api) {
    const { P, W, rand, pick } = api;
    const X0 = 110, X1 = 700, LANES = [246, 280], R = 11, DOOR_W = 60, OPEN_T = 0.32, FALL_T = 0.35;
    const COLORS = [["Red", P.red], ["Green", P.green], ["Blue", P.blue], ["Yellow", P.yellow]];
    // Shuffle door colors along the channel.
    const order = [0, 1, 2, 3];
    for (let i = 3; i > 0; i--) { const j = Math.floor(api.rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    const doors = order.map((c, i) => ({
      x: X0 + (X1 - X0) * (i + 0.5) / 4, name: COLORS[c][0], color: COLORS[c][1], left: api.var1 ? 1 : 3, open: 0
    }));
    const tgtDoor = pick(doors);

    function makeMarble(color, name, lane, speed) {
      return { color, name, y: lane, x: rand(X0 + 40, X1 - 40), dir: pick([-1, 1]), v: speed,
        ph: rand(0, 6.28), state: "roll", fallT: 0, door: null };
    }
    const laneT = pick([0, 1]);
    const target = makeMarble(tgtDoor.color, tgtDoor.name, LANES[laneT], rand(135, 195));
    const marbles = [target];
    let distractor = null;
    if (api.var2) {
      const other = pick(doors.filter(d => d !== tgtDoor));
      // distractor speed clearly differs from the target's, so the two drift in and out of phase
      const v = target.v * pick([rand(0.62, 0.8), rand(1.2, 1.38)]);
      distractor = makeMarble(other.color, other.name, LANES[1 - laneT], v);
      marbles.push(distractor);
    }
    let acted = false, pending = null; // pending: success after the fall animation

    const BTN_Y = 420, BTN_W = 120, BTN_H = 54;
    const btnX = i => W / 2 + (i - 1.5) * (BTN_W + 26);

    function openDoor(i) {
      const d = doors[i];
      if (!d || d.left <= 0 || d.open > 0) { api.sfx.tick(); return; }
      d.left -= 1; d.open = OPEN_T; acted = true; api.sfx.whoosh();
    }

    return {
      keyDown(k) { if (["1", "2", "3", "4"].includes(k)) openDoor(+k - 1); },
      pointerDown(x, y) {
        for (let i = 0; i < 4; i++) {
          if (Math.abs(x - btnX(i)) < BTN_W / 2 && Math.abs(y - BTN_Y) < BTN_H / 2) return openDoor(i);
          if (Math.abs(x - doors[i].x) < DOOR_W / 2 && y > 225 && y < 300) return openDoor(i);
        }
      },
      update(dt) {
        for (const d of doors) d.open = Math.max(0, d.open - dt);
        for (const m of marbles) {
          if (m.state === "roll") {
            m.ph += dt * 2.1;
            m.x += m.dir * m.v * (1 + 0.08 * Math.sin(m.ph)) * dt;
            if (m.x < X0 + R) { m.x = X0 + R; m.dir = 1; m.v *= rand(0.95, 1.05); api.sfx.tick(); }
            if (m.x > X1 - R) { m.x = X1 - R; m.dir = -1; m.v *= rand(0.95, 1.05); api.sfx.tick(); }
            m.v = Math.max(130, Math.min(320, m.v));
            for (const d of doors) {
              if (d.open > 0 && Math.abs(m.x - d.x) < DOOR_W / 2 - R * 0.4) {
                m.state = "fall"; m.door = d; m.fallT = 0;
                if (m === distractor) return api.fail(`The ${m.name.toLowerCase()} distractor fell through the ${d.name.toLowerCase()} door`);
                if (d !== tgtDoor) return api.fail(`The target fell through the ${d.name.toLowerCase()} door`);
                pending = true;
              }
            }
          } else if (m.state === "fall") {
            m.fallT += dt; m.x += (m.door.x - m.x) * Math.min(1, dt * 8);
            if (m.fallT >= FALL_T) m.state = "gone";
          }
        }
        if (pending && target.state === "gone") return api.succeed(`Trapped the ${tgtDoor.name.toLowerCase()} marble`);
        if (!pending && tgtDoor.left <= 0 && tgtDoor.open <= 0) return api.fail(`No opens left on the ${tgtDoor.name.toLowerCase()} door`);
      },
      draw(g) {
        g.scene();
        const c = g.ctx;
        // box under the tray (slightly visible around it) and the tray itself
        g.rr(84, 196, W - 168, 148, 16, P.shadow);
        g.rr(80, 186, W - 160, 148, 16, P.woodDark);
        g.rr(88, 194, W - 176, 132, 12, P.wood);
        g.rr(X0 - 6, 222, X1 - X0 + 12, 82, 10, "#e9d8bf", P.woodDark, 2);   // channel floor
        g.line(X0, 263, X1, 263, "rgba(120,90,50,0.25)", 1.5, [6, 6]);       // lane divider
        // end bumpers
        g.rr(X0 - 12, 222, 10, 82, 4, P.metal, P.metalDark);
        g.rr(X1 + 2, 222, 10, 82, 4, P.metal, P.metalDark);
        // trapdoors: colored frame; when open, a dark hole with the two flaps folded aside
        for (const d of doors) {
          const k = d.open > 0 ? Math.sin(Math.min(1, (OPEN_T - d.open) / 0.08, d.open / 0.08) * Math.PI / 2) : 0;
          g.rr(d.x - DOOR_W / 2 - 4, 218, DOOR_W + 8, 90, 6, d.color);
          g.rr(d.x - DOOR_W / 2, 224, DOOR_W, 78, 4, "#1d2530");
          const fw = DOOR_W / 2 * (1 - k);
          if (fw > 0.5) {
            g.rr(d.x - DOOR_W / 2, 224, fw, 78, 3, Arcade.shade(d.color, 0.55), Arcade.shade(d.color, -0.2), 1);
            g.rr(d.x + DOOR_W / 2 - fw, 224, fw, 78, 3, Arcade.shade(d.color, 0.55), Arcade.shade(d.color, -0.2), 1);
          }
          g.line(d.x, 226, d.x, 300, `rgba(0,0,0,${0.18 * (1 - k)})`, 1);
        }
        // marbles
        for (const m of marbles) {
          if (m.state === "gone") continue;
          if (m.state === "fall") {
            const f = m.fallT / FALL_T;
            c.save(); c.globalAlpha = 1 - f * 0.8; g.ball(m.x, m.y, R * (1 - 0.55 * f), m.color); c.restore();
            continue;
          }
          g.shadow(m.x + 2, m.y + 4, R * 0.9, 0);
          g.ball(m.x, m.y, R, m.color);
          if (m === target && marbles.length > 1) g.circle(m.x, m.y, R + 4, null, "rgba(15,34,51,0.35)", 1.5);
        }
        // target card
        g.rr(40, 108, 190, 52, 12, "#fff", P.line, 2);
        g.text("TARGET", 58, 124, { size: 11, weight: 800, color: P.muted });
        g.ball(66, 143, 9, tgtDoor.color);
        g.text(tgtDoor.name + " marble", 84, 143, { size: 15, weight: 800 });
        if (distractor) {
          g.rr(W - 230, 108, 190, 52, 12, "#fff", P.line, 2);
          g.text("DISTRACTOR", W - 212, 124, { size: 11, weight: 800, color: P.muted });
          g.ball(W - 204, 143, 9, distractor.color);
          g.text("keep it on the tray", W - 186, 143, { size: 13, weight: 700, color: P.muted });
        }
        // buttons with opens left
        for (let i = 0; i < 4; i++) {
          const d = doors[i], x = btnX(i), on = d.left > 0;
          c.save(); c.globalAlpha = on ? 1 : 0.35;
          g.rr(x - BTN_W / 2, BTN_Y - BTN_H / 2 + 4, BTN_W, BTN_H, 14, Arcade.shade(d.color, -0.3));
          g.rr(x - BTN_W / 2, BTN_Y - BTN_H / 2 + (d.open > 0 ? 3 : 0), BTN_W, BTN_H, 14, d.color);
          g.text(String(i + 1), x - 32, BTN_Y + (d.open > 0 ? 3 : 0), { size: 22, weight: 800, color: "#fff", align: "center" });
          for (let j = 0; j < (api.var1 ? 1 : 3); j++) g.circle(x + 6 + j * 14, BTN_Y + (d.open > 0 ? 3 : 0), 5, j < d.left ? "#fff" : "rgba(255,255,255,0.3)");
          c.restore();
        }
        if (!acted) g.chip("Press 1–4 (or a button) when the target marble is over its door", W / 2, 505, P.navy);
      },
      hud() { return [{ k: "Target", v: tgtDoor.name }, { k: "Opens", v: String(tgtDoor.left), warn: tgtDoor.left <= 1 }]; }
    };
  }
});
