/* Hit Target — dynamic avoidance.
 * A round target (red/white rings, yellow centre) glides back and forth along a curved track.
 * Click once: the robot throws a dart from its base (bottom centre) straight at the clicked point. The dart
 * needs flight time, so the click must lead the target. Success: the dart sticks in the yellow centre.
 * Blockers stand between the robot and the track; a dart whose flight line meets a blocker fails.
 * var1: a static green blocker.  var2: a red blocker sliding back and forth.  var1+2: both.
 */
Arcade.register({
  id: "hit_target",
  maxTime: 14,
  levels: {
    base: "A moving target, no blockers.",
    var1: "A static green blocker stands in front of part of the track.",
    var2: "A red blocker slides back and forth in front of the track.",
    "var1+2": "Both the static green and the sliding red blocker."
  },
  controls: [["Click", "throw the dart at that point (only once)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const BASE = { x: W / 2, y: 500 }, DART_V = 680, CENTER_R = 22, RINGS = [52, 40, 30];
    // Target path: x = cx + Ax sin(th), y = cy - Ay cos(2 th) -> a shallow arc; both velocities vanish at
    // the two ends, which is where a careful thrower aims.
    const cx = rand(330, 480), cy = rand(185, 215), Ax = rand(150, 200), Ay = api.var1 || api.var2 ? rand(18, 34) : rand(10, 26);
    const om = (2 * Math.PI) / rand(4.4, 6.0), ph = rand(0, 6.28), sgn = pick([1, -1]);
    const tgt = t => { const th = sgn * om * t + ph; return { x: cx + Ax * Math.sin(th), y: cy - Ay * Math.cos(2 * th) }; };

    const blockers = [];
    if (api.var1) {                               // static wall covering one end of the arc
      const side = pick([-1, 1]);
      blockers.push({ color: P.green, dark: "#2c7a42", x0: W / 2 + side * rand(40, 90), y: rand(330, 350), w: rand(80, 105), h: 22, A: 0 });
    }
    if (api.var2) {                               // sliding wall
      blockers.push({ color: P.red, dark: "#a8332a", x0: W / 2 + rand(-40, 40), y: api.var1 ? rand(400, 412) : rand(360, 400), w: rand(70, 90), h: 22,
        A: rand(150, 210), om: (2 * Math.PI) / rand(2.2, 3.2), ph: rand(0, 6.28) });
    }
    const bx = b => b.x0 + (b.A ? b.A * Math.sin(b.om * api.time + b.ph) : 0);

    let dart = null, ghost = null, stuck = null;

    function finish() {
      const p = tgt(api.time), d = Math.hypot(dart.x - p.x, dart.y - p.y);
      dart.state = "done"; api.sfx.tick();
      if (d <= CENTER_R) { stuck = { dx: dart.x - p.x, dy: dart.y - p.y }; return api.succeed("Bull's-eye: the dart is in the yellow centre"); }
      if (d <= RINGS[0]) { stuck = { dx: dart.x - p.x, dy: dart.y - p.y }; return api.fail("Hit the rings but missed the yellow centre"); }
      api.fail("The dart missed the target");
    }

    return {
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (dart || y > BASE.y - 40) return;
        const d = Math.hypot(x - BASE.x, y - BASE.y);
        dart = { x: BASE.x, y: BASE.y, sx: BASE.x, sy: BASE.y, tx: x, ty: y, d, s: 0, state: "fly", ang: Math.atan2(y - BASE.y, x - BASE.x) };
        api.sfx.whoosh();
      },
      update(dt) {
        if (!dart || dart.state !== "fly") return;
        dart.s = Math.min(dart.d, dart.s + DART_V * dt);
        const k = dart.s / dart.d;
        dart.x = dart.sx + (dart.tx - dart.sx) * k; dart.y = dart.sy + (dart.ty - dart.sy) * k;
        dart.z = Math.sin(Math.PI * k) * dart.d * 0.12;
        for (const b of blockers) {
          if (Math.abs(dart.x - bx(b)) < b.w / 2 + 3 && Math.abs(dart.y - b.y) < b.h / 2 + 3) {
            dart.state = "blocked";
            return api.fail(`The dart hit the ${b.color === P.green ? "green" : "red"} blocker`);
          }
        }
        if (dart.s >= dart.d) finish();
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // target track (rail) drawn as a faint arc
        c.save(); c.beginPath();
        for (let i = 0; i <= 60; i++) {
          const th = -Math.PI / 2 + (Math.PI * i) / 60, x = cx + Ax * Math.sin(th), y = cy - Ay * Math.cos(2 * th);
          i ? c.lineTo(x, y) : c.moveTo(x, y);
        }
        c.strokeStyle = P.line; c.lineWidth = 10; c.lineCap = "round"; c.stroke(); c.restore();
        // target
        const p = tgt(api.time);
        g.shadow(p.x + 4, p.y + 8, RINGS[0], 0);
        g.circle(p.x, p.y, RINGS[0], P.red, "#a8332a", 2);
        g.circle(p.x, p.y, RINGS[1], P.white);
        g.circle(p.x, p.y, RINGS[2], P.red);
        g.circle(p.x, p.y, CENTER_R, P.yellow, "#c99c12", 2);
        // blockers (tall walls, top-down)
        for (const b of blockers) {
          const x = bx(b);
          g.rr(x - b.w / 2 + 4, b.y - b.h / 2 + 6, b.w, b.h, 5, P.shadow);
          g.rr(x - b.w / 2, b.y - b.h / 2, b.w, b.h, 5, b.color, b.dark, 2);
          if (b.A) {
            g.line(b.x0 - b.A - b.w / 2, b.y + 18, b.x0 + b.A + b.w / 2, b.y + 18, "rgba(224,75,60,0.18)", 3, [6, 6]);
          }
        }
        // robot base
        g.circle(BASE.x, BASE.y + 30, 42, P.navy);
        g.circle(BASE.x, BASE.y + 4, 14, "#1d4468");
        // dart
        const drawDart = (x, y, ang, z) => {
          const L = 26, ux = Math.cos(ang), uy = Math.sin(ang);
          if (z > 0) g.shadow(x, y + 6, 5, z);
          const yy = y - (z || 0) * 0.5;
          g.line(x - ux * L, yy - uy * L, x, yy, P.ink, 3);
          g.poly([[x - ux * L, yy - uy * L], [x - ux * (L + 8) - uy * 6, yy - uy * (L + 8) + ux * 6], [x - ux * (L + 8) + uy * 6, yy - uy * (L + 8) - ux * 6]], P.orange);
          g.circle(x, yy, 2.5, P.metalDark);
        };
        if (!dart) {
          drawDart(BASE.x, BASE.y - 6, -Math.PI / 2, 0);
          if (ghost && ghost.y < BASE.y - 40) {
            g.circle(ghost.x, ghost.y, 10, null, "rgba(11,42,69,0.5)", 2);
            g.line(ghost.x - 15, ghost.y, ghost.x + 15, ghost.y, "rgba(11,42,69,0.5)", 1.5);
            g.line(ghost.x, ghost.y - 15, ghost.x, ghost.y + 15, "rgba(11,42,69,0.5)", 1.5);
          }
          g.chip("Click once to throw the dart — it needs time to fly", 210, 512, P.navy);
        } else if (stuck) {
          drawDart(p.x + stuck.dx, p.y + stuck.dy, dart.ang, 0);
        } else {
          drawDart(dart.x, dart.y, dart.ang, dart.state === "fly" ? dart.z : 0);
        }
      }
    };
  }
});
