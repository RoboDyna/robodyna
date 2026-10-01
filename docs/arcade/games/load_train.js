/* Load Train — dynamic avoidance.
 * A toy train (engine + five open wagons: three red, two gray) circles an oval track at a random speed.
 * The robot holds a marble above a fixed drop point on the near side of the track. Space releases it; the
 * marble needs a fall time before it reaches the track, so the release must lead the wagon.
 * Success: the marble lands inside an allowed wagon (never a gray wagon, the engine or the bare track).
 * var1: only the nominated red wagon (star) counts.
 * var2: two matching tunnels (far and near) hide the train; the near tunnel ends just before the drop point,
 *       so the release happens while the wagon is still hidden and must be timed from the train's rhythm.
 * var1+2: nominated wagon + tunnels.
 */
Arcade.register({
  id: "load_train",
  maxTime: 25,
  levels: {
    base: "Any of the three red wagons is allowed.",
    var1: "Only the nominated red wagon (★) is allowed.",
    var2: "Any red wagon; a far and a near tunnel hide the train.",
    "var1+2": "Only the nominated wagon, and tunnels hide the train."
  },
  controls: [["Space", "release the marble (only once)"]],
  keys: [{ key: " ", label: "Drop" }],
  create(api) {
    const { P, W, rand, pick } = api;
    const CX = W / 2, CY = 296, RX = 300, RY = 166, FALL = 0.6, HOVER = 90, MARBLE_R = 9;
    const ENG = 74, WL = 64, GAP = 10, CAR_W = 34;
    // Arc-length table of the oval so the train moves at constant speed.
    const N = 900, pts = [], cum = [0];
    for (let i = 0; i <= N; i++) { const a = (2 * Math.PI * i) / N; pts.push([CX + RX * Math.cos(a), CY + RY * Math.sin(a)]); }
    for (let i = 1; i <= N; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    const L = cum[N];
    const sgn = pick([1, -1]);                     // +1: clockwise on screen
    function at(s) {                               // position + heading for arc-length s (train coordinate)
      s = ((s % L) + L) % L; if (sgn < 0) s = L - s;
      let lo = 0, hi = N;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (cum[m] <= s) lo = m; else hi = m; }
      const f = (s - cum[lo]) / (cum[hi] - cum[lo] || 1), p = pts[lo], q = pts[hi];
      return { x: p[0] + (q[0] - p[0]) * f, y: p[1] + (q[1] - p[1]) * f, a: Math.atan2(q[1] - p[1], q[0] - p[0]) + (sgn < 0 ? Math.PI : 0) };
    }
    const sDrop = sgn > 0 ? L / 4 : L - L / 4;     // bottom of the oval (angle pi/2) in train coordinates
    const DROP = { x: CX, y: CY + RY };

    // Train: engine + 5 wagons; three red (allowed), two gray.
    const colors = ["red", "red", "red", "gray", "gray"].sort(() => api.rng() - 0.5);
    const cars = [{ kind: "engine", off: 0, len: ENG }];
    colors.forEach((c, k) => cars.push({ kind: c, off: ENG / 2 + GAP + WL / 2 + k * (WL + GAP), len: WL }));
    const reds = cars.filter(c => c.kind === "red");
    const target = api.var1 ? pick(reds) : null;
    if (target) target.star = true;
    const v = rand(150, 205);
    // Start so the train first reaches the drop point after a short wait.
    let head = sDrop - v * rand(1.6, 3.2);

    // Tunnels (train coordinate arcs): far one centred on the top, near one ending just upstream of the drop.
    const tunnels = [];
    if (api.var2) {
      const len = rand(250, 310);
      const sTop = sDrop + L / 2;
      tunnels.push({ s0: sTop - len / 2, s1: sTop + len / 2 });
      const end = sDrop - rand(34, 46);
      tunnels.push({ s0: end - len, s1: end });
    }

    let marble = { state: "hold", z: HOVER, vz: 0, car: null, rel: 0 };

    function land() {
      api.sfx.tick();
      for (const c of cars) {
        const d = ((sDrop - (head - c.off)) % L + L * 1.5) % L - L / 2;     // signed arc distance to car centre
        if (Math.abs(d) < c.len / 2 - (c.kind === "engine" ? 0 : 5)) {
          marble.car = c; marble.rel = d; marble.state = "in";
          if (c.kind === "engine") { marble.state = "bounce"; return api.fail("The marble hit the engine"); }
          if (c.kind === "gray") return api.fail("The marble landed in a gray wagon");
          if (target && c !== target) return api.fail("Wrong red wagon: only the ★ wagon is allowed");
          return api.succeed(target ? "Loaded the nominated wagon" : "Loaded a red wagon");
        }
      }
      marble.state = "bounce";
      api.fail("The marble missed the train and hit the track");
    }

    function drawCar(c) {
      const p = at(head - c.off), g = drawCar.g, ctx = g.ctx;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.a);
      g.rr(-c.len / 2 + 3, -CAR_W / 2 + 5, c.len, CAR_W, 6, P.shadow);
      if (c.kind === "engine") {
        g.rr(-c.len / 2, -CAR_W / 2, c.len, CAR_W, 8, P.navy);
        g.rr(-c.len / 2 + 6, -CAR_W / 2 + 6, c.len * 0.45, CAR_W - 12, 4, "#1d4468");
        g.circle(c.len / 2 - 16, 0, 8, P.black, "#555", 2);
        g.rr(c.len / 2 - 5, -CAR_W / 2 + 4, 6, CAR_W - 8, 3, P.yellow);
      } else {
        const col = c.kind === "red" ? P.red : P.metal, dark = c.kind === "red" ? "#a8332a" : P.metalDark;
        g.rr(-c.len / 2, -CAR_W / 2, c.len, CAR_W, 6, col, c.star ? P.yellow : dark, c.star ? 4 : 2);
        g.rr(-c.len / 2 + 6, -CAR_W / 2 + 6, c.len - 12, CAR_W - 12, 3, Arcade.shade(col, -0.3));
      }
      ctx.restore();
      if (c.star) g.text("★", p.x, p.y + 1, { size: 18, color: P.yellow, align: "center" });
      if (marble.car === c && marble.state === "in") g.ball(p.x + Math.cos(p.a) * -marble.rel * 0.3, p.y + Math.sin(p.a) * -marble.rel * 0.3, MARBLE_R, P.blue);
    }

    return {
      keyDown(k) {
        if (k !== " " || marble.state !== "hold") return;
        marble.state = "fall"; marble.vz = 0; api.sfx.whoosh();
      },
      update(dt) {
        head += v * dt;
        if (marble.state === "fall") {
          marble.vz -= (2 * HOVER / (FALL * FALL)) * dt; marble.z += marble.vz * dt;
          if (marble.z <= 0) { marble.z = 0; land(); }
        } else if (marble.state === "bounce") {
          marble.vz = marble.vz < 0 ? -marble.vz * 0.4 : marble.vz - 2 * HOVER / (FALL * FALL) * dt;
          marble.z = Math.max(0, marble.z + marble.vz * dt);
        }
      },
      draw(g) {
        const ctx = g.ctx;
        g.scene();
        // grass infield + track (sleepers and rails)
        g.ellipse(CX, CY, RX - 34, RY - 34, "#eaf3e6", "#d3e4cc", 2);
        g.ellipse(CX, CY, RX + 22, RY + 22, null, "#e5d9c8", 30);
        for (let i = 0; i < 90; i++) {
          const p = at((L * i) / 90), nx = -Math.sin(p.a), ny = Math.cos(p.a);
          g.line(p.x - nx * 20, p.y - ny * 20, p.x + nx * 20, p.y + ny * 20, P.woodDark, 4);
        }
        g.ellipse(CX, CY, RX - 12, RY - 12, null, P.metalDark, 2.5);
        g.ellipse(CX, CY, RX + 12, RY + 12, null, P.metalDark, 2.5);
        // robot standing in the infield, reaching toward the near rail
        g.circle(CX, CY + 20, 30, P.navy); g.circle(CX, CY + 20, 14, "#1d4468");
        // drop marker on the track
        g.circle(DROP.x, DROP.y, 14, null, "rgba(47,109,181,0.55)", 2);
        drawCar.g = g;
        for (let i = cars.length - 1; i >= 0; i--) drawCar(cars[i]);
        // tunnels: thick covered arcs drawn over the train
        for (const t of tunnels) {
          ctx.save(); ctx.beginPath();
          for (let s = t.s0; s <= t.s1; s += 6) { const p = at(s); s === t.s0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); }
          const pe = at(t.s1); ctx.lineTo(pe.x, pe.y);
          ctx.lineCap = "butt"; ctx.strokeStyle = "rgba(15,34,51,0.16)"; ctx.lineWidth = 64; ctx.translate(4, 6); ctx.stroke();
          ctx.translate(-4, -6); ctx.strokeStyle = "#8f7a63"; ctx.lineWidth = 60; ctx.stroke();
          ctx.strokeStyle = "#a99079"; ctx.lineWidth = 44; ctx.stroke();
          ctx.restore();
          for (const s of [t.s0, t.s1]) {
            const p = at(s), nx = -Math.sin(p.a), ny = Math.cos(p.a);
            g.line(p.x - nx * 30, p.y - ny * 30, p.x + nx * 30, p.y + ny * 30, "#5e4c3b", 6);
          }
        }
        // marble: held by the robot above the drop point, shadow on the track below
        const mz = marble.state === "in" ? null : marble.z;
        if (mz !== null) {
          g.shadow(DROP.x, DROP.y + 2, MARBLE_R, mz);
          if (marble.state === "hold") {
            g.line(DROP.x, CY + 20, DROP.x, DROP.y - HOVER * 0.55 - 12, P.navy, 10);
            g.circle(DROP.x, DROP.y - HOVER * 0.55, 15, null, P.navy, 5);
          }
          g.ball(DROP.x, DROP.y - mz * 0.55, MARBLE_R * (1 + mz / 300), P.blue);
        }
        if (marble.state === "hold") g.chip("Press Space to drop the marble", 150, 516, P.navy);
      },
      hud() { return [{ k: "Allowed", v: target ? "★ wagon" : "red wagons" }]; }
    };
  }
});
