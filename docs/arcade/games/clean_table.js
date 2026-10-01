/* Clean Table — household, dynamic avoidance.
 * A tipped mug leaks coffee onto a table that tilts slightly toward a laptop. The spill is simulated on a
 * coarse height grid: coffee keeps pouring from the mug for several seconds and creeps downhill once a cell
 * is wet enough (thin films stick). Each click sends the sponge to that spot (finite travel speed); when it
 * touches down it soaks up a round patch. Success: the mug runs dry and the spill is wiped up or has stopped
 * short of the laptop. Failure: coffee touches the laptop.
 */
Arcade.register({
  id: "clean_table",
  maxTime: 24,
  levels: { base: "A tipped mug leaks coffee that creeps toward the laptop." },
  controls: [["Click", "wipe that spot with the sponge (repeatable)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const X0 = 30, Y0 = 70, CS = 10, NX = 75, NY = 45;
    const H_MIN = 0.24, K = 13, WIPE_R = 44, SPONGE_V = 900, CONTACT_T = 0.16;
    const idx = (i, j) => j * NX + i;
    const h = new Float32Array(NX * NY), bed = new Float32Array(NX * NY), flow = new Float32Array(NX * NY);

    // Layout: mug on one side, laptop on the other; the table slopes from the mug toward the laptop.
    const side = pick([-1, 1]);
    const mug = { x: W / 2 - side * rand(170, 230), y: rand(220, 360) };
    const lap = { w: 170, h: 112 };
    lap.x = W / 2 + side * rand(140, 190) - lap.w / 2; lap.y = rand(180, 330) - lap.h / 2;
    const lc = { x: lap.x + lap.w / 2, y: lap.y + lap.h / 2 };
    const sAng = Math.atan2(lc.y - mug.y, lc.x - mug.x) + rand(-0.15, 0.15), SLOPE = rand(0.08, 0.095);
    for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const x = X0 + (i + 0.5) * CS, y = Y0 + (j + 0.5) * CS;
      bed[idx(i, j)] = -SLOPE * ((x - mug.x) * Math.cos(sAng) + (y - mug.y) * Math.sin(sAng)) / CS + rand(-0.035, 0.035);
    }
    const spout = { x: mug.x + Math.cos(sAng) * 34, y: mug.y + Math.sin(sAng) * 34 };
    const si = Math.floor((spout.x - X0) / CS), sj = Math.floor((spout.y - Y0) / CS);
    const POUR_T = rand(7.5, 9.0), POUR_Q = rand(26, 31);     // seconds of leaking, volume per second
    let poured = 0, sponge = { x: W - 70, y: 485, tx: W - 70, ty: 485, state: "idle", t: 0 }, queue = [], acted = false;

    const off = document.createElement("canvas"); off.width = NX; off.height = NY;
    const img = off.getContext("2d").createImageData(NX, NY);

    function step(dt) {
      if (api.time < POUR_T) {                                    // the mug keeps leaking
        const q = POUR_Q * dt; poured += q;
        for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) h[idx(si + di, sj + dj)] += q / 9;
      }
      flow.fill(0);
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        const a = idx(i, j), ha = h[a];
        if (ha <= H_MIN) continue;                                // thin film sticks
        const head = ha + bed[a], outs = [];
        let tot = 0;
        if (i > 0) outs.push(a - 1); if (i < NX - 1) outs.push(a + 1);
        if (j > 0) outs.push(a - NX); if (j < NY - 1) outs.push(a + NX);
        const amt = [];
        for (const b of outs) { const d = head - (h[b] + bed[b]); const f = d > 0 ? d : 0; amt.push(f); tot += f; }
        if (tot <= 0) continue;
        const scale = Math.min(K * dt, ((ha - H_MIN) * 0.9) / tot);
        outs.forEach((b, k) => { const f = amt[k] * scale; flow[b] += f; flow[a] -= f; });
      }
      for (let n = 0; n < h.length; n++) h[n] += flow[n];
    }
    function wipe(x, y) {
      const r = WIPE_R / CS;
      const ci = (x - X0) / CS - 0.5, cj = (y - Y0) / CS - 0.5;
      for (let j = Math.max(0, Math.floor(cj - r)); j <= Math.min(NY - 1, Math.ceil(cj + r)); j++)
        for (let i = Math.max(0, Math.floor(ci - r)); i <= Math.min(NX - 1, Math.ceil(ci + r)); i++)
          if ((i - ci) ** 2 + (j - cj) ** 2 <= r * r) h[idx(i, j)] = 0;
    }
    function check() {
      let total = 0, moving = false;
      for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
        const v = h[idx(i, j)]; if (v < 0.04) continue;
        total += v; if (v > H_MIN + 0.03) moving = true;
        const x = X0 + (i + 0.5) * CS, y = Y0 + (j + 0.5) * CS;
        if (x > lap.x - 4 && x < lap.x + lap.w + 4 && y > lap.y - 4 && y < lap.y + lap.h + 4) return api.fail("Coffee reached the laptop");
      }
      if (api.time < POUR_T) return;
      if (total < 0.04 * poured) return api.succeed("Spill wiped up before it reached the laptop");
      if (!moving && api.time > POUR_T + 1) api.succeed("The mug ran dry and the spill stopped short of the laptop");
    }

    return {
      _test: { mug, lap, spout, get sponge() { return sponge; } },                                // positions, for the automated tests
      pointerDown(x, y) {
        if (x < X0 || x > X0 + NX * CS || y < Y0 || y > Y0 + NY * CS) return;
        acted = true; api.sfx.click();
        if (sponge.state === "wipe") queue = [{ x, y }];                // finish this wipe first
        else { sponge.state = "move"; sponge.tx = x; sponge.ty = y; }
      },
      update(dt) {
        const n = Math.max(1, Math.ceil(dt / (1 / 60)));
        for (let k = 0; k < n; k++) step(dt / n);
        if (sponge.state === "move") {
          const dx = sponge.tx - sponge.x, dy = sponge.ty - sponge.y, d = Math.hypot(dx, dy), s = SPONGE_V * dt;
          if (d <= s) { sponge.x = sponge.tx; sponge.y = sponge.ty; sponge.state = "wipe"; sponge.t = 0; api.sfx.whoosh(); }
          else { sponge.x += (dx / d) * s; sponge.y += (dy / d) * s; }
        } else if (sponge.state === "wipe") {
          sponge.t += dt; wipe(sponge.x, sponge.y);
          if (sponge.t >= CONTACT_T) {
            if (queue.length) { const p = queue.shift(); sponge.tx = p.x; sponge.ty = p.y; sponge.state = "move"; }
            else sponge.state = "idle";
          }
        }
        if (!api.ended) check();
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // laptop
        g.rr(lap.x + 5, lap.y + 7, lap.w, lap.h, 10, P.shadow);
        g.rr(lap.x, lap.y, lap.w, lap.h, 10, "#c9d2db", P.metalDark, 2);
        g.rr(lap.x + 12, lap.y + 10, lap.w - 24, lap.h * 0.55, 4, "#aeb9c4");
        for (let r = 0; r < 3; r++) for (let k = 0; k < 9; k++) g.rr(lap.x + 16 + k * 15.5, lap.y + 16 + r * 18, 12, 12, 2, "#e8edf2");
        g.rr(lap.x + lap.w / 2 - 26, lap.y + lap.h - 36, 52, 26, 4, "#b6c1cc");
        // coffee: the height grid is painted into a tiny offscreen image and scaled up smoothly
        const px = img.data;
        for (let n = 0; n < h.length; n++) {
          const v = h[n], a = v < 0.03 ? 0 : Math.min(0.92, 0.45 + v * 0.7);
          px[4 * n] = 111 - Math.min(40, v * 30); px[4 * n + 1] = 66 - Math.min(26, v * 18); px[4 * n + 2] = 32; px[4 * n + 3] = a * 255;
        }
        off.getContext("2d").putImageData(img, 0, 0);
        c.save(); c.imageSmoothingEnabled = true; c.filter = "blur(3px)"; c.drawImage(off, X0, Y0, NX * CS, NY * CS); c.restore();
        // tipped mug lying on its side, mouth toward the slope
        c.save(); c.translate(mug.x, mug.y); c.rotate(sAng);
        g.rr(-30, -18, 60, 40, 8, P.shadow);
        g.rr(-34, -22, 62, 44, 8, P.white, "#9fb1c2", 2.5);
        g.ellipse(28, 0, 8, 22, "#5a361a", "#9fb1c2", 2.5);
        g.rr(-48, -9, 16, 18, 6, null, "#9fb1c2", 4);
        c.restore();
        // sponge
        const lift = sponge.state === "move" ? 30 : sponge.state === "wipe" ? 0 : 20;
        if (sponge.state === "wipe") g.circle(sponge.x, sponge.y, WIPE_R, "rgba(63,165,91,0.12)", "rgba(63,165,91,0.5)", 2);
        g.shadow(sponge.x + 3, sponge.y + 6, 28, lift);
        g.rr(sponge.x - 30, sponge.y - 20 - lift * 0.4, 60, 40, 8, "#f2d35b", "#c99c12", 2);
        g.rr(sponge.x - 30, sponge.y - 4 - lift * 0.4, 60, 24, 6, P.green);
        if (!acted) g.chip("Click to wipe — stop the coffee before it reaches the laptop", W / 2, 510, P.navy);
      },
      hud() { return [{ k: "Leaking", v: api.time < POUR_T ? "yes" : "no", warn: api.time < POUR_T }]; }
    };
  }
});
