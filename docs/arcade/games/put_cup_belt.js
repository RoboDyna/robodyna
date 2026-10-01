/* Put Cup Belt — dynamic avoidance.
 * A conveyor belt carries pairs of yellow tools; the gap between the two tools of a pair is the slot.
 * Click once on the belt: the robot carries the cup from its fixed start (bottom centre) straight to the
 * click point at finite speed and sets it down there. The slot keeps moving during the carry, so the click
 * must lead the slot. Success: the cup lands between the tools without touching them.
 * Blue curtains hang over the belt; the carry path must not cross a curtain (and the cup cannot be set
 * down under one).
 * var1: two static blue curtains.  var2: swaying blue curtains.  var1+2: a static and a swaying curtain.
 */
Arcade.register({
  id: "put_cup_belt",
  maxTime: 18,
  levels: {
    base: "No curtains: lead the moving slot.",
    var1: "Static blue curtains hang over the belt. Do not carry the cup through them.",
    var2: "Swaying blue curtains hang over the belt.",
    "var1+2": "A mix of static and swaying curtains."
  },
  controls: [["Click", "place the cup on the belt (only once)"]],
  create(api) {
    const { P, W, rand, pick } = api;
    const BX0 = 40, BX1 = W - 40, BY0 = 190, BY1 = 320, BCY = (BY0 + BY1) / 2;
    const CUP_R = 23, TOOL_W = 30, TOOL_H = 84, GAP = 2 * CUP_R + 40;   // slot = inner gap between tools
    const CARRY_V = 360, LOWER_T = 0.18, CUR_PAD = 16;
    const START = { x: W / 2, y: 488 };
    const dir = pick([1, -1]);                     // belt direction (+1: left -> right)
    const v = rand(62, 92);                        // belt speed, px/s
    // Pairs of tools along the belt; p = distance travelled from the upstream belt end.
    const spacing = rand(330, 410), p0 = rand(120, 230);
    const pairs = [0, 1, 2].map(i => ({ p: p0 - i * spacing }));
    const xOf = p => (dir > 0 ? BX0 + p : BX1 - p);

    // Curtains: centres drawn from three belt sections so clear gaps always remain.
    const curtains = [];
    const sections = [190, 405, 620].sort(() => api.rng() - 0.5);
    const addCurtain = (sway, i) => curtains.push({
      x0: sections[i] + rand(-35, 35), w: rand(64, 92), sway,
      A: sway ? rand(45, 75) : 0, om: rand(1.9, 3.0), ph: rand(0, 6.28)
    });
    if (api.var1 && !api.var2) { addCurtain(false, 0); addCurtain(false, 1); }
    if (api.var2 && !api.var1) { addCurtain(true, 0); if (api.chance(0.5)) addCurtain(true, 1); }
    if (api.var1 && api.var2) { addCurtain(false, 0); addCurtain(true, 1); }
    const curX = c => c.x0 + c.A * Math.sin(c.om * api.time + c.ph);

    let cup = null, ghost = null, beltPhase = 0;

    function hitsCurtain(x, y) {
      if (y < BY0 - CUR_PAD - CUP_R || y > BY1 + CUR_PAD + CUP_R) return null;
      for (const c of curtains) {
        const cx = curX(c);
        if (x + CUP_R * 0.8 > cx - c.w / 2 && x - CUP_R * 0.8 < cx + c.w / 2) return c;
      }
      return null;
    }
    function setDown() {
      cup.state = "down";
      api.sfx.tick();
      for (const pr of pairs) {
        const sx = xOf(pr.p), dx = cup.x - sx, dy = Math.abs(cup.y - BCY);
        const inner = GAP / 2, outer = GAP / 2 + TOOL_W;
        // overlap with either tool block (disc vs rect, approximately)
        const overTool = Math.abs(dx) + CUP_R > inner + 2 && Math.abs(dx) - CUP_R < outer && dy - CUP_R < TOOL_H / 2;
        if (overTool) return api.fail("The cup landed on a yellow tool");
        if (Math.abs(dx) < inner - CUP_R + 6 && dy < TOOL_H / 2 - 8) { cup.pair = pr; return api.succeed("Cup set in the slot"); }
      }
      api.fail("The cup missed the slot");
    }

    return {
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (cup || x < BX0 + CUP_R || x > BX1 - CUP_R || y < BY0 + CUP_R || y > BY1 - CUP_R) return;
        const d = Math.hypot(x - START.x, y - START.y);
        cup = { x: START.x, y: START.y, tx: x, ty: y, sx: START.x, sy: START.y, d, s: 0, state: "carry", t: 0 };
        api.sfx.whoosh();
      },
      update(dt) {
        beltPhase = (beltPhase + v * dt) % 40;
        for (const pr of pairs) pr.p += v * dt;
        if (!cup) return;
        if (cup.state === "carry") {
          cup.s = Math.min(cup.d, cup.s + CARRY_V * dt);
          const k = cup.s / cup.d;
          cup.x = cup.sx + (cup.tx - cup.sx) * k; cup.y = cup.sy + (cup.ty - cup.sy) * k;
          if (cup.s >= cup.d) { cup.state = "lower"; cup.t = 0; }
        } else if (cup.state === "lower") {
          cup.t += dt;
          if (cup.t >= LOWER_T) setDown();
        } else if (cup.state === "down") {
          cup.x += dir * v * dt;                              // rides the belt
        }
        if (!api.ended && cup.state !== "down" && hitsCurtain(cup.x, cup.y)) {
          cup.state = "crash"; api.fail("The cup hit a blue curtain");
        }
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // belt frame + moving cleats
        g.rr(BX0 - 8, BY0 - 10, BX1 - BX0 + 16, BY1 - BY0 + 20, 10, P.metal, P.metalDark, 2);
        g.rr(BX0, BY0, BX1 - BX0, BY1 - BY0, 6, "#4a525c");
        c.save(); c.beginPath(); c.rect(BX0, BY0, BX1 - BX0, BY1 - BY0); c.clip();
        for (let x = BX0 - 40 + (dir > 0 ? beltPhase : 40 - beltPhase); x < BX1 + 40; x += 40) g.line(x, BY0, x, BY1, "rgba(255,255,255,0.08)", 3);
        c.restore();
        // direction arrows on the frame
        for (let x = 110; x < W - 60; x += 150) {
          const ax = x, ay = BY1 + 22;
          g.poly([[ax - 8 * dir, ay - 6], [ax + 8 * dir, ay], [ax - 8 * dir, ay + 6]], "rgba(85,104,122,0.45)");
        }
        // tool pairs
        c.save(); c.beginPath(); c.rect(BX0, BY0 - 6, BX1 - BX0, BY1 - BY0 + 12); c.clip();
        for (const pr of pairs) {
          const sx = xOf(pr.p);
          g.rr(sx - GAP / 2 + 3, BCY - TOOL_H / 2 + 6, GAP - 6, TOOL_H - 12, 6, "rgba(63,165,91,0.22)", "rgba(63,165,91,0.6)", 1.5);
          for (const s of [-1, 1]) {
            const tx = sx + s * (GAP / 2 + TOOL_W / 2);
            g.rr(tx - TOOL_W / 2 + 3, BCY - TOOL_H / 2 + 4, TOOL_W, TOOL_H, 5, P.shadow);
            g.rr(tx - TOOL_W / 2, BCY - TOOL_H / 2, TOOL_W, TOOL_H, 5, P.yellow, "#c99c12", 2);
            g.line(tx, BCY - TOOL_H / 2 + 12, tx, BCY + TOOL_H / 2 - 12, "rgba(160,120,10,0.35)", 4);
          }
        }
        c.restore();
        // cup on the belt (drawn under the curtains)
        const drawCup = (x, y, lift, alpha) => {
          c.save(); c.globalAlpha = alpha;
          g.shadow(x, y + 4, CUP_R, lift);
          g.circle(x, y - lift * 0.5, CUP_R, P.white, "#9fb1c2", 3);
          g.circle(x, y - lift * 0.5, CUP_R - 7, "#e3eaf1");
          g.rr(x + CUP_R - 3, y - lift * 0.5 - 5, 12, 10, 4, null, "#9fb1c2", 3);   // handle
          c.restore();
        };
        if (cup && (cup.state === "down")) drawCup(cup.x, cup.y, 0, 1);
        // curtains: translucent blue drapes hanging from a rod across the belt
        for (const cu of curtains) {
          const cx = curX(cu), x0 = cx - cu.w / 2, y0 = BY0 - CUR_PAD, h = BY1 - BY0 + 2 * CUR_PAD;
          g.rr(x0, y0, cu.w, h, 6, "rgba(47,109,181,0.55)", "#1f4f8a", 2);
          for (let k = 1; k < 5; k++) g.line(x0 + (cu.w * k) / 5, y0 + 4, x0 + (cu.w * k) / 5, y0 + h - 4, "rgba(255,255,255,0.22)", 2);
          if (cu.sway) g.chip("sways", cx, y0 - 12, P.blue);
        }
        // robot gripper + carried cup
        g.circle(START.x, 536, 44, P.navy);
        if (cup && cup.state !== "down") {
          const lift = cup.state === "lower" ? 40 * (1 - cup.t / LOWER_T) : 40;
          g.line(START.x, 520, cup.x, cup.y - lift * 0.5, "rgba(11,42,69,0.55)", 8);
          drawCup(cup.x, cup.y, lift, 1);
        } else if (!cup) {
          drawCup(START.x, START.y, 40, 1);
          if (ghost && ghost.x > BX0 + CUP_R && ghost.x < BX1 - CUP_R && ghost.y > BY0 + CUP_R && ghost.y < BY1 - CUP_R) {
            drawCup(ghost.x, ghost.y, 0, 0.35);
          }
          g.chip("Click once on the belt to place the cup", W / 2, 140, P.navy);
        }
      }
    };
  }
});
