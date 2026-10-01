/* Place Block Belt — dynamic avoidance.
 * A conveyor belt runs left -> right and drops whatever reaches its end into the exit area, where a bowl waits.
 * Click once on the belt to set down a top-heavy block (it lowers for a moment, then rides the belt).
 * The click picks both the lane (height on the belt) and the travel time to the exit.
 * Success: the block reaches the end upright and falls into the bowl. Blocks set over the belt edge, blocks
 * that hit the blocker (they topple) and blocks that miss the bowl fail.
 * var1: the bowl slides back and forth along the exit, so the block must arrive when the bowl is aligned.
 * var2: a fixed blocker arm reaches over part of the belt.  var1+2: both.
 */
Arcade.register({
  id: "place_block_belt",
  maxTime: 18,
  levels: {
    base: "Fixed bowl, clear belt: choose the right lane.",
    var1: "The exit bowl slides back and forth: time the arrival.",
    var2: "Fixed bowl; a blocker arm reaches over the belt and topples blocks.",
    "var1+2": "Sliding bowl and a blocker arm."
  },
  controls: [["Click", "set the block on the belt (only once)"]],
  create(api) {
    const { P, rand, pick } = api;
    const BX0 = 60, BX1 = 620, BY0 = 175, BY1 = 355, BCY = (BY0 + BY1) / 2;
    const HB = 17, BOWL_R = 44, BOWL_X = BX1 + BOWL_R + 14, LOWER_T = 0.25, FALL_T = 0.22;
    const v = rand(78, 108);
    // Bowl: fixed lane (base/var2) or sliding along the exit (var1).
    const bowl = api.var1
      ? { c: BCY + rand(-12, 12), A: rand(55, 72), om: (2 * Math.PI) / rand(3.0, 4.4), ph: rand(0, 6.28) }
      : { c: rand(BY0 + 30, BY1 - 30), A: 0, om: 0, ph: 0 };
    const bowlY = t => bowl.c + bowl.A * Math.sin(bowl.om * t + bowl.ph);
    // Blocker arm: fixed post at the belt side, arm across part of the belt (often over the bowl's lane).
    let blocker = null;
    if (api.var2) {
      const top = api.var1 ? pick([true, false]) : (bowl.c < BCY ? api.chance(0.75) : api.chance(0.25));
      const reach = rand(0.5, 0.62) * (BY1 - BY0);
      blocker = { x: rand(330, 450), w: 22, y0: top ? BY0 - 26 : BY1 - reach, y1: top ? BY0 + reach : BY1 + 26, top };
    }

    let block = null, ghost = null, beltPhase = 0;
    const onBelt = (x, y) => x > BX0 + 4 && x < BX1 - 30 && y > BY0 - 24 && y < BY1 + 24;

    function topple(msg, ang) { block.state = "down"; block.ang = ang; api.fail(msg); }

    return {
      pointerMove(x, y) { ghost = { x, y }; },
      pointerDown(x, y) {
        if (block || !onBelt(x, y)) return;
        block = { x, y, state: "lower", t: 0, ang: 0 };
        api.sfx.click();
      },
      update(dt) {
        beltPhase = (beltPhase + v * dt) % 36;
        if (!block) return;
        if (block.state === "lower") {
          block.t += dt;
          if (block.t >= LOWER_T) {
            block.state = "ride"; api.sfx.tick();
            if (block.y - HB < BY0 || block.y + HB > BY1) return topple("Set over the belt edge: the block tipped over", block.y < BCY ? -Math.PI / 2 : Math.PI / 2);
            if (blocker && Math.abs(block.x - blocker.x) < HB + blocker.w / 2 && block.y + HB > blocker.y0 && block.y - HB < blocker.y1)
              return topple("Set on the blocker arm: the block tipped over", 0);
          }
        } else if (block.state === "ride") {
          const nx = block.x + v * dt;
          if (blocker && block.x + HB <= blocker.x - blocker.w / 2 && nx + HB > blocker.x - blocker.w / 2 &&
              block.y + HB > blocker.y0 && block.y - HB < blocker.y1) {
            block.x = blocker.x - blocker.w / 2 - HB; api.sfx.bad();
            return topple("The block hit the blocker and toppled", Math.PI / 2);
          }
          block.x = nx;
          if (block.x >= BX1) { block.state = "fall"; block.t = 0; api.sfx.whoosh(); }
        } else if (block.state === "fall") {
          block.t += dt; block.x += v * 0.9 * dt;
          if (block.t >= FALL_T) {
            const by = bowlY(api.time);
            if (Math.abs(block.y - by) < BOWL_R - HB + 2 && block.x > BOWL_X - BOWL_R) {
              block.state = "bowl"; block.dy = block.y - by;
              api.succeed("The block rode upright into the bowl");
            } else {
              block.state = "down"; block.ang = Math.PI / 2;
              api.fail("The block fell off the end and missed the bowl");
            }
          }
        }
      },
      draw(g) {
        const c = g.ctx;
        g.scene();
        // exit trough and bowl track
        g.rr(BX1 + 6, BY0 - 30, 2 * BOWL_R + 28, BY1 - BY0 + 60, 12, "#eef2f6", P.line, 2);
        if (bowl.A) g.line(BOWL_X, bowl.c - bowl.A, BOWL_X, bowl.c + bowl.A, "rgba(85,104,122,0.35)", 4, [6, 6]);
        // belt
        g.rr(BX0 - 10, BY0 - 12, BX1 - BX0 + 20, BY1 - BY0 + 24, 10, P.metal, P.metalDark, 2);
        g.rr(BX0, BY0, BX1 - BX0, BY1 - BY0, 4, "#4a525c");
        c.save(); c.beginPath(); c.rect(BX0, BY0, BX1 - BX0, BY1 - BY0); c.clip();
        for (let x = BX0 - 36 + beltPhase; x < BX1 + 36; x += 36) g.line(x, BY0, x, BY1, "rgba(255,255,255,0.08)", 3);
        c.restore();
        for (let x = 120; x < BX1 - 20; x += 140) g.poly([[x - 8, BY1 + 22], [x + 8, BY1 + 28], [x - 8, BY1 + 34]], "rgba(85,104,122,0.45)");
        // bowl
        const by = bowlY(api.time);
        g.shadow(BOWL_X + 3, by + 6, BOWL_R, 0);
        g.circle(BOWL_X, by, BOWL_R, P.blue, "#1f4f8a", 3);
        g.circle(BOWL_X, by, BOWL_R - 9, "#24558f");
        // block (top-heavy: narrow base, wide top)
        const drawBlock = (x, y, alpha, lift) => {
          c.save(); c.globalAlpha = alpha;
          g.shadow(x + 2, y + 5, HB, lift);
          const yy = y - lift * 0.5;
          g.rr(x - HB * 0.6, yy - HB * 0.6 + 4, HB * 1.2, HB * 1.2, 3, "#b4620f");
          g.rr(x - HB, yy - HB - 4, 2 * HB, 2 * HB, 4, P.orange, "#b4620f", 2);
          g.rr(x - HB + 5, yy - HB + 1, 2 * HB - 10, 2 * HB - 10, 3, "rgba(255,255,255,0.25)");
          c.restore();
        };
        const drawFallen = (x, y, ang) => {
          c.save(); c.translate(x, y); c.rotate(ang);
          g.rr(-HB * 2, -HB + 3, HB * 4, HB * 2, 4, P.shadow);
          g.rr(-HB * 0.6 - HB, -HB * 0.6, HB * 1.6, HB * 1.2, 3, "#b4620f");
          g.rr(HB * 0.1, -HB, HB * 1.9, HB * 2, 4, P.orange, "#b4620f", 2);
          c.restore();
        };
        if (block) {
          if (block.state === "lower") drawBlock(block.x, block.y, 1, 50 * (1 - block.t / LOWER_T));
          else if (block.state === "ride") drawBlock(block.x, block.y, 1, 0);
          else if (block.state === "fall") { c.save(); c.globalAlpha = 1 - block.t / FALL_T * 0.3; drawBlock(block.x, block.y, 1, -10 * block.t / FALL_T); c.restore(); }
          else if (block.state === "bowl") drawBlock(BOWL_X + 2, by + block.dy * 0.6, 1, 0);
          else drawFallen(block.x, block.y, block.ang);
        } else if (ghost && onBelt(ghost.x, ghost.y)) drawBlock(ghost.x, ghost.y, 0.35, 0);
        // blocker arm (drawn above the belt)
        if (blocker) {
          const b = blocker, py = b.top ? b.y0 : b.y1;
          g.rr(b.x - b.w / 2 + 4, b.y0 + 6, b.w, b.y1 - b.y0, 6, P.shadow);
          g.rr(b.x - b.w / 2, b.y0, b.w, b.y1 - b.y0, 6, P.metal, P.metalDark, 2);
          g.circle(b.x, py, 16, P.metalDark);
          g.circle(b.x, py, 7, P.metal);
        }
        if (!block) g.chip("Click once on the belt to set the block down", 330, 500, P.navy);
      }
    };
  }
});
