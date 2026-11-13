// Drives the judging page's own script against a running server, with a stand-in for the browser's DOM and audio.
// Usage: node judge_page_smoke.js http://127.0.0.1:PORT   (prints "ok" and exits 0 when every check holds)
const vm = require("vm");
const assert = require("assert");

const base = process.argv[2];
const els = {}, kids = {}, handlers = {}, played = [];

function makeEl(id) {
  const cls = new Set(["reveal", "next", "done", "results", "waiting", "why"].includes(id) ? ["hidden"] : []);
  const e = {
    id, dataset: {}, style: {}, value: "", checked: id === "loop", disabled: false, textContent: "", clientWidth: 100, html: "",
    classList: {
      toggle(name, force) { const on = force === undefined ? !cls.has(name) : !!force; on ? cls.add(name) : cls.delete(name); return on; },
      add: name => cls.add(name), remove: name => cls.delete(name), contains: name => cls.has(name),
    },
  };
  Object.defineProperty(e, "innerHTML", {
    get() { return e.html; },
    set(v) {
      e.html = v;
      kids[id] = [...v.matchAll(/<button class="[^"]*"(?: disabled)? data-(name|pick|step|why)="([^"]+)">/g)].map(m => {
        const b = makeEl(id + ":" + m[2]);
        b.dataset[m[1]] = m[2];
        return b;
      });
    },
  });
  return e;
}
global.document = {
  getElementById: id => els[id] || (els[id] = makeEl(id)),
  querySelectorAll: sel => kids[sel.split(" ")[0].slice(1)] || [],
  addEventListener: (type, fn) => { handlers[type] = fn; },
};
global.Audio = class {
  constructor(src) { this.src = src; this.currentTime = 0; this.paused = true; played.push(src); }
  addEventListener() {}
  pause() { this.paused = true; }
  play() { this.paused = false; return Promise.resolve(); }
};
global.requestAnimationFrame = () => {};
const realFetch = fetch;
global.fetch = (path, options) => realFetch(base + path, options);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const see = expr => vm.runInThisContext(expr);
const hidden = id => els[id].classList.contains("hidden");
const key = k => handlers.keydown({ key: k, target: { tagName: "BODY" }, preventDefault() {} });

(async () => {
  const page = await (await realFetch(base + "/")).text();
  vm.runInThisContext(page.match(/<script>([\s\S]*)<\/script>/)[1]);
  await sleep(300);

  // the first comparison is on screen, and nothing on it says which version is which
  const first = see("current.id");
  assert.ok(els.title.textContent.includes("Song One"), els.title.textContent);
  assert.strictEqual(els.progress.textContent, "0 of 3 judged");
  assert.deepStrictEqual(kids.players.map(b => b.dataset.name), ["record", "A", "B"]);
  assert.strictEqual(kids.picks.length, 4);
  assert.ok(hidden("reveal") && hidden("next") && !hidden("skip"));
  assert.ok(!/default|RoFormer|Demucs/.test(els.about.textContent + els.ask.textContent + els.sheets.html));

  // the scores: cropped page images, a cursor on each, the differing notes shaded
  assert.ok(/id="score-A"[^>]*aspect-ratio:1576\/264/.test(els.sheets.html), els.sheets.html);
  assert.strictEqual((els.sheets.html.match(/class="mark"/g) || []).length, 1);
  assert.ok(els.legend.textContent.length > 20);
  assert.strictEqual(els.stepname.textContent, "Step 1 of 6 · Isolate the bass");
  assert.deepStrictEqual(kids.steps.map(b => b.dataset.step), ["separator", "replucks", "together", "barlines", "rhythm", "tab"]);
  assert.ok(/class="step current" data-step="separator">1 Isolate the bass<span>0\/1</, els.steps.html);
  assert.ok(els.steps.html.includes('disabled data-step="barlines">4 Count the bars<span>–<'), els.steps.html);
  const left = name => parseFloat(els["cursor-" + name].style.left);
  assert.ok(Math.abs(left("A") - 100 * (216 - 36) / 1576) < 0.01, els["cursor-A"].style.left);
  see("place(1.4)");   // half way between the notes at 0.4 s (x 216) and 2.4 s (x 520)
  assert.ok(Math.abs(left("A") - 100 * (368 - 36) / 1576) < 0.01 && left("B") === left("A"));
  see("place(99)");
  assert.ok(Math.abs(left("A") - 100 * (1602 - 36) / 1576) < 0.01);

  // a click on a score plays that version from the note under the click
  els["score-B"].onclick({ offsetX: 50, currentTarget: { clientWidth: 100 } });   // x 824, between 520 and 1602
  assert.strictEqual(see("playing"), "B");
  assert.ok(Math.abs(see("audio.B.currentTime") - (2.4 + 6 * (824 - 520) / (1602 - 520))) < 1e-6);
  assert.ok(els["score-B"].classList.contains("active") && !els["score-A"].classList.contains("active"));
  els["score-A"].onclick({ offsetX: 0, currentTarget: { clientWidth: 100 } });    // B keeps playing, from the start
  assert.strictEqual(see("playing"), "B");
  assert.strictEqual(see("audio.B.currentTime"), 0.4);
  key(" ");

  // playing: buttons and keys switch source, the bass-only switch swaps the files
  kids.players[1].onclick();
  assert.strictEqual(see("playing"), "A");
  key("b");
  assert.strictEqual(see("playing"), "B");
  key(" ");
  assert.strictEqual(see("playing"), null);
  key("s");
  assert.ok(played.some(src => src.endsWith("A_bass.flac")), played.join(" "));

  // a pick is saved and only then are the versions named
  els.note.value = "too busy";
  kids.picks[0].onclick();
  await sleep(300);
  assert.ok(!hidden("reveal") && !hidden("next") && hidden("skip"));
  assert.ok(/A<\/b> was .+B<\/b> was /.test(els.reveal.html), els.reveal.html);
  assert.strictEqual(els.progress.textContent, "1 of 3 judged");

  // "neither is right" asks what is wrong: a reason is one click, and is saved with the pick as its note
  assert.ok(hidden("why"));
  kids.picks[3].onclick();
  await sleep(300);
  assert.ok(!hidden("why") && els.why.html.includes("What is wrong?"));
  assert.deepStrictEqual(kids.why.map(b => b.dataset.why), ["wrong notes", "too high", "too low", "extra notes", "missing notes", "wrong timing"]);
  await kids.why[1].onclick();
  assert.strictEqual(els.note.value, "too busy, too high");
  assert.ok(/class="chip chosen" data-why="too high"/.test(els.why.html), els.why.html);
  let saved = (await (await realFetch(base + "/api/results")).json());
  assert.ok(JSON.stringify(saved).includes("too busy, too high"), JSON.stringify(saved).slice(0, 400));
  await kids.why[1].onclick();                                   // a second click takes it back
  assert.strictEqual(els.note.value, "too busy");
  // a note written after the pick is saved as well
  els.note.value = "too busy, bar 2";
  await els.note.onchange();
  saved = (await (await realFetch(base + "/api/results")).json());
  assert.ok(JSON.stringify(saved).includes("too busy, bar 2"));
  els.note.value = "too busy";
  kids.picks[0].onclick();                                       // back to A: the reasons go away
  await sleep(300);
  assert.ok(hidden("why"));

  // next, skip (the skipped one comes back after the rest), previous
  els.next.onclick();
  await sleep(300);
  const second = see("current.id");
  assert.notStrictEqual(second, first);

  // the second step: its own sources, and a pitch picture with the notes and the tracked pitch drawn on it
  assert.strictEqual(kids.players.length, 3);
  assert.ok(els.players.html.includes("▶ Isolated bass") && els.soloLabel.textContent === "Real bass left, notes right");
  assert.ok(els.keys.textContent.includes("S real bass left, notes right"), els.keys.textContent);
  const drawn = els.sheets.html;
  assert.strictEqual((drawn.match(/<svg /g) || []).length, 2);
  assert.strictEqual((drawn.match(/<rect class="n"/g) || []).length, 2);     // one plain note on each picture
  assert.strictEqual((drawn.match(/<rect class="n d"/g) || []).length, 2);   // and one that the other lacks
  assert.strictEqual((drawn.match(/<polyline /g) || []).length, 4);          // the pitch line, broken where it is unsure
  assert.ok(drawn.includes('<rect class="n" x="68.2" y="198.0" width="67.2" height="9.0" rx="2"/>'), drawn);
  assert.ok(drawn.includes(">E1</text>") && drawn.includes(">G2</text>") && els.legend.textContent.includes("open strings"));
  see("place(4.4)");
  assert.strictEqual(els["cursor-A"].style.left, "50%");
  key("s");
  assert.ok(played.some(src => src.endsWith("A_both.flac")), played.join(" "));
  els.skip.onclick();
  await sleep(300);
  const third = see("current.id");
  assert.ok(third !== first && third !== second);
  assert.deepStrictEqual(kids.players.map(b => b.dataset.name), ["record", "A"]);  // two fingerings sound alike
  els["score-B"].onclick({ offsetX: 50, currentTarget: { clientWidth: 100 } });      // so B's score plays the one there is
  assert.strictEqual(see("playing"), "A");
  key(" ");
  kids.picks[2].onclick();
  await sleep(300);
  els.next.onclick();
  await sleep(300);
  assert.strictEqual(see("current.id"), second);
  els.back.onclick();
  await sleep(300);
  assert.strictEqual(see("current.id"), first);
  assert.ok(!hidden("reveal") && els.note.value === "too busy");

  // the step bar jumps to a step and shows what is done
  kids.steps[5].onclick();
  await sleep(300);
  assert.strictEqual(see("current.id"), third);
  assert.ok(/class="step current done" data-step="tab">6 Place it on the neck<span>1\/1</.test(els.steps.html), els.steps.html);
  kids.steps[0].onclick();
  await sleep(300);
  assert.strictEqual(see("current.id"), first);

  // the picks so far
  els.toggleResults.onclick();
  await sleep(300);
  assert.ok(!hidden("results") && hidden("judging"));
  assert.ok(els.results.html.includes("too busy") && els.results.html.includes("No difference"));
  // nothing left to show while a step is being prepared: wait, and look again
  const idle = { total: 3, judged: 3, steps: [], error: null, comparison: null, previous: null };
  els.toggleResults.onclick();
  see("show")({ ...idle, preparing: true }, "");
  assert.ok(!hidden("waiting") && hidden("done") && hidden("judging") && see("timer") !== null);
  see("show")({ ...idle, preparing: false, error: "ValueError: boom" }, "");
  assert.ok(!hidden("waiting") && els.workerError.textContent.includes("boom") && see("timer") === null);
  see("show")({ ...idle, preparing: false }, "");
  assert.ok(hidden("waiting") && !hidden("done"));
  console.log("ok");
})().catch(e => { console.error(e); process.exit(1); });
