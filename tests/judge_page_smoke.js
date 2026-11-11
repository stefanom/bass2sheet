// Drives the judging page's own script against a running server, with a stand-in for the browser's DOM and audio.
// Usage: node judge_page_smoke.js http://127.0.0.1:PORT   (prints "ok" and exits 0 when every check holds)
const vm = require("vm");
const assert = require("assert");

const base = process.argv[2];
const els = {}, kids = {}, handlers = {}, played = [];

function makeEl(id) {
  const cls = new Set(["reveal", "next", "done", "results"].includes(id) ? ["hidden"] : []);
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
      kids[id] = [...v.matchAll(/<button class="[^"]*" data-(name|pick)="([^"]+)">/g)].map(m => {
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

  // next, skip (the skipped one comes back after the rest), previous
  els.next.onclick();
  await sleep(300);
  const second = see("current.id");
  assert.notStrictEqual(second, first);
  els.skip.onclick();
  await sleep(300);
  const third = see("current.id");
  assert.ok(third !== first && third !== second);
  assert.deepStrictEqual(kids.players.map(b => b.dataset.name), ["record", "A"]);  // two fingerings sound alike
  kids.picks[2].onclick();
  await sleep(300);
  els.next.onclick();
  await sleep(300);
  assert.strictEqual(see("current.id"), second);
  els.back.onclick();
  await sleep(300);
  assert.strictEqual(see("current.id"), first);
  assert.ok(!hidden("reveal") && els.note.value === "too busy");

  // the picks so far
  els.toggleResults.onclick();
  await sleep(300);
  assert.ok(!hidden("results") && hidden("judging"));
  assert.ok(els.results.html.includes("too busy") && els.results.html.includes("No difference"));
  console.log("ok");
})().catch(e => { console.error(e); process.exit(1); });
