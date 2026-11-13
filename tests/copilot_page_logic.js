// The part of the copilot page that needs no browser: which note lends its sound to which, and getting about.
// Usage: node copilot_page_logic.js PATH/copilot_page.html   (prints "ok" and exits 0 when every check holds)
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.split('<script id="logic">')[1].split("</script>")[0];
const box = {module: {exports: null}};
vm.runInNewContext(src, box);
const L = box.module.exports;
const same = (a, b) => assert.strictEqual(JSON.stringify(a), JSON.stringify(b));

const n = (id, start, end, pitch, also) => ({id, start, end, pitch, also: also || []});
const notes = [
  n("a", 0.0, 0.5, 40), n("b", 0.5, 0.7, 40), n("c", 0.7, 1.4, 40), n("d", 1.4, 1.9, 43),
  n("e", 1.9, 1.95, 45), n("f", 2.0, 2.5, 47, [52]), n("g", 2.5, 3.0, 52), n("h", 3.0, 3.4, 40),
];
const by = L.indexNotes(notes);
const lends = (id, pitch) => { const s = L.sampleFor(by, notes.find(m => m.id === id), pitch); return s && [s.from.id, +s.rate.toFixed(4)]; };

// very short notes and chords lend no sound
same([...by.keys()].sort(), [40, 43, 52]);
same(by.get(40).map(m => m.id), ["a", "b", "c", "h"]);

// a note is never played by its own sound: of its pitch, by one that lasts as long, the nearest in length
same(lends("a", 40), ["c", 1]);   // h is the nearer in length but too short
same(lends("b", 40), ["h", 1]);   // all three last long enough: the nearest in length
same(lends("c", 40), ["a", 1]);   // none lasts as long: the longest
// with none of its pitch, by the nearest pitch, played faster or slower
same(lends("d", 43), ["a", +Math.pow(2, 3 / 12).toFixed(4)]);
same(lends("e", 45), ["d", +Math.pow(2, 2 / 12).toFixed(4)]);
same(lends("f", 52), ["g", 1]);   // the second note of a chord
same(lends("f", 47), ["d", +Math.pow(2, 4 / 12).toFixed(4)]);
same(lends("g", 52), null);       // nothing within reach: it stays silent
// a note set to another pitch takes that pitch's sound at once
same(lends("h", 43), ["d", 1]);

// getting about
assert.strictEqual(L.neighbour(notes, "a", 1, 0), "b");
assert.strictEqual(L.neighbour(notes, "a", -1, 0), "a");
assert.strictEqual(L.neighbour(notes, "h", 1, 0), "h");
assert.strictEqual(L.neighbour(notes, null, -1, 1.0), "c");   // nothing selected: the note just heard
assert.strictEqual(L.neighbour(notes, null, 1, 1.0), "d");
assert.strictEqual(L.neighbour(notes, null, -1, -1), null);
assert.strictEqual(L.neighbour([notes[3], notes[0]], "a", 1, 0), "d");   // in order of time, whatever the list's
same(L.sorted([notes[3], notes[0]]).map(m => m.id), ["a", "d"]);

// where a play-through picks up
assert.strictEqual(L.resumeAt(notes, [[0, 1.2], [5, 6]]), 1.2);
assert.strictEqual(L.resumeAt(notes, []), 0);
assert.strictEqual(L.resumeAt([n("x", 3, 3.5, 40)], [[10, 12]]), 2);
assert.ok(L.inside(1, [[0, 2]]) && !L.inside(3, [[0, 2], [5, 6]]));
assert.strictEqual(L.total([[0, 2], [5, 6]]), 3);

// what is shown
assert.strictEqual(L.clock(75.2), "1:15.2");
assert.strictEqual(L.clock(5), "0:05.0");
assert.strictEqual(L.clock(-1), "0:00.0");
const bars = [{bar: 1, time: 1.0}, {bar: 2, time: 3.0}];
same(L.barAt(bars, 3.5), [2, 0.5]);
same(L.barAt(bars, 0.5), [0, 0.5]);

console.log("ok");
