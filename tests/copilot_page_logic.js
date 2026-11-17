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
same(L.between(notes, "f", "c"), ["c", "d", "e", "f"]);   // a run of notes, whichever end is named first
same(L.between(notes, "b", "b"), ["b"]);
same(L.between(notes, "b", "nope"), []);

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

// the mouse: which part of a note the pointer is on. A note 40 wide from x = 100, its ends grabbed from 7 outside
const part = (x, left, width, ends) => { const z = L.zone(x, left, width, 7, ends); return z && [z.mode, z.off]; };
same(part(92, 100, 40), null);                 // too far before it
same(part(94, 100, 40), ["start", 6]);         // from outside: the start
same(part(103, 100, 40), ["start", 0]);        // the outer part inside: 8 wide
same(part(108, 100, 40), ["move", 0]);         // the body
same(part(131, 100, 40), ["move", 0]);
same(part(133, 100, 40), ["end", 0]);
same(part(146, 100, 40), ["end", 6]);
same(part(148, 100, 40), null);
same(part(103, 100, 8), ["move", 0]);          // a short note: its body is most of it, its ends are grabbed from outside
same(part(101, 100, 8), ["start", 0]);
same(part(112, 100, 8), ["end", 4]);
same(part(94, 100, 40, false), null);          // what sounds with a note has no ends of its own
same(part(103, 100, 40, false), ["move", 0]);

// a note dragged in time, between the note before (which gives way) and the note after (which does not)
const fix = v => ({start: +v.start.toFixed(3), end: +v.end.toFixed(3)});
const before = n("p", 0.5, 1.0, 40), mid = n("q", 1.0, 1.4, 43), after = n("r", 1.6, 2.0, 45);
same(fix(L.shifted(mid, before, after, 0.1, 10)), {start: 1.1, end: 1.5});     // room: it keeps its length
same(fix(L.shifted(mid, before, after, -0.3, 10)), {start: 0.7, end: 1.1});    // into the note before: that one gives way
same(fix(L.shifted(mid, before, after, -0.9, 10)), {start: 0.53, end: 0.93});  // but keeps 30 ms of its own
same(fix(L.shifted(mid, before, after, 0.3, 10)), {start: 1.3, end: 1.6});     // against the note after: the end stays, the start goes on
same(fix(L.shifted(mid, before, after, 5, 10)), {start: 1.57, end: 1.6});      // as long as a note is left
same(fix(L.shifted(mid, null, null, -5, 10)), {start: 0, end: 0.4});           // the first note stops at the beginning
same(fix(L.shifted(mid, null, null, 20, 10)), {start: 9.6, end: 10});          // the last one at the end of the song
// an end dragged
same(fix(L.resized(mid, before, after, "start", 0.2, 10)), {start: 0.53, end: 1.4});
same(fix(L.resized(mid, before, after, "start", 1.39, 10)), {start: 1.37, end: 1.4});
same(fix(L.resized(mid, before, after, "end", 1.9, 10)), {start: 1.0, end: 1.6});
same(fix(L.resized(mid, before, after, "end", 0.2, 10)), {start: 1.0, end: 1.03});
same(fix(L.resized(mid, null, null, "end", 99, 10)), {start: 1.0, end: 10});

// zooming keeps the moment under the pointer where it is
let z = L.zoomAbout(10, 100, 300, 2);
same([z.t0, z.pps], [11.5, 200]);              // 13 s was 300 px in, and still is
assert.strictEqual(z.t0 + 300 / z.pps, 13);
z = L.zoomAbout(10, 100, 300, 1000); assert.strictEqual(z.pps, 2000);   // no further in than this
z = L.zoomAbout(10, 100, 300, 0.001); assert.strictEqual(z.pps, 20);    // nor further out
assert.strictEqual(L.zoomAbout(0, 100, 600, 0.5).t0, -0.2);             // the song's beginning stays in sight

// the picture's colours: from the first to the last, the faint tones fading or showing with the curve
assert.strictEqual(L.gammaOf(50), 2);          // the middle of the slider: the squares the page always used
assert.ok(L.gammaOf(0) > 7.9 && L.gammaOf(100) < 0.51);
const blue = L.palette([[255, 255, 255], [50, 75, 130]], 2);
same([...blue.slice(0, 3)], [255, 255, 255]);
same([...blue.slice(255 * 3)], [50, 75, 130]);
same([...blue.slice(128 * 3, 128 * 3 + 3)], [Math.round(255 - 205 * (128 / 255) ** 2), Math.round(255 - 180 * (128 / 255) ** 2), Math.round(255 - 125 * (128 / 255) ** 2)]);
const three = L.palette([[0, 0, 0], [100, 0, 0], [100, 200, 0]], 1);    // through a colour on the way
same([...three.slice(51 * 3, 51 * 3 + 3)], [40, 0, 0]);
same([...three.slice(204 * 3, 204 * 3 + 3)], [100, 120, 0]);
assert.ok(L.palette([[0, 0, 0], [255, 255, 255]], 0.5)[64 * 3] > L.palette([[0, 0, 0], [255, 255, 255]], 2)[64 * 3] + 80);  // faint tones show

// the repeats of a bar, one under the other: the pitches their strips show, and where each bar is drawn
same(L.pitchRange([40, 45, 43], 26, 59), [38, 47]);            // a margin of two around the notes
same(L.pitchRange([40], 26, 59), [36, 45]);                    // never fewer than ten rows
same(L.pitchRange([27, 30], 26, 59), [26, 35]);                // nor outside what the picture has
same(L.pitchRange([58], 26, 59), [50, 59]);
same(L.pitchRange([], 26, 59), [26, 35]);
same(L.pitchRange([30, 55], 26, 59), [28, 57]);
assert.strictEqual(L.alignX(300, 200, 50, 1000), 300);         // right under the bar it is compared with
assert.strictEqual(L.alignX(-400, 200, 50, 1000), 50);         // that bar is out of sight to the left: at the left edge
assert.strictEqual(L.alignX(950, 200, 50, 1000), 800);         // or to the right: all of it still in sight
assert.strictEqual(L.alignX(300, 2000, 50, 1000), 50);         // a bar wider than the window starts at its left

// the repeats' strips: as tall as the window leaves room for
assert.strictEqual(L.dockHeight(1300, 100, 900, 500), 378);    // many strips, a tall window: 42% of it
assert.strictEqual(L.dockHeight(200, 100, 900, 500), 200);     // few strips: all of them
assert.strictEqual(L.dockHeight(1300, 100, 720, 281), 281);    // a low window: what is left, the panel under them stays in sight
assert.strictEqual(L.dockHeight(1300, 170, 600, 149), 149);    // lower than one strip: it scrolls
assert.strictEqual(L.dockHeight(1300, 100, 500, 40), 72);      // a strip's buttons and a few rows at the least

// notes played together: one of them taken out, or moved, by itself
const duo = {id: "n7", pitch: 42, also: [43]}, trio = {id: "n8", pitch: 40, also: [47, 52]};
same(L.dropTone(duo, 43), {kind: "also", id: "n7", also: []});                    // the upper one goes: the note stays
same(L.dropTone(duo, 42), {kind: "pitch", id: "n7", pitch: 43, offer: false});    // the note itself goes: the other one is the note
same(L.dropTone(trio, 47), {kind: "also", id: "n8", also: [52]});
same(L.dropTone(trio, 40), {kind: "pitch", id: "n8", pitch: 47, offer: false});   // the lowest that is left
assert.strictEqual(L.dropTone(duo, 50), null);                                    // not one of them
assert.strictEqual(L.dropTone({id: "n9", pitch: 42}, 42), null);                   // a note by itself is deleted whole
same(L.moveTone(duo, 43, 49), {kind: "also", id: "n7", also: [49]});
same(L.moveTone(trio, 52, 45), {kind: "also", id: "n8", also: [45, 47]});
same(L.moveTone(trio, 52, 47), {kind: "also", id: "n8", also: [47]});             // laid on another of them: it is gone
same(L.moveTone(duo, 43, 42), {kind: "also", id: "n7", also: []});                // ... or on the note itself
assert.strictEqual(L.moveTone(duo, 43, 43), null);
assert.strictEqual(L.moveTone(duo, 42, 45), null);                                // the note itself moves as any note does

// what is heard and how loud: a slider for each sound. At first the record and the notes in the song's own bass
// sound as they are, the rest off
same(L.heard(null), {record: 100, bass: 0, notes: 100, synth: 0, chords: 0, split: false});
same(L.heard({record: 40, bass: 100, notes: 0, synth: 70, chords: 65.4, split: true}), {record: 40, bass: 100, notes: 0, synth: 70, chords: 65, split: true});
// what was kept before the notes had a second sound: that one is off
same(L.heard({record: 50, bass: 100, notes: 0, chords: 0, split: false}), {record: 50, bass: 100, notes: 0, synth: 0, chords: 0, split: false});
// what was kept when each sound was a box to tick: ticked is up, clear is down
same(L.heard({record: true, bass: true, notes: false, split: false, chords: false}), {record: 100, bass: 100, notes: 0, synth: 0, chords: 0, split: false});
same(L.heard({chords: true}).chords, 80);
same(L.heard({record: 140, bass: -3, notes: "loud", synth: NaN, chords: null}), {record: 100, bass: 0, notes: 100, synth: 0, chords: 0, split: false});  // nonsense: as at first
// a slider's position as a gain: its square, so that half way is a quarter; 100 is the sound as it is
same([L.gainAt(100), L.gainAt(50), L.gainAt(0), L.gainAt(200), L.gainAt(-5)], [1, 0.25, 0, 1, 0]);
assert.ok(Math.abs(L.gainAt(80) - 0.64) < 1e-9);
// the chords' slider all the way down: not with the song, but a chord that is asked for is still heard
same([L.chordsAt(0, false), L.chordsAt(0, true), L.chordsAt(30, true), L.chordsAt(30, false), L.chordsAt(100, true)], [0, L.CHORDS_ASKED, 30, 30, 100]);
assert.ok(L.CHORDS_ASKED >= 60 && L.CHORDS_ASKED <= 100);

console.log("ok");
