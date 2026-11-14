// The part of the play-along page that needs no browser: which line and note are being played, where the playhead
// is, how the two lines fit a screen, and looping.
// Usage: node playalong_page_logic.js PATH/playalong_page.html   (prints "ok" and exits 0 when every check holds)
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.split('<script id="logic">')[1].split("</script>")[0];
const box = {module: {exports: null}};
vm.runInNewContext(src, box);
const L = box.module.exports;
const same = (a, b) => assert.strictEqual(JSON.stringify(a), JSON.stringify(b));
const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol, `${a} is not ${b}`);

// two lines of two bars; the record slows down: the bars last 2.0, 2.2, 2.4 and 2.6 seconds
const lines = [
  {file: "a", end: 5.2, bars: [[1, 0.1, 0.55, 1.0], [2, 0.55, 1.0, 3.0]],
   marks: [[0.2, 0.02, 1.0, "n"], [0.4, 0.02, 2.0, "n"], [0.6, 0.02, 3.0, "r"], [0.8, 0.02, 4.1, "t"]]},
  {file: "b", end: 10.2, bars: [[3, 0.05, 0.5, 5.2], [4, 0.5, 1.0, 7.6]],
   marks: [[0.1, 0.02, 5.2, "n"], [0.3, 0.02, 6.4, "n"], [0.55, 0.02, 7.6, "n"]]},
];
const bars = L.prepare(lines);
same(lines.map(l => [l.index, l.start, l.first]), [[0, 1, 0], [1, 5.2, 2]]);
same(bars.map(b => [b.number, b.t, b.end, b.line]), [[1, 1, 3, 0], [2, 3, 5.2, 0], [3, 5.2, 7.6, 1], [4, 7.6, 10.2, 1]]);

// the line and the bar being played: the last that has started; before the song's first bar, the first
same([0, 1, 5.19, 5.2, 9, 99].map(t => L.lineAt(lines, t)), [0, 0, 0, 1, 1, 1]);
same([0, 2.99, 3, 5.2, 7.59, 7.6, 99].map(t => L.barAt(bars, t)), [0, 0, 1, 2, 2, 3, 3]);

// the playhead: on a note when it starts, evenly between two, at the end of the last bar when the line ends
near(L.xAt(lines[0], 0), 0.2);          // before the line: where its first note is
near(L.xAt(lines[0], 1.0), 0.2);        // the bar line and the note on it share a moment: the note has it
near(L.xAt(lines[0], 1.5), 0.3);
near(L.xAt(lines[0], 2.0), 0.4);
near(L.xAt(lines[0], 3.55), 0.7);       // halfway through the rest
near(L.xAt(lines[0], 4.65), 0.9);       // between the last mark and the line's end
near(L.xAt(lines[0], 5.2), 1.0);
near(L.xAt(lines[0], 9), 1.0);
let last = -1;
for (let t = 0; t <= 6; t += 0.01) { const x = L.xAt(lines[0], t); assert.ok(x >= last); last = x; }  // never back

// the note being played, and the box drawn on it
same([0.5, 1, 1.99, 2, 3.5, 4.1, 5.1].map(t => L.markAt(lines[0], t)), [-1, 0, 0, 1, 2, 3, 3]);
const b0 = L.boxOf(lines[0], 0), b3 = L.boxOf(lines[0], 3);
near(b0.left, 0.194); near(b0.left + b0.width, 0.394); same(b0.kind, "n");   // up to just before the next note
near(b3.left + b3.width, 1.0); same(b3.kind, "t");                           // the last: up to the end of the line
const close = {bars: [[1, 0, 1, 0]], marks: [[0.5, 0.03, 0, "n"], [0.51, 0.03, 0.1, "n"]], end: 1};
assert.ok(L.boxOf(close, 0).width >= 0.03);                                  // never thinner than the note

// a touch finds its bar
same([0, 0.1, 0.54, 0.55, 0.99].map(x => L.barUnder(lines[0], x)), [0, 0, 0, 1, 1]);

// two lines on a phone held sideways (750 x 312 of score): the one being played as large as fits, the next under
// it, smaller, nothing cut off
const phone = L.fit(750, 312, 2037, 479, 6);
assert.ok(phone.cw <= 750 + 1e-9 && phone.ch + phone.nh + 6 <= 312 + 1e-9);
assert.ok(phone.ch > 170 && phone.nh < phone.ch && phone.nh > 0.6 * phone.ch);
near(phone.cw / phone.ch, 2037 / 479, 1e-9); near(phone.scale, phone.nh / phone.ch);
// a taller picture (a line with high notes) is held to its share of the height and centred
const tall = L.fit(750, 312, 2037, 606, 6);
near(tall.ch, 312 * 0.58, 1e-9); near(tall.left, (750 - tall.cw) / 2, 1e-9);
// on a tablet both lines are as wide as the screen
const tablet = L.fit(1180, 700, 2037, 479, 6);
near(tablet.cw, 1180); near(tablet.nh, tablet.ch);
// a low window still shows both
const low = L.fit(800, 150, 2037, 479, 6);
assert.ok(low.ch + low.nh + 6 <= 150 + 1e-9 && low.nh >= 0.5 * low.ch - 1e-9);

// a loop of bars: its span in seconds, kept inside the song; at its end the song jumps back
same(L.loopSpan(bars, 1, 2), [3, 7.6, 1, 2]);
same(L.loopSpan(bars, 3, 4), [7.6, 10.2, 3, 3]);
same(L.loopSpan(bars, -2, 1), [1, 3, 0, 0]);
const span = L.loopSpan(bars, 1, 2);
same([3, 5, 7.58].map(t => L.loopJump(t, span)), [null, null, null]);
same([7.59, 8, 2.5].map(t => L.loopJump(t, span)), [3, 3, 3]);   // at its end, past it, or well before it
same(L.loopJump(2.8, span), null);                                // the sound arriving late shows a moment before

// a slowed file lasts longer by as much as it is slower
near(L.fileTime(30, 80), 37.5); near(L.songTime(37.5, 80), 30); near(L.songTime(L.fileTime(12.3, 50), 50), 12.3);

// the position runs with the clock, leans toward what the player reports, and jumps when the player does
let est = 10;
for (let i = 0; i < 300; i++) est = L.follow(est, 10 + (i + 1) / 60 + 0.05, 1 / 60, 0);   // the player is 50 ms ahead
near(est, 15 + 0.05, 0.002);
near(L.follow(10, 42, 1 / 60, 0), 42);
let steady = 10, worst = 0, stale = 0, heard = 10;
for (let i = 0; i < 120; i++) {   // a player that reports a new time only every quarter of a second
  const before = steady, reported = 10 + Math.floor((i + 1) / 15) * 0.25;
  stale = reported === heard ? stale + 1 / 60 : 0; heard = reported;
  steady = L.follow(steady, reported, 1 / 60, stale);
  worst = Math.max(worst, Math.abs(steady - before - 1 / 60));
}
assert.ok(worst < 0.002); near(steady, 12, 0.002);   // moves on evenly all the same
for (let i = 0; i < 40; i++) { stale += 1 / 60; steady = L.follow(steady, 12, 1 / 60, stale); }
near(steady, 12);   // the player waits for more sound: the page does not run ahead of it

same([0, 59.9, 60, 754, -3].map(L.clock), ["0:00", "0:59", "1:00", "12:34", "0:00"]);
console.log("ok");
