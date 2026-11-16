// The part of the harmony page that needs no browser: where the song is, the steps it takes on a map, the neck.
// Usage: node copilot_harmony_logic.js PATH/copilot_harmony.html   (prints "ok" and exits 0 when every check holds)
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.split('<script id="logic">')[1].split("</script>")[0];
const box = { module: { exports: null } };
vm.runInNewContext(src, box);
const L = box.module.exports;
const same = (a, b) => assert.strictEqual(JSON.stringify(a), JSON.stringify(b));

same([L.clock(0), L.clock(59.9), L.clock(60), L.clock(207.2)], ["0:00", "0:59", "1:00", "3:27"]);
same(L.esc('Rock & "Roll" <live>'), "Rock &amp; &quot;Roll&quot; &lt;live&gt;");
// flats and sharps are written as music writes them, in chord names and in steps
same(["Bb", "F#m7", "bVII", "bIII7", "#iv°", "b3", "b7↓", "Eb major", "R"].map(L.flat), ["B♭", "F♯m7", "♭VII", "♭III7", "♯iv°", "♭3", "♭7↓", "E♭ major", "R"]);

// four chords, two bars each, on three spots of a map (the last is the first again)
const chord = (start, end, spot, far = 0) => ({ start, end, far, on: { roads: spot } });
const chords = [chord(0, 2, 5), chord(2, 4, 5), chord(4, 6, 8), chord(6, 8, 2), chord(8, 10, 5), chord(10, 12, 8)];
same([L.at(chords, 0), L.at(chords, 1.99), L.at(chords, 2), L.at(chords, 11.9), L.at(chords, 12), L.at(chords, -1)], [0, 0, 1, 5, -1, -1]);
same(L.at([], 3), -1);
same(L.at([chord(1, 2, 0), chord(5, 6, 0)], 3), -1);  // between two stretches
// the step into a chord comes from the last chord on another spot
same([L.before(chords, 0, "roads"), L.before(chords, 1, "roads"), L.before(chords, 2, "roads"), L.before(chords, 4, "roads")], [-1, -1, 1, 3]);
// every step once, with how often; a chord held for another bar is no step; a stretch of time narrows it
same(L.moves(chords, "roads"), [{ a: 5, b: 8, times: 2 }, { a: 8, b: 2, times: 1 }, { a: 2, b: 5, times: 1 }]);
same(L.moves(chords, "roads", 6, 10), [{ a: 8, b: 2, times: 1 }, { a: 2, b: 5, times: 1 }]);
// a click on a spot goes to where the song is on it next, and round to the first time when there is none later
same([L.next(chords, "roads", 8, 0), L.next(chords, "roads", 8, 4.5), L.next(chords, "roads", 8, 11), L.next(chords, "roads", 9, 0)], [2, 5, 2, -1]);

// an arrow stops short of both ends and bows a little; two chords that touch get none
const d = L.bow([0, 0], [100, 0], 22);
assert.ok(/^M22\.0,0\.0Q48\.5,-16\.0 75\.0,0\.0$/.test(d), d);
same(L.bow([0, 0], [40, 0], 22), null);
assert.ok(L.wedge(-1.5, 1.5, 62, 206, 300, 215).startsWith("M256.2,171.2L154.3,69.3A206,206 0 0 1 445.7,69.3"), L.wedge(-1.5, 1.5, 62, 206, 300, 215));

// the places a bar is played at, each once, in the order they are first played, with how often
const played = (string, fret, degree, role) => ({ string, fret, degree, role });
same(L.shape([played(0, 0, "R", "root"), played(0, 0, "R", "root"), played(1, 2, "5", "fifth"), played(2, 2, "8", "root"), played(1, 2, "5", "fifth"), played(1, 0, "4", "other")]), [
  { string: 0, fret: 0, degree: "R", role: "root", times: 2 }, { string: 1, fret: 2, degree: "5", role: "fifth", times: 2 },
  { string: 2, fret: 2, degree: "8", role: "root", times: 1 }, { string: 1, fret: 0, degree: "4", role: "other", times: 1 },
]);
same(L.shape([]), []);
same([L.fretName(0, 0), L.fretName(1, 2), L.fretName(2, 1), L.fretName(3, 3), L.fretName(3, 7), L.fretName(0, 11), L.fretName(1, 12)], [
  "the open E string", "the 2nd fret of the A string", "the 1st fret of the D string", "the 3rd fret of the G string",
  "the 7th fret of the G string", "the 11th fret of the E string", "the 12th fret of the A string",
]);

// the profile: each chord holds its distance from rest for as long as it lasts
same(L.steps([chord(0, 2, 0, 0), chord(2, 4, 0, 3.6)], t => 10 * t, v => 100 - 10 * v), "M0.0,100.0L20.0,100.0L20.0,64.0L40.0,64.0");
same(L.steps([], t => t, v => v), "");

// a part of 8 bars that goes round a loop of 4, starting at bar 3 (bars are 2 s long, bar 1 at 0 s)
const bars = Array.from({ length: 12 }, (_, k) => ({ bar: k + 1, time: 2 * k }));
const part = { bar: 3, bars: 8, size: 4 };
same([L.round(part, bars, 4), L.round(part, bars, 5.9), L.round(part, bars, 11.5), L.round(part, bars, 12), L.round(part, bars, 19.9)], [[1, 0, 2], [1, 0, 2], [1, 3, 5], [2, 0, 6], [2, 3, 9]]);
// a part that begins three bars into its loop of four: its first bar is the loop's last
const late = { bar: 3, bars: 6, size: 4, skip: 3 };
same([L.round(late, bars, 4), L.round(late, bars, 6), L.round(late, bars, 13.9), L.round(late, bars, 14)], [[1, 3, 2], [2, 0, 3], [2, 3, 6], [3, 0, 7]]);
// the bar of the song where a part plays a bar of its loop, kept inside the part
same([L.barOf(part, 1, 0), L.barOf(part, 2, 1), L.barOf(late, 2, 0), L.barOf(late, 1, 3), L.barOf(late, 1, 0), L.barOf(late, 3, 3)], [2, 7, 3, 2, 2, 7]);
same(L.spans([{ bar: 1, bars: 24 }, { bar: 97, bars: 24 }]), "bars 1–24, 97–120");

// a loop said short: each chord once; "two bars each" when every chord lasts the same
const c = (numeral, job) => [{ numeral, job }];
same(L.said([c("i", "rest"), c("i", "rest"), c("bVII", "drift"), c("bVII", "drift")]), { runs: [{ key: "i", bars: 2, job: "rest" }, { key: "bVII", bars: 2, job: "drift" }], each: 2 });
same(L.said([c("I", "rest"), c("IV", "drift"), c("IV", "drift"), []]).each, 0);
same(L.said([c("I", "rest").concat(c("V", "pull")), c("IV", "drift")]).runs.map(r => r.key), ["I V", "IV"]);
// a chord in the keyboard sound: its root below middle C, the other notes above the root
same([L.voicing(4, [4, 7, 11]), L.voicing(0, [0, 4, 7, 10]), L.voicing(11, [11, 2, 6]), L.voicing(9, [0, 4, 9])], [[52, 55, 59], [48, 52, 55, 58], [59, 62, 66], [57, 60, 64]]);
// the bars where the parts with one letter play a bar of their loop: here the second bar of a loop of four
const song = [{ letter: "A", bar: 1, bars: 8, size: 4, skip: 0 }, { letter: "B", bar: 9, bars: 4, size: 2, skip: 0 }, { letter: "A", bar: 13, bars: 6, size: 4, skip: 3 }];
same([L.rounds(song, "A", 1), L.rounds(song, "A", 3), L.rounds(song, "B", 0), L.rounds(song, "C", 0)], [[2, 6, 15], [4, 8, 13, 17], [9, 11], []]);
// the beat nearest to a time, as sixteenths from the start of bar 1: where a tap puts a mark
const beats = bars.map(bar => ({ ...bar, beats: [0, 0.5, 1, 1.5].map(x => bar.time + x) }));
same([L.beatAt(beats, 0), L.beatAt(beats, 0.2), L.beatAt(beats, 0.3), L.beatAt(beats, 1.9), L.beatAt(beats, 5.6), L.beatAt(beats, 99), L.beatAt(beats, -3)], [0, 0, 4, 16, 44, 188, 0]);
// what a bar shows of the person's chords: stretches of beats, with what comes into the bar from before
const Bm = { name: "Bm", job: "rest" }, A7 = { name: "A7", job: "pull" };
const marks = [{ at: 16, chord: Bm, open: false }, { at: 40, chord: null, open: true }, { at: 44, chord: null, open: false }, { at: 64, chord: A7, open: false }];
same(L.barCells(marks, 1, null), [{ beat: 0, beats: 4, kind: "chord", chord: Bm }]);
same(L.barCells(marks, 2, Bm), [{ beat: 0, beats: 2, kind: "goes on", chord: Bm }, { beat: 2, beats: 1, kind: "open", chord: null }, { beat: 3, beats: 1, kind: "none", chord: null }]);
same(L.barCells(marks, 3, null), [{ beat: 0, beats: 4, kind: "empty", chord: null }]);
same(L.barCells(marks, 0, null).map(c => c.kind), ["empty"]);
same(L.barCells([], 5, A7), [{ beat: 0, beats: 4, kind: "goes on", chord: A7 }]);
// the bars shown in a row around a bar: its round; for a loop of one or two bars, four bars; inside the part
same([L.window(part, 2), L.window(part, 5), L.window(part, 6), L.window(late, 2), L.window(late, 3), L.window(late, 7)], [[2, 3, 4, 5], [2, 3, 4, 5], [6, 7, 8, 9], [2], [3, 4, 5, 6], [7]]);
same([L.window({ bar: 3, bars: 7, size: 1 }, 2), L.window({ bar: 3, bars: 7, size: 1 }, 7), L.window({ bar: 3, bars: 7, size: 2, skip: 1 }, 2), L.window({ bar: 1, bars: 9, size: 3 }, 7)], [[2, 3, 4, 5], [6, 7, 8], [2, 3, 4], [6, 7, 8]]);
// the rounds of the parts with one letter, as sixteenths: a part that comes in on its loop's last bar has a
// first round of one bar, and a round cut short at the end of its part is a round too
same(L.roundsOf(song, "A"), [[0, 64, 0], [64, 128, 64], [192, 208, 144], [208, 272, 208], [272, 288, 272]]);
same(L.roundsOf(song, "B"), [[128, 160, 128], [160, 192, 160]]);
same(L.roundsOf(song, "C"), []);
// the round a beat is in, how many marks it has, and how many other rounds of its part have none
const at = n => ({ at: n });
same(L.emptyRounds(song, [at(0), at(32)], 20), { round: [0, 64, 0], marked: 2, empty: 4 });
same(L.emptyRounds(song, [at(0), at(32), at(64), at(280)], 20), { round: [0, 64, 0], marked: 2, empty: 2 });
same(L.emptyRounds(song, [at(0)], 100), { round: [64, 128, 64], marked: 0, empty: 3 });
same(L.emptyRounds(song, [], 16 * 40), { round: null, marked: 0, empty: 0 });
console.log("ok");
