// The part of the copilot's landing page that needs no browser: what a file is refused for, and the few words
// each step and each song gets.
// Usage: node copilot_landing_logic.js PATH/copilot_landing.html   (prints "ok" and exits 0 when every check holds)
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.split('<script id="logic">')[1].split("</script>")[0];
const box = {module: {exports: null}};
vm.runInNewContext(src, box);
const L = box.module.exports;
const same = (a, b) => assert.strictEqual(JSON.stringify(a), JSON.stringify(b));

// times
same([L.clock(0), L.clock(59.6), L.clock(187.2)], ["0:00", "1:00", "3:07"]);
same([L.took(null), L.took(0.4), L.took(12.4), L.took(75)], ["", "under a second", "12 s", "1:15"]);

// a file is not sent when its name or its size already says no
const kinds = [".wav", ".mp3"], limit = 10 * 1048576;
same(L.refusal("Song.WAV", 1000, kinds, limit), null);
same(L.refusal("a.b.mp3", limit, kinds, limit), null);
assert(L.refusal("notes.txt", 1000, kinds, limit).includes("not an audio file"));
assert(L.refusal("wav", 1000, kinds, limit).includes("not an audio file"));
assert(L.refusal("song.wav", 0, kinds, limit).includes("is empty"));
same(L.refusal("song.wav", 11 * 1048576, kinds, limit), "song.wav is 11 MB: the copilot takes files up to 10 MB");

// what a step found, once it is done
const done = (id, found) => L.fact({id, state: "done", found});
same(done("reading", {length: 187.2}), "3:07 of music");
same([done("isolating", {by: "roformer"}), done("isolating", {by: "given"})], ["by BS-RoFormer", "from the stems that came with it"]);
same(done("pitch", {pitched: 0.416}), "a pitch in 42% of it");
same([done("notes", {notes: 266, low: "F#1", high: "Ab2"}), done("notes", {notes: 1})], ["266 notes, F#1 to Ab2", "1 note"]);
same(done("second", {notes: 300, alone: 21}), "21 of 300 only MuScriptor hears");
same(done("beats", {tempo: 157.9, bars: 167, key: "B minor"}), "158 bpm · 167 bars · B minor");
same(done("beats", {tempo: 120, bars: 12, key: "A minor", tracker: "librosa", fallback: true}),
  "120 bpm · 12 bars · A minor · by the built-in beat tracker: the beat model could not be used");
same([done("checks", {doubted: 38}), done("checks", {doubted: 0})], ["38 notes doubted", "0 notes doubted"]);
same([done("ready", {}), done("unknown", {a: 1}), done("notes", {})], ["", "", ""]);
same(L.fact({id: "notes", state: "running", found: {notes: 3}}), "");

// how far the person is with a song
const song = {length: 247, worked: 0, checked: 0, fixes: 0};
same(L.progress(song), "Not started");
same(L.progress({...song, checked: 41}), "Heard 0:41 of 4:07, nothing fixed yet");
same(L.progress({...song, checked: 85, worked: 24.1, fixes: 22}), "Worked through 0:24 of 4:07 · 22 fixes");
same(L.progress({...song, worked: 5, fixes: 1}), "Worked through 0:05 of 4:07 · 1 fix");
same([L.whose("own"), L.whose("muscriptor")], ["bass2sheet's own notes", "MuScriptor's notes"]);
same(L.esc(`<b a="1">R&B's</b>`), "&lt;b a=&quot;1&quot;&gt;R&amp;B&#39;s&lt;/b&gt;");


// when a song's score was made
const now = new Date(2026, 10, 14, 13, 5);
same([L.when("2026-11-14T12:55:38", now), L.when("2026-11-13T08:05:00", now), L.when("2026-10-30T19:30:00", now), L.when("?", now)],
  ["today at 12:55", "yesterday at 8:05", "30 Oct at 19:30", ""]);

console.log("ok");
