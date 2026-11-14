// The part of the test-songs page that needs no browser: what is shown of a song, and where it starts to play.
// Usage: node testsongs_page_logic.js PATH/testsongs_page.html   (prints "ok" and exits 0 when every check holds)
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.split('<script id="logic">')[1].split("</script>")[0];
const box = {module: {exports: null}};
vm.runInNewContext(src, box);
const L = box.module.exports;
const same = (a, b) => assert.strictEqual(JSON.stringify(a), JSON.stringify(b));

same([L.clock(0), L.clock(59.9), L.clock(60), L.clock(207.2), L.clock(-3)], ["0:00", "0:59", "1:00", "3:27", "0:00"]);
same([L.score(0.8249), L.score(1), L.score(0), L.score(null), L.score(undefined)], ["0.82", "1.00", "0.00", "–", "–"]);
same(L.esc('Rock & "Roll" <live>'), "Rock &amp; &quot;Roll&quot; &lt;live&gt;");

// a new song starts at its busiest stretch; the same song goes on from where it is when the sound is switched
const song = {id: "a", busy: 61.5, length: 200};
same(L.startAt(song, null, 0), 61.5);
same(L.startAt(song, {song: "b", kind: "record"}, 33), 61.5);
same(L.startAt(song, {song: "a", kind: "record"}, 33), 33);
same(L.startAt({id: "c", busy: 195, length: 200}, null, 0), 190);  // not so late that it ends at once
same(L.startAt({id: "d", busy: 2, length: 6}, null, 0), 0);

same([L.shown({rating: "yes"}, true), L.shown({rating: null}, true), L.shown({rating: "no"}, false)], [false, true, true]);
console.log("ok");
