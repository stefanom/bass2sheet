// The part of the copilot's score page that needs no browser: when a score was made, and from how much work.
// Usage: node copilot_score_logic.js PATH/copilot_score.html   (prints "ok" and exits 0 when every check holds)
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.split('<script id="logic">')[1].split("</script>")[0];
const box = {module: {exports: null}};
vm.runInNewContext(src, box);
const S = box.module.exports;
const same = (a, b) => assert.strictEqual(JSON.stringify(a), JSON.stringify(b));

const now = new Date(2026, 10, 14, 13, 5);
same(S.when("2026-11-14T12:55:38", now), "today at 12:55");
same(S.when("2026-11-13T23:59:00", now), "yesterday at 23:59");
same(S.when("2026-11-02T09:05:00", now), "2 Nov at 9:05");
same(S.when("nonsense", now), "");

same(S.from({fixes: 0, checked: 0}), "0 fixes · not checked by ear yet");
same(S.from({fixes: 1, checked: 0.124}), "1 fix · 12% checked by ear");
same(S.from({fixes: 22, checked: 0.997}), "22 fixes · all of it checked by ear");
same(S.esc(`a "b" <c>`), "a &quot;b&quot; &lt;c&gt;");


// the versions to choose from, the latest first
const versions = [{version: 3, when: "2026-11-14T12:55:38", fixes: 22, checked: 0.5}, {version: 1, when: "2026-11-02T09:05:00", fixes: 0, checked: 0}];
same(S.label(versions[0], versions, now), "Latest: today at 12:55 · 22 fixes · 50% checked by ear");
same(S.label(versions[1], versions, now), "Version 1: 2 Nov at 9:05 · 0 fixes · not checked by ear yet");

console.log("ok");
