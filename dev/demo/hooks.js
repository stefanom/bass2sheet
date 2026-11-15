// Put into every page that is filmed, before its own scripts run (film.py). It changes nothing the page does.
// 1. A pointer drawn in the page: a recording made without a window shows none. It follows the mouse and shows
//    clicks; a small label can ride with it (a file being dragged in).
// 2. A log of what the page plays, so that the film's soundtrack can be built from the files themselves and not
//    guessed: every Web Audio source with its buffer, start, offset, rate and the gains it passes through, and
//    every <audio> element's play, pause and place. Times are the audio clock's and the wall clock's, side by side.
(() => {
  if (window.__demo) return;
  const D = (window.__demo = {log: [], sent: 0, doc: Math.random().toString(36).slice(2, 10)});
  const wall = () => performance.timeOrigin + performance.now();  // milliseconds since 1970
  const put = e => { e.wall = wall(); e.doc = D.doc; D.log.push(e); };  // doc: the numbers given to things start again in every page
  D.flush = () => {  // hand what was logged to the film (a binding made by film.py), a batch at a time
    if (D.sent >= D.log.length || !window.__demoSend) return;
    const batch = D.log.slice(D.sent);
    D.sent = D.log.length;
    try { window.__demoSend(JSON.stringify(batch)); } catch (e) { D.sent -= batch.length; }
  };
  setInterval(D.flush, 200);
  window.addEventListener("pagehide", D.flush);

  // ------------------------------------------------------------------------------------------------ the pointer
  const ARROW = `<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24"><path d="M5 2.5v17l4.6-4.2 2.9 6.4 2.7-1.2-2.9-6.3 6.2-.3z" fill="#1f2328" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
  let box = null, ring = null, chip = null, down = false;
  function build() {
    if (box || !document.documentElement) return;
    box = document.createElement("div");
    box.id = "__demoPointer";
    box.style.cssText = "position:fixed;left:0;top:0;width:0;height:0;z-index:2147483647;pointer-events:none;display:none";
    ring = document.createElement("div");
    ring.style.cssText = "position:absolute;left:-17px;top:-17px;width:34px;height:34px;border-radius:50%;border:3px solid #0969da;background:rgba(9,105,218,.18);opacity:0;transform:scale(.4)";
    const arrow = document.createElement("div");
    arrow.style.cssText = "position:absolute;left:-6px;top:-3px;filter:drop-shadow(0 1px 2px rgba(0,0,0,.35))";
    arrow.innerHTML = ARROW;
    chip = document.createElement("div");
    chip.style.cssText = "position:absolute;left:20px;top:22px;display:none;white-space:nowrap;padding:5px 11px 5px 9px;border-radius:7px;border:1px solid #d1d9e0;background:#fff;color:#1f2328;font:600 13px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;box-shadow:0 3px 12px rgba(31,35,40,.3)";
    box.append(ring, arrow, chip);
    document.documentElement.appendChild(box);
  }
  function at(x, y) { build(); if (!box) return; box.style.display = "block"; box.style.transform = `translate(${x}px,${y}px)`; }
  function pulse(on) {
    if (!ring) return;
    ring.style.transition = on ? "none" : "opacity .45s ease-out, transform .45s ease-out";
    ring.style.opacity = on ? "1" : "0";
    ring.style.transform = on ? "scale(.75)" : "scale(1.5)";
  }
  const hide = () => { if (box) box.style.display = "none"; };
  window.addEventListener("mousemove", e => at(e.clientX, e.clientY), true);
  window.addEventListener("mousedown", e => { down = true; at(e.clientX, e.clientY); pulse(true); }, true);
  window.addEventListener("mouseup", () => { down = false; pulse(false); }, true);
  // inside a frame the frame's own pointer is drawn; the page around it lets go of its own
  window.addEventListener("mouseover", e => { if (e.target && e.target.tagName === "IFRAME") hide(); }, true);
  document.addEventListener("mouseleave", () => { if (!down) hide(); });
  // for what cannot be clicked in a recording without a window (a list of choices is drawn by the system, not by
  // the page): the film shows the pointer there and a click, then makes the choice
  D.pointer = (x, y) => at(x, y);
  D.hide = hide;
  D.click = (x, y) => { at(x, y); pulse(true); setTimeout(() => pulse(false), 140); };
  D.carry = text => { build(); if (!chip) return; chip.style.display = text ? "block" : "none"; chip.textContent = text ? "♪ " + text : ""; };
  document.addEventListener("DOMContentLoaded", build);

  // ---------------------------------------------------------------------------------------- what the page plays
  let ids = 0;
  const known = new WeakMap();
  const idOf = o => known.get(o) || (known.set(o, ++ids), ids);
  const from = new WeakMap();  // bytes that were fetched -> where from
  const arrayBuffer = Response.prototype.arrayBuffer;
  Response.prototype.arrayBuffer = function () {
    const url = this.url;
    return arrayBuffer.call(this).then(b => { from.set(b, url); return b; });
  };
  const AC = window.AudioContext || window.webkitAudioContext;
  if (AC) {
    const clock = c => { const o = c.getOutputTimestamp(); return {ctx: c.currentTime, outCtx: o.contextTime, outWall: performance.timeOrigin + o.performanceTime}; };
    const made = [];
    const Wrapped = function (...a) { const c = new AC(...a); made.push(c); put(Object.assign({ev: "context", id: idOf(c), rate: c.sampleRate}, clock(c))); return c; };
    Wrapped.prototype = AC.prototype;
    Object.setPrototypeOf(Wrapped, AC);
    window.AudioContext = Wrapped;
    if (window.webkitAudioContext) window.webkitAudioContext = Wrapped;
    setInterval(() => { for (const c of made) put(Object.assign({ev: "clock", id: idOf(c), state: c.state}, clock(c))); }, 250);
    const decode = AC.prototype.decodeAudioData;
    AC.prototype.decodeAudioData = function (data, ...rest) {
      const url = from.get(data) || null, done = b => { put({ev: "buffer", buf: idOf(b), url, seconds: b.duration, channels: b.numberOfChannels, rate: b.sampleRate}); return b; };
      if (rest.length) return decode.call(this, data, b => { done(b); rest[0](b); }, rest[1]);
      return decode.call(this, data).then(done);
    };
    const start = AudioBufferSourceNode.prototype.start, stop = AudioBufferSourceNode.prototype.stop;
    AudioBufferSourceNode.prototype.start = function (when, offset, duration) {
      put(Object.assign({ev: "start", ac: idOf(this.context), node: idOf(this), buf: this.buffer ? idOf(this.buffer) : null, speed: this.playbackRate.value,
        when: when === undefined ? null : when, offset: offset === undefined ? null : offset, duration: duration === undefined ? null : duration}, clock(this.context)));
      return start.apply(this, arguments);
    };
    AudioBufferSourceNode.prototype.stop = function (when) {
      put(Object.assign({ev: "stop", node: idOf(this), when: when === undefined ? null : when}, clock(this.context)));
      return stop.apply(this, arguments);
    };
    const connect = AudioNode.prototype.connect;
    AudioNode.prototype.connect = function (to) {
      if (to instanceof AudioNode) put({ev: "connect", node: idOf(this), to: to === this.context.destination ? "out" : idOf(to)});
      return connect.apply(this, arguments);
    };
    for (const m of ["setValueAtTime", "linearRampToValueAtTime", "exponentialRampToValueAtTime", "setTargetAtTime", "cancelScheduledValues"]) {
      const native = AudioParam.prototype[m];
      AudioParam.prototype[m] = function (...a) { put({ev: "param", param: idOf(this), how: m, args: a}); return native.apply(this, a); };
    }
    const named = (make, kind, param) => {
      const native = AC.prototype[make];
      AC.prototype[make] = function () { const n = native.apply(this, arguments); put({ev: "node", node: idOf(n), kind, param: idOf(n[param]), value: n[param].value}); return n; };
    };
    named("createGain", "gain", "gain");
    named("createStereoPanner", "pan", "pan");
  }
  // <audio> elements: when each starts, stops and jumps, and where it is in its file, several times a second
  const heard = new WeakSet();
  const listen = el => {
    if (heard.has(el)) return;
    heard.add(el);
    const say = type => put({ev: "media", el: idOf(el), type, url: el.currentSrc || el.src, at: el.currentTime, speed: el.playbackRate, paused: el.paused, volume: el.muted ? 0 : el.volume});
    for (const type of ["playing", "pause", "seeking", "seeked", "waiting", "ended", "ratechange", "emptied", "timeupdate"]) el.addEventListener(type, () => say(type));
    let last = -1;
    setInterval(() => { if (!el.paused && el.currentTime !== last) { last = el.currentTime; say("tick"); } }, 100);
  };
  const play = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function () { listen(this); return play.apply(this, arguments); };
})();
