// The score in a panel over a song's page: the copilot's score page (/score) in a frame, closed with its button,
// with Esc or with a click beside it. A song's page loads this file and calls showScore(SONG_ID).
(() => {
  let panel = null, frame = null;
  const open = () => panel && panel.style.display !== "none";
  function build() {
    panel = document.createElement("div");
    panel.id = "scorePanel";
    panel.style.cssText = "position:fixed;inset:0;z-index:1000;display:none;background:rgba(31,35,40,.55)";
    frame = document.createElement("iframe");
    frame.title = "The score";
    frame.style.cssText = "position:absolute;left:4%;top:3%;width:92%;height:94%;border:0;border-radius:10px;background:#fff;box-shadow:0 8px 40px rgba(0,0,0,.35)";
    panel.appendChild(frame);
    panel.addEventListener("mousedown", e => { if (e.target === panel) hideScore(); });
    document.body.appendChild(panel);
  }
  function showScore(songId) {
    if (!panel) build();
    frame.src = "/score?embed=1&at=" + Date.now() + "#" + encodeURIComponent(songId);  // a score made again is read again
    panel.style.display = "block";
  }
  function hideScore() {
    if (!open()) return;
    panel.style.display = "none";
    frame.src = "about:blank";
  }
  window.addEventListener("message", e => { if (e.origin === location.origin && e.data === "copilot-score-close") hideScore(); });
  // while the score is shown the keys are not the song page's: Space does not play, the arrows move no note
  document.addEventListener("keydown", e => {
    if (!open()) return;
    e.stopImmediatePropagation();
    if (e.key === "Escape") hideScore();
    if (e.key === " ") e.preventDefault();
  }, true);
  window.showScore = showScore;
  window.hideScore = hideScore;
})();
