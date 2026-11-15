# What bass2sheet takes from Homebrew. `./bootstrap` runs this file; by hand:
#
#     brew bundle install --no-upgrade
#
# Every name was checked with `brew info` on 2026-11-14. Homebrew has built packages of all of them for
# Apple Silicon. For Intel Macs it has none today (only rubberband has old ones, for macOS 13 and 14), so
# there brew compiles them from source, which is slow and not tested here.

# ---- Needed

# The Python the tool runs on. 3.12 and no newer: several of the audio models have no builds for later ones.
brew "python@3.12"

# Creates the Python environment from uv.lock.
brew "uv"

# ---- For the whole tool. Each can be left out: `bass2sheet-doctor` says what is lost.

# MuseScore 4 engraves the PDF and the pictures of the score in the copilot, in play-along and in judging.
# Without it you get MusicXML and no PDF. The cask also puts `mscore` on the PATH.
cask "musescore" if OS.mac?

# The separation library and yt-dlp call ffmpeg. Skipped where Homebrew already has an ffmpeg on the PATH:
# `ffmpeg-full` provides one too, and the two cannot both be linked. Without any, bass2sheet uses the copy
# inside its Python environment.
brew "ffmpeg" unless (HOMEBREW_PREFIX/"bin/ffmpeg").exist?

# Rubber Band slows a song down without changing its pitch, for the copilot and play-along. Without it
# ffmpeg does that, and a bass line comes out rougher.
brew "rubberband"

# Adding a song to the copilot by a link. Without it the copilot takes files only.
brew "yt-dlp"

# ---- For the tests only

# The tests run the pages' scripts under Node.js. Without it those tests are skipped; the tool needs no Node.js.
brew "node"

# ---- Optional: MuScriptor, the other note finder (docs/muscriptor.md)

# What its program needs to build. Not installed unless asked for: `./bootstrap --with-muscriptor`, or
# HOMEBREW_BASS2SHEET_MUSCRIPTOR=1 brew bundle install --no-upgrade. The build also needs Xcode, for its
# C++ compiler and its Metal toolchain; Homebrew cannot install Xcode. The weights are not fetched by
# anything here: they have conditions of their own.
if ENV["HOMEBREW_BASS2SHEET_MUSCRIPTOR"]
  brew "cmake"
  brew "ninja"
end
