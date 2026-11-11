"""bass2sheet: song audio file -> printable bass sheet, with quality gates.

    bass2sheet SONG.mp3 [-o OUT_DIR] [--title T] [--artist A] [--strict]

Exit codes: 0 = sheet written and the gates passed (possibly with warnings), 2 = the result failed its quality
gates (the sheet, if written, is marked DRAFT), 1 = error (unreadable input, no bass found, ...).
"""

import argparse
import logging
import os
import sys
import warnings

from . import __version__
from .common import Bass2SheetError


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="bass2sheet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("audio", help="song audio or video file (mp3, wav, m4a, mp4, ...)")
    ap.add_argument("-o", "--out", help="output directory (default: '<input name> bass' next to the input)")
    ap.add_argument("--title", help="score title (default: from the file name)")
    ap.add_argument("--artist", help="artist shown on the score (default: from 'Artist - Title' file names)")
    ap.add_argument(
        "--separator",
        choices=["auto", "roformer", "demucs"],
        default="auto",
        help="auto: BS-RoFormer, falling back to Demucs when it fails or its result fails the gates",
    )
    ap.add_argument("--strict", action="store_true", help="write no sheet when the result fails its quality gates")
    ap.add_argument("--no-audio", action="store_true", help="skip the listening files (about 160 MB per song)")
    ap.add_argument("--no-pdf", action="store_true", help="write MusicXML and MIDI but skip the PDF")
    ap.add_argument("--bass-stem", help="use this audio file as the bass instead of the separated one (for evaluation)")
    ap.add_argument(
        "--stems",
        metavar="DIR",
        help="skip separation: read bass.*, drums.* and optionally harmony.*/other.* and vocals.* from this directory",
    )
    ap.add_argument(
        "--beat-tracker",
        choices=["auto", "beat_this", "librosa"],
        default="auto",
        help="auto: the Beat This! model, falling back to the built-in tracker when it is unavailable",
    )
    ap.add_argument(
        "--crepe-model",
        choices=["full", "tiny"],
        default="full",
        help="pitch model: full is the most accurate; tiny is several times faster",
    )
    ap.add_argument(
        "--second-opinion",
        choices=["auto", "off"],
        default="auto",
        help="run Basic Pitch as an independent check (needs its own Python; see README)",
    )
    ap.add_argument("--basic-pitch-python", help="Python interpreter that has basic-pitch and onnxruntime")
    ap.add_argument("--mscore", help="MuseScore 4 executable (default: found on PATH or in /Applications)")
    ap.add_argument(
        "--cache-dir", default="~/.cache/bass2sheet", help="models, stems and pitch tracks (default: %(default)s)"
    )
    ap.add_argument("--device", choices=["mps", "cuda", "cpu"], help="for separation (default: best available)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"bass2sheet {__version__}")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        warnings.filterwarnings("ignore")
        os.environ.setdefault("TQDM_DISABLE", "1")  # model download and inference progress bars
        for noisy in ("audio_separator", "numba", "matplotlib", "urllib3", "httpx", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    if not os.path.isfile(args.audio):
        print(f"bass2sheet: no such file: {args.audio}", file=sys.stderr)
        return 1
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.audio)), os.path.splitext(os.path.basename(args.audio))[0] + " bass"
    )
    from .pipeline import run

    try:
        return run(
            args.audio,
            out,
            title=args.title,
            artist=args.artist,
            separator=args.separator,
            strict=args.strict,
            cache_dir=args.cache_dir,
            mscore=args.mscore,
            renders=not args.no_audio,
            second_mode=args.second_opinion,
            bp_python=args.basic_pitch_python,
            device=args.device,
            bass_stem=args.bass_stem,
            pdf=not args.no_pdf,
            stems_dir=args.stems,
            crepe_model=args.crepe_model,
            beat_tracker=args.beat_tracker,
        )
    except Bass2SheetError as e:
        print(f"bass2sheet: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
