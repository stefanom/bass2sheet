"""The film's picture for the README: one frame of the film with a play button on it, and the same frame plain.

    python dev/demo/poster.py bass2sheet-demo.mp4 --at 89.5 --out docs/demo-poster.jpg --plain bass2sheet-demo-poster.jpg

The README shows the first and links it to the page that plays the film; that page shows the second until the
film starts. Needs ffmpeg on the PATH and Pillow.
"""

import argparse
import io
import subprocess

from PIL import Image, ImageDraw, ImageFont

INK, WHITE = (31, 35, 40), (255, 255, 255)
BAND = 0.125  # the share of the picture's height the captions take, at the bottom
FONTS = ("/System/Library/Fonts/HelveticaNeue.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def frame(film, at, width, height):
    """One frame of the film as an image."""
    cmd = ["ffmpeg", "-v", "error", "-ss", str(at), "-i", film, "-frames:v", "1"]
    cmd += ["-vf", f"scale={width}:{height}", "-f", "image2pipe", "-vcodec", "png", "-"]
    return Image.open(io.BytesIO(subprocess.run(cmd, capture_output=True, check=True).stdout)).convert("RGB")


def font(size):
    for path in FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def with_button(picture, label):
    """The picture with a play button in the middle of what is above the captions, and `label` under it."""
    w, h = picture.size
    over = Image.new("RGBA", picture.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    cx, cy, r = w // 2, int(h * (1 - BAND) / 2), int(h * 0.115)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*INK, 215), outline=(*WHITE, 255), width=max(3, r // 14))
    t = int(r * 0.42)
    d.polygon([(cx - t * 0.72, cy - t), (cx - t * 0.72, cy + t), (cx + t * 1.05, cy)], fill=(*WHITE, 255))
    face = font(int(h * 0.036))
    box = d.textbbox((0, 0), label, font=face)
    tw, th, pad = box[2] - box[0], box[3] - box[1], int(h * 0.016)
    top = cy + r + int(h * 0.03)
    d.rounded_rectangle(
        [cx - tw // 2 - 2 * pad, top, cx + tw // 2 + 2 * pad, top + th + 2 * pad], radius=th, fill=(*INK, 225)
    )
    d.text((cx - tw // 2 - box[0], top + pad - box[1]), label, font=face, fill=(*WHITE, 255))
    return Image.alpha_composite(picture.convert("RGBA"), over).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("film")
    ap.add_argument("--at", type=float, default=89.5, help="the moment of the film to show, in seconds")
    ap.add_argument("--label", default="Watch the film · 3:54, with sound")
    ap.add_argument("--out", required=True, help="the picture with the play button (a .jpg)")
    ap.add_argument("--plain", help="the same frame without the button (a .jpg)")
    ap.add_argument("--size", default="1280x720")
    args = ap.parse_args()
    width, height = (int(v) for v in args.size.split("x"))
    picture = frame(args.film, args.at, width, height)
    with_button(picture, args.label).save(args.out, quality=86, optimize=True, progressive=True)
    if args.plain:
        picture.save(args.plain, quality=88, optimize=True, progressive=True)


if __name__ == "__main__":
    main()
