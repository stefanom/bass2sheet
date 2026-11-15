"""A stand-in for MuseScore, for tests: it takes the command lines bass2sheet gives MuseScore and writes files of
the kinds MuseScore writes, laid out in the simplest way that is still a score's layout. The tests that need a
score drawn can then run on a machine without MuseScore, and in a fraction of the time.

    mscore -S STYLE.mss -o OUT.pdf IN.musicxml          a PDF
    mscore -S STYLE.mss [-r DPI] -j JOB.json            for each job's outputs: .pdf, .svg (a picture a page),
                                                        .png (a picture a page), .mpos and .spos (where every
                                                        measure and every note or rest is, and when it is played)

What it draws is not music: staff lines, bar lines and a block where each note or rest is. But it reads the
MusicXML it is given (which has to be well formed, with bars that add up), puts a line on a new page where the
score asks for one, and says where and when everything is as MuseScore does: positions in twelfths of a pixel of
the picture, times in milliseconds at the score's own tempo. See playalong.positions for how they are read.

The tests reach it through conftest.mscore_stand_in, which makes a program of it.
"""

import json
import re
import struct
import sys
import zlib
from xml.etree import ElementTree

SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="1100" viewBox="0 0 850 1100">'
    '<rect width="850" height="1100" fill="white"/><title>page {k}</title>'
    '<text x="60" y="80" font-size="28">{title}: page {k} of {n} (a stand-in for MuseScore)</text>'
    '<g stroke="black">{lines}</g></svg>'
)


def measures_of(xml_path):
    """(how long a bar lasts at the tempo the score names, in milliseconds; for each measure: whether it starts a
    new page, and the sixteenths of the bar where a note or rest of the upper staff is) of a MusicXML file."""
    root = ElementTree.parse(xml_path).getroot()
    assert root.tag == "score-partwise", root.tag
    tempo = next((float(e.get("tempo")) for e in root.iter("sound") if e.get("tempo")), 120.0)
    out = []
    for measure in root.iter("measure"):
        new_page = any(p.get("new-page") == "yes" for p in measure.iter("print"))
        at, places, total = 0, [], {}
        for el in measure:
            if el.tag == "backup":
                at -= int(el.findtext("duration"))
            elif el.tag == "forward":
                at += int(el.findtext("duration"))
            elif el.tag == "note" and el.find("grace") is None:
                length, staff = int(el.findtext("duration")), el.findtext("staff")
                if el.find("chord") is None:
                    if staff == "1":
                        places.append(at)
                    total[staff] = total.get(staff, 0) + length
                    at += length
        assert set(total.values()) == {16}, f"measure {measure.get('number')} does not add up: {total}"
        out.append((new_page, places))
    assert out, "the score has no measures"
    return 4 * 60000.0 / round(tempo), out


def pages_of(measures):
    """The measures by page: [[(index of the measure, its places)]]."""
    pages = []
    for k, (new_page, places) in enumerate(measures):
        if new_page or not pages:
            pages.append([])
        pages[-1].append((k, places))
    return pages


def png(path, rows):
    """A greyscale PNG of `rows`, a list of bytearrays of one length."""
    height, width = len(rows), len(rows[0])
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    head = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(
            b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")
        )


def positions(path, items):
    """A positions file as MuseScore writes it: items are (x, y, width, height, page, ms), in pixels."""
    elements = "".join(
        f'<element id="{i}" x="{x * 12:.2f}" y="{y * 12:.2f}" sx="{w * 12:.2f}" sy="{h * 12:.2f}" page="{page}">'
        "</element>"
        for i, (x, y, w, h, page, _) in enumerate(items)
    )
    events = "".join(f'<event elid="{i}" position="{ms:.4f}"></event>' for i, (*_, ms) in enumerate(items))
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n<score><elements>{elements}</elements>')
        f.write(f"<events>{events}</events></score>\n")


def layout(bar_ms, measures, style, dpi):
    """Where everything is: (page width, page height, the measures' boxes, the notes' and rests' boxes), the boxes
    as `positions` takes them. Every page holds one line; its bars share the width evenly."""
    inches = lambda name, default: (
        float(m.group(1)) if (m := re.search(rf"<{name}>([\d.]+)</{name}>", style)) else default
    )
    width, height = round(inches("pageWidth", 8.5) * dpi), round(inches("pageHeight", 11) * dpi)
    left = round(inches("pageOddLeftMargin", 0.5) * dpi)
    across = round(inches("pagePrintableWidth", 7.5) * dpi)
    top, tall = round(0.16 * height), round(0.27 * height)
    boxes, marks = [], []
    for page, held in enumerate(pages_of(measures)):
        each = across / len(held)
        for slot, (k, places) in enumerate(held):
            x0 = left + slot * each
            boxes.append((x0, top, each, tall, page, k * bar_ms))
            for at in places:
                x = x0 + (at + 0.5) / 16 * each
                marks.append((x, top, min(0.3 * each / 16, 30), tall, page, (k + at / 16) * bar_ms))
    return width, height, boxes, marks


def draw(stem, width, height, boxes, marks):
    """One picture a page, as MuseScore names them: STEM-1.png, STEM-2.png, ..."""
    for page in sorted({b[4] for b in boxes}):
        rows = [bytearray([255]) * width for _ in range(height)]
        mine = [b for b in boxes if b[4] == page]
        x0, x1 = round(mine[0][0]), round(mine[-1][0] + mine[-1][2])
        top, tall = round(mine[0][1]), round(mine[0][3])
        for line in range(9):  # five lines of the staff, four of the tab under it
            y = top + round(line * tall / 8)
            for dy in (0, 1):
                rows[min(height - 1, y + dy)][x0:x1] = bytes(x1 - x0)
        for x, *_ in mine + [(x1 - 2,)]:  # the bar lines
            for y in range(top, top + tall):
                rows[y][round(x) : round(x) + 2] = bytes(2)
        for x, y, w, h, p, _ in marks:
            if p == page:  # a block for the note or the rest, and a sign above the staff, as chord symbols are
                for yy in (*range(round(y + 0.25 * h), round(y + 0.35 * h)), *range(round(y - 40), round(y - 30))):
                    rows[yy][round(x) : round(x + w)] = bytes(round(x + w) - round(x))
        png(f"{stem}-{page + 1}.png", rows)


def engrave(xml_path, outs, style, dpi):
    bar_ms, measures = measures_of(xml_path)
    width, height, boxes, marks = layout(bar_ms, measures, style, dpi)
    for out in outs:
        stem, kind = out.rsplit(".", 1)
        if kind == "pdf":
            with open(out, "wb") as f:
                f.write(b"%PDF-1.4\n% a stand-in for MuseScore\n")
        elif kind == "svg":
            n = max(1, -(-len(measures) // 20))  # twenty bars to a page
            title = re.sub(r"[<&]", "", open(xml_path).read().split("<movement-title>")[-1].split("<")[0])[:60]
            lines = "".join(f'<line x1="60" x2="790" y1="{y}" y2="{y}"/>' for y in range(160, 1000, 60))
            for k in range(1, n + 1):
                with open(f"{stem}-{k}.svg", "w") as f:
                    f.write(SVG.format(k=k, n=n, title=title or "Score", lines=lines))
        elif kind == "png":
            draw(stem, width, height, boxes, marks)
        elif kind == "mpos":
            positions(out, boxes)
        elif kind == "spos":
            positions(out, marks)
        else:
            sys.exit(f"the stand-in for MuseScore does not write .{kind} files")


def main(argv):
    args, style, dpi, job, out, rest = list(argv), "", 300.0, None, None, []
    while args:
        a = args.pop(0)
        if a == "-S":
            style = open(args.pop(0)).read()
            assert "<museScore" in style and "<Style>" in style, "not a style file"
        elif a == "-r":
            dpi = float(args.pop(0))
        elif a == "-j":
            job = args.pop(0)
        elif a == "-o":
            out = args.pop(0)
        else:
            rest.append(a)
    if job:
        for one in json.load(open(job)):
            engrave(one["in"], one["out"], style, dpi)
    elif out and len(rest) == 1:
        engrave(rest[0], [out], style, dpi)
    else:
        sys.exit(f"the stand-in for MuseScore does not know what to do with {argv}")


if __name__ == "__main__":
    main(sys.argv[1:])
