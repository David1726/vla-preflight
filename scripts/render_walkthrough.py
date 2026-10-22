"""Render actual demo evidence into an illustrated GIF (not a browser screenshot).

Optional maintainer tool: Pillow and NumPy. The core/report does not require it.
"""

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_preflight import Trace  # noqa: E402

INK, MUTED, TEAL, ORANGE = "#17353f", "#5b7078", "#087e8b", "#d86c3c"


def font(size, bold=False):
    windows = Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf")
    linux = Path("/usr/share/fonts/truetype/dejavu") / (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    )
    for path in (windows, linux):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo-output")
    parser.add_argument("--out", default="docs/assets")
    args = parser.parse_args()
    root, output = Path(args.demo), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    ref = Trace(root / "reference")
    frames = []
    cases = [
        ("healthy", "1 / Replay the captured input"),
        ("camera_mapping", "2 / Swap cameras - a seeded integration fault"),
        ("missing_reset", "3 / Omit reset - inspect the next episode"),
        ("fixed", "4 / Restore the integration and verify again"),
    ]
    for name, title in cases:
        result = json.loads((root / name / "result.json").read_text())
        cand = Trace(root / name / "trace")
        first = result["first_recorded_difference"]
        episode, step = (first["episode"], first["step"]) if first else ("episode-0", 0)
        image = Image.new("RGB", (1280, 820), "#f2f6f7")
        d = ImageDraw.Draw(image)

        def text(x, y, value, size=18, color=INK, bold=False):
            d.text((x, y), value, font=font(size, bold), fill=color)

        def card(box):
            d.rounded_rectangle(box, radius=16, fill="white", outline="#d9e3e7", width=2)

        text(45, 25, "VLA Preflight", 26, TEAL, True)
        text(790, 34, "EXECUTED DEMO / ILLUSTRATED WALKTHROUGH", 13, MUTED, True)
        d.line((45, 75, 1235, 75), fill="#d9e3e7", width=2)
        text(45, 103, title, 32, INK, True)
        text(
            45,
            154,
            "Same saved observations. Explicit hooks. First recorded difference.",
            19,
            MUTED,
        )
        card((45, 210, 1235, 360))
        status = result["status"]
        color, bg = ("#11634d", "#d7f1e8") if status == "PASS" else ("#9f3c20", "#fce0d5")
        d.rounded_rectangle((1090, 233, 1207, 277), radius=8, fill=bg)
        text(1111, 240, status, 22, color, True)
        text(70, 227, "FIRST RECORDED DIFFERENCE", 13, TEAL, True)
        text(
            70,
            251,
            f"{first['stage']} / {first['field']}" if first else "No recorded divergence",
            27,
            INK,
            True,
        )
        if first:
            text(
                70,
                293,
                f"{episode} / step {step} / index {first.get('first_index', '-')} / max |delta| {first.get('max_abs_error', '-'):.4g}",
                18,
                MUTED,
            )
        else:
            text(70, 293, "All declared recorded boundaries are within tolerance.", 18, MUTED)
        text(
            70,
            324,
            f"{result['counts']['PASS']} PASS    {result['counts']['FAIL']} FAIL    {result['counts']['UNKNOWN']} UNKNOWN",
            14,
            MUTED,
        )
        card((45, 383, 630, 714))
        card((650, 383, 1235, 714))
        text(70, 402, f"Camera input / front / {episode}:{step}", 20, INK, True)
        key = ("default", episode, step, "model_input")
        for index, trace in enumerate((ref, cand)):
            array = trace.array(trace.records[key], "front")
            thumbnail = Image.fromarray(array).resize((254, 159), Image.Resampling.NEAREST)
            image.paste(thumbnail, (70 + index * 275, 458))
            text(
                70 + index * 275,
                629,
                "Reference" if index == 0 else "Candidate",
                17,
                TEAL if index == 0 else ORANGE,
            )
        text(70, 670, "Preview from captured arrays; no camera identity inferred.", 13, MUTED)
        text(675, 402, f"Command / actions / {episode}", 20, INK, True)
        data = []
        for key, record in ref.records.items():
            if key[1] == episode and key[3] == "command":
                data.append(
                    (
                        key[2],
                        float(ref.array(record, "actions").flat[0]),
                        float(cand.array(cand.records[key], "actions").flat[0]),
                    )
                )
        low, high = min(x for r in data for x in r[1:]), max(x for r in data for x in r[1:])
        span = high - low or 1
        d.line((705, 455, 705, 637, 1200, 637), fill="#c5d4da", width=2)
        for column, color in ((1, TEAL), (2, ORANGE)):
            points = [
                (720 + r[0] / (len(data) - 1) * 460, 620 - (r[column] - low) / span * 145)
                for r in data
            ]
            d.line(points, fill=color, width=4)
            for x, y in points:
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        text(680, 447, f"{high:.3g}", 12, MUTED)
        text(680, 622, f"{low:.3g}", 12, MUTED)
        text(705, 648, "Reference", 15, TEAL)
        text(825, 648, "Candidate", 15, ORANGE)
        text(675, 680, "Flat channel 0 / declared rad / step-aligned samples", 13, MUTED)
        text(
            45,
            743,
            "Seeded fixtures, not upstream bug claims. No model weights, robot, simulation or API key.",
            16,
            INK,
        )
        text(
            45,
            777,
            "Drawn from actual trace/report data. Not a browser screenshot. PASS covers recorded I/O parity only.",
            13,
            MUTED,
        )
        frames.append(image)
        image.save(output / f"walkthrough-{name}.png")
    frames[0].save(
        output / "walkthrough.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[2200, 3000, 3200, 2400],
        loop=0,
        optimize=False,
    )
    print(output.resolve())


if __name__ == "__main__":
    main()
