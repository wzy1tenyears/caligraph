import json
import os.path
import random
import re
import sys
import urllib.parse

import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from strokes_code import STOKES_CODE


PATH_TOKEN_RE = re.compile(r'[MLQCZ]|-?\d+(?:\.\d+)?')
FONT_STYLES = ('standard', 'kaishu', 'bold', 'thin')
SYSTEM_FONT_ALIASES = {
    'simkai': 'simkai.ttf',
    'kaiti': 'simkai.ttf',
    'simhei': 'simhei.ttf',
    'heiti': 'simhei.ttf',
    'simsun': 'simsun.ttc',
    'songti': 'simsun.ttc',
    'msyh': 'msyh.ttc',
    'yahei': 'msyh.ttc',
}
FONT_CHOICES = FONT_STYLES + tuple(SYSTEM_FONT_ALIASES)
FONT_FILE_EXTENSIONS = ('.ttf', '.ttc', '.otf')


def _app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _existing_font_file(path):
    if os.path.isfile(path) and os.path.splitext(path)[1].lower() in FONT_FILE_EXTENSIONS:
        return os.path.abspath(path)
    return None


def _font_file_names(font):
    if os.path.splitext(font)[1]:
        return [font]
    return [font + extension for extension in FONT_FILE_EXTENSIONS]


def _resolve_font_file(font, search_local_dirs=False):
    expanded = os.path.expandvars(os.path.expanduser(font))
    dirname = os.path.dirname(expanded)

    if dirname or os.path.isabs(expanded):
        for candidate in _font_file_names(expanded):
            resolved = _existing_font_file(candidate)
            if resolved:
                return resolved
        return None

    resolved = _existing_font_file(expanded)
    if resolved:
        return resolved

    if not search_local_dirs:
        return None

    search_dirs = []
    for directory in (os.getcwd(), _app_dir()):
        directory = os.path.abspath(directory)
        if directory not in search_dirs:
            search_dirs.append(directory)

    for directory in search_dirs:
        for filename in _font_file_names(expanded):
            resolved = _existing_font_file(os.path.join(directory, filename))
            if resolved:
                return resolved
    return None


def _canvas_point(point, size):
    x, y = point
    return (round(x / 1024 * size), round((900 - y) / 1024 * size))


def _sample_line(start, end, size, steps=8):
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        points.append(_canvas_point((x, y), size))
    return points


def _sample_quadratic(start, control, end, size, steps=24):
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        x = mt * mt * start[0] + 2 * mt * t * control[0] + t * t * end[0]
        y = mt * mt * start[1] + 2 * mt * t * control[1] + t * t * end[1]
        points.append(_canvas_point((x, y), size))
    return points


def _sample_cubic(start, control1, control2, end, size, steps=32):
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1 - t
        x = (
            mt * mt * mt * start[0]
            + 3 * mt * mt * t * control1[0]
            + 3 * mt * t * t * control2[0]
            + t * t * t * end[0]
        )
        y = (
            mt * mt * mt * start[1]
            + 3 * mt * mt * t * control1[1]
            + 3 * mt * t * t * control2[1]
            + t * t * t * end[1]
        )
        points.append(_canvas_point((x, y), size))
    return points


def _path_to_polygons(path, size):
    tokens = PATH_TOKEN_RE.findall(path)
    polygons = []
    points = []
    command = None
    current = None
    start = None
    i = 0

    def read_point():
        nonlocal i
        point = (float(tokens[i]), float(tokens[i + 1]))
        i += 2
        return point

    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1

        if command == 'M':
            if len(points) >= 3:
                polygons.append(points)
            current = read_point()
            start = current
            points = [_canvas_point(current, size)]
            command = 'L'
        elif command == 'L':
            end = read_point()
            points.extend(_sample_line(current, end, size))
            current = end
        elif command == 'Q':
            control = read_point()
            end = read_point()
            points.extend(_sample_quadratic(current, control, end, size))
            current = end
        elif command == 'C':
            control1 = read_point()
            control2 = read_point()
            end = read_point()
            points.extend(_sample_cubic(current, control1, control2, end, size))
            current = end
        elif command == 'Z':
            if start:
                points.extend(_sample_line(current, start, size))
                current = start
            if len(points) >= 3:
                polygons.append(points)
            points = []
            command = None
        else:
            raise ValueError(f'unsupported SVG path command: {command}')

    if len(points) >= 3:
        polygons.append(points)
    return polygons


def _render_stroke(path, size=150):
    render_scale = 2 if size <= 512 else 1
    render_size = size * render_scale
    image = Image.new('L', (render_size, render_size), 255)
    draw = ImageDraw.Draw(image)
    for polygon in _path_to_polygons(path, render_size):
        draw.polygon(polygon, fill=0)
    if render_scale > 1:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _apply_font_style(image, font):
    size = image.size[0]

    def odd_kernel(ratio, minimum=3):
        value = max(minimum, int(round(size * ratio)))
        return value if value % 2 == 1 else value + 1

    if font == 'standard':
        return image
    if font == 'kaishu':
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, -0.10, max(1, round(size * 8 / 150)), 0, 1, 0),
            fillcolor=255,
        )
    if font == 'bold':
        return image.filter(ImageFilter.MinFilter(odd_kernel(5 / 150)))
    if font == 'thin':
        return image.filter(ImageFilter.MaxFilter(odd_kernel(3 / 150)))
    raise ValueError(f"unsupported font: {font}")


def resolve_font(font):
    if not font:
        return 'standard'

    font_key = font.lower()
    if font_key in FONT_STYLES:
        return font_key

    font_path = _resolve_font_file(font)
    if font_path:
        return font_path

    alias = SYSTEM_FONT_ALIASES.get(font_key)
    if alias:
        windows_dir = os.environ.get('WINDIR', r'C:\Windows')
        alias_path = os.path.join(windows_dir, 'Fonts', alias)
        if os.path.exists(alias_path):
            return alias_path

    font_path = _resolve_font_file(font, search_local_dirs=True)
    if font_path:
        return font_path

    fonts_dir = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
    for extension in FONT_FILE_EXTENSIONS:
        pattern = re.compile(r'.*', re.IGNORECASE)
        if not os.path.isdir(fonts_dir):
            break
        for filename in os.listdir(fonts_dir):
            if not filename.lower().endswith(extension):
                continue
            stem = os.path.splitext(filename)[0].lower()
            if stem == font_key:
                return os.path.join(fonts_dir, filename)
        if not pattern:
            break

    raise ValueError(
        f"unsupported font: {font}. Use --list-fonts to see built-in names, "
        "or pass a .ttf/.ttc/.otf file path/name."
    )


def _render_font_glyph(char, font_path, size=150):
    image = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(image)
    font_size = int(size * 0.86)

    while font_size > 12:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), char, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= size * 0.90 and height <= size * 0.90:
            break
        font_size -= 2

    x = (size - width) / 2 - bbox[0]
    y = (size - height) / 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    return image


def _custom_font_stroke(char, font_path, stroke_path, size=150):
    glyph = _render_font_glyph(char, font_path, size)
    kernel = max(3, min(41, int(round(size * 5 / 150))))
    if kernel % 2 == 0:
        kernel += 1
    stroke_region = _render_stroke(stroke_path, size).filter(ImageFilter.MinFilter(kernel))
    mask = ImageChops.lighter(glyph, stroke_region)
    if ImageChops.invert(mask).getbbox() is None:
        return _render_stroke(stroke_path, size)
    return mask


def _load_hanziwu_data(char, cache):
    code = hex(ord(char))[2:]
    json_path = os.path.join(cache, f"{code}.json")
    url = f"https://www.hanziwu.com/assets/bishun/json/{urllib.parse.quote(char)}.json"

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        return data


def char_data(word, cache='./cache'):
    if not os.path.exists(cache):
        os.makedirs(cache)
    return _load_hanziwu_data(word, cache)


def char_medians(word, cache='./cache'):
    return char_data(word, cache).get('medians', [])


def char_glyph_frame(word, cache='./cache', font='standard', size=150):
    resolved_font = resolve_font(font)
    if resolved_font not in FONT_STYLES:
        return _render_font_glyph(word, resolved_font, size)

    image = Image.new('L', (size, size), 255)
    for stroke in char_frames(word, cache=cache, vibe=False, font=font, size=size)[1:]:
        image = ImageChops.darker(image, stroke)
    return image


def char_frames(word, cache='./cache', vibe=True, font='standard', size=150):
    if not os.path.exists(cache):
        os.makedirs(cache)

    data = _load_hanziwu_data(word, cache)
    resolved_font = resolve_font(font)
    out = [Image.new('L', (size, size), 255)]

    for stroke_path in data.get('strokes', []):
        if resolved_font in FONT_STYLES:
            mask = _render_stroke(stroke_path, size)
            mask = _apply_font_style(mask, resolved_font)
        else:
            mask = _custom_font_stroke(word, resolved_font, stroke_path, size)
        if vibe:
            mask = mask.rotate(
                random.randint(-3, 3),
                fillcolor=255,
                translate=(random.randint(-1, 1), random.randint(-1, 1)),
            )
        out.append(mask)

    return out


def char_strokes(word):
    return STOKES_CODE[word]
