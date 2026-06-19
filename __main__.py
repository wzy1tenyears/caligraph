import argparse
import codecs
import os
import re
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from loguru import logger

from animation import animation
from strokes import FONT_CHOICES, FONT_STYLES, SYSTEM_FONT_ALIASES, resolve_font


CODEPOINT_RE = re.compile(r'^(?:U\+|0x)?([0-9a-fA-F]{4,6})$')


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_usage(self):
        return super().format_usage().replace("usage:", "用法:")

    def format_help(self):
        return (
            super()
            .format_help()
            .replace("usage:", "用法:")
            .replace("positional arguments:", "位置参数:")
            .replace("options:", "选项:")
            .replace("show this help message and exit", "显示帮助信息并退出")
        )

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 错误: {message}\n")


def configure_console():
    if os.name == "nt":
        os.system("chcp 65001 > nul")
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} 不是整数") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("数值必须大于 0")
    return number


def positive_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} 不是数字") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("数值必须大于 0")
    return number


def build_parser():
    parser = ChineseArgumentParser(
        prog="caligraph",
        description="生成汉字笔顺书写动画。",
    )
    parser.add_argument(
        "text",
        nargs="*",
        metavar="文本",
        help="要生成动画的汉字文本。多个参数会自动拼接。",
    )
    parser.add_argument(
        "-t",
        "--text",
        dest="text_option",
        default=None,
        metavar="文本",
        help="要生成动画的汉字文本，也支持 \\u6c38 或 U+6C38 写法。",
    )
    parser.add_argument(
        "-o",
        "--output",
        "-fp",
        "--filepath",
        dest="output",
        default=None,
        metavar="路径",
        help="主输出路径。默认是 <文本>.mp4。",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        "-w",
        "--width",
        dest="resolution",
        type=positive_int,
        default=150,
        metavar="像素",
        help="每个汉字方格的像素尺寸。默认：150。",
    )
    parser.add_argument(
        "-f",
        "--font",
        default="standard",
        metavar="字体",
        help="字体/样式名称，或 .ttf/.ttc/.otf 字体文件路径。默认：standard。",
    )
    parser.add_argument(
        "--format",
        choices=("mp4", "gif", "both"),
        default="both",
        metavar="格式",
        help="输出格式：mp4、gif 或 both。默认：both。",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=30,
        metavar="帧率",
        help="输出视频/GIF 的帧率。默认：30。",
    )
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        metavar="倍速",
        help="动画速度倍数。2 表示两倍速，0.5 表示半速。默认：1。",
    )
    parser.add_argument(
        "--list-fonts",
        action="store_true",
        help="列出可用字体/样式后退出。",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="进入交互模式，逐项输入文本、分辨率、字体、帧率、速度、格式和输出路径。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="生成时显示调试日志。",
    )
    return parser


def decode_text_value(value):
    value = value.strip()
    if not value:
        return value

    codepoint_parts = re.split(r'[\s,;]+', value)
    if codepoint_parts and all(CODEPOINT_RE.match(part) for part in codepoint_parts):
        return ''.join(chr(int(CODEPOINT_RE.match(part).group(1), 16)) for part in codepoint_parts)

    if "\\u" in value or "\\U" in value:
        try:
            return codecs.decode(value, "unicode_escape")
        except UnicodeDecodeError:
            return value

    return value


def resolve_text(args):
    if args.text_option:
        return decode_text_value(args.text_option)

    if args.text:
        joined = " ".join(args.text) if any(CODEPOINT_RE.match(part) for part in args.text) else "".join(args.text)
        return decode_text_value(joined)

    text = input("请输入汉字文本：").strip()
    if not text:
        raise SystemExit("未输入文本。")
    return decode_text_value(text)


def prompt_required(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("不能为空，请重新输入。")


def prompt_choice(prompt, choices, default, allow_custom=True):
    while True:
        print(prompt)
        for index, (label, value) in enumerate(choices, start=1):
            suffix = " [默认]" if value == default else ""
            print(f"  {index}. {label}{suffix}")
        custom_hint = "，输入 c 可自定义" if allow_custom else ""
        answer = input(f"请选择编号，直接回车使用默认值{custom_hint}：").strip()
        if not answer:
            return default
        if answer.lower() == "c" and allow_custom:
            custom = prompt_required("请输入自定义值：")
            return custom
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(choices):
                return choices[index - 1][1]
        print("无效选择，请重新输入。")


def prompt_positive_int(prompt, default):
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            return positive_int(answer)
        except argparse.ArgumentTypeError as exc:
            print(exc)


def prompt_positive_float(prompt, default):
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            return positive_float(answer)
        except argparse.ArgumentTypeError as exc:
            print(exc)


def prompt_interactive(args):
    print("caligraph 交互模式")
    args.text_option = prompt_required("汉字文本（可输入中文、\\u6c38 或 U+6C38）：")
    args.text = []

    resolution = prompt_choice(
        "每个汉字的分辨率：",
        [
            ("72x72", "72"),
            ("96x96", "96"),
            ("150x150", "150"),
            ("256x256", "256"),
        ],
        "150",
    )
    args.resolution = positive_int(resolution)

    args.font = prompt_choice(
        "字体/样式：",
        [
            ("标准", "standard"),
            ("楷书风格", "kaishu"),
            ("加粗风格", "bold"),
            ("细体风格", "thin"),
            ("楷体 simkai", "simkai"),
            ("黑体 simhei", "simhei"),
            ("宋体 simsun", "simsun"),
            ("微软雅黑 msyh", "msyh"),
        ],
        "standard",
    )

    args.fps = prompt_positive_int("帧率 FPS", args.fps)
    args.speed = prompt_positive_float("动画速度倍数", args.speed)
    args.format = prompt_choice(
        "输出格式：",
        [("MP4 视频", "mp4"), ("GIF 动图", "gif"), ("同时输出 MP4 和 GIF", "both")],
        args.format,
        allow_custom=False,
    )

    output = input("输出路径 [默认：<文本>.mp4]：").strip()
    args.output = output or None
    return args


def output_paths(text, output, output_format):
    default_base = Path(f"{text}.mp4")
    primary = Path(output) if output else default_base

    if output_format == "mp4":
        return primary.with_suffix(".mp4"), None
    if output_format == "gif":
        return None, primary.with_suffix(".gif")

    mp4_path = primary.with_suffix(".mp4")
    gif_path = primary.with_suffix(".gif")
    return mp4_path, gif_path


def speed_adjusted_frames(frames, speed):
    if speed == 1:
        yield from frames
        return

    if speed > 1:
        next_emit = 0.0
        last_frame = None
        emitted_last = False
        for index, frame in enumerate(frames):
            last_frame = frame
            if index >= next_emit:
                yield frame
                next_emit += speed
                emitted_last = True
            else:
                emitted_last = False
        if last_frame is not None and not emitted_last:
            yield last_frame
        return

    repeat = max(1, round(1 / speed))
    for frame in frames:
        for _ in range(repeat):
            yield frame


class StreamingGifWriter:
    def __init__(self, path, size, fps):
        self.writer = imageio_ffmpeg.write_frames(
            str(path),
            size,
            pix_fmt_in="gray",
            pix_fmt_out="rgb8",
            fps=float(fps),
            codec="gif",
            macro_block_size=1,
            output_params=["-loop", "0"],
            ffmpeg_log_level="error",
        )
        self.writer.send(None)

    def append_data(self, frame):
        self.writer.send(np.ascontiguousarray(frame))

    def close(self):
        self.writer.close()


def write_animation(text, mp4_path, gif_path, resolution, font, fps, speed):
    frame_size = (resolution * len(text), resolution)
    video_writer = None
    gif_writer = None

    try:
        if mp4_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), frame_size)
            if not video_writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {mp4_path}")

        if gif_path:
            gif_writer = StreamingGifWriter(gif_path, frame_size, fps)

        frames = animation(text, height=resolution, font=font)
        for key_frame in speed_adjusted_frames(frames, speed):
            if video_writer:
                video_writer.write(cv2.cvtColor(key_frame, cv2.COLOR_GRAY2BGR))
            if gif_writer:
                gif_writer.append_data(key_frame)
    finally:
        if video_writer:
            video_writer.release()
        if gif_writer:
            gif_writer.close()


def main():
    configure_console()
    parser = build_parser()
    args = parser.parse_args()

    if args.list_fonts:
        print("内置样式：")
        for font in FONT_STYLES:
            print(f"  {font}")
        print("Windows 字体别名：")
        for font in SYSTEM_FONT_ALIASES:
            try:
                resolved = resolve_font(font)
            except ValueError:
                continue
            print(f"  {font} -> {resolved}")
        print("自定义字体文件：")
        print("  可通过 --font 传入 .ttf、.ttc 或 .otf 文件路径")
        return

    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")

    if args.interactive or len(sys.argv) == 1:
        try:
            args = prompt_interactive(args)
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n已取消。")

    text = resolve_text(args)
    try:
        resolve_font(args.font)
    except ValueError as exc:
        parser.error(str(exc))

    mp4_path, gif_path = output_paths(text, args.output, args.format)

    print(f"文本：{text}")
    print(f"分辨率：每字 {args.resolution}x{args.resolution}")
    print(f"字体：{args.font}")
    print(f"帧率：{args.fps} FPS")
    print(f"速度：{args.speed}x")
    print("正在生成...")

    write_animation(text, mp4_path, gif_path, args.resolution, args.font, args.fps, args.speed)

    if mp4_path:
        print(f"MP4：{mp4_path}")
    if gif_path:
        print(f"GIF：{gif_path}")
    print("完成。")


if __name__ == "__main__":
    main()
