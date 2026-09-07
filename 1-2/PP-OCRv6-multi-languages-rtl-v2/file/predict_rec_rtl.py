# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0

import os
import re
import sys
import unicodedata
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

import cv2
import numpy as np
from bidi.algorithm import get_display

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import check_and_read, get_image_file_list
from tools.infer.predict_rec import TextRecognizer


FSI = "\u2068"
PDI = "\u2069"
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z]+)*")
RTL_RE = re.compile(r"[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f]")
RTL_DISPLAY_RUN_RE = re.compile(
    r"[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+"
)
DISPLAY_BLOCK_RE = re.compile(
    r"\([A-Za-z0-9][A-Za-z0-9._%+\-]*\)"
    r"|[A-Za-z]+(?:[-_][A-Za-z]+)*"
    r"|\d+(?:[.,]\d+)?%?"
    r"|[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+"
    r"|[^\s]"
)
NO_SPACE_BEFORE = set(".,!?;:%؟،)]}\"'")
NO_SPACE_AFTER = set("([{\"'")
JOINERS = set("-‑־")


def has_rtl(text):
    return any(
        "\u0590" <= ch <= "\u05ff"
        or "\u0600" <= ch <= "\u06ff"
        or "\u0750" <= ch <= "\u077f"
        for ch in text
    )


def visual_to_logical(text):
    if not has_rtl(text):
        return text
    return clean_mixed_bidi_text(get_display(text, base_dir="L"))


def logical_to_readable_image_order(text):
    if not has_rtl(text):
        return text

    # For human-readable image order, reverse the whole sentence by blocks while
    # preserving the characters inside each block: Ticket, Wi-Fi, 24, and RTL words
    # stay readable, but their positions follow the RTL visual line.
    blocks = DISPLAY_BLOCK_RE.findall(text)
    out = []
    for block in reversed(blocks):
        if not out:
            out.append(block)
        elif block in NO_SPACE_BEFORE:
            out[-1] += block
        elif out[-1] in NO_SPACE_AFTER:
            out[-1] += block
        elif block in JOINERS or out[-1][-1:] in JOINERS:
            out[-1] += block
        else:
            out.append(" " + block)
    return "".join(out).strip()


def clean_mixed_bidi_text(text):
    text = re.sub(r"\s+([,.;:!?؟،])", r"\1", text)
    text = re.sub(r"([({\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)}\]])", r"\1", text)
    text = re.sub(r"([.!?؟،,;:])(?=[^\s.!?؟،,;:)\\]}])", r"\1 ", text)
    text = re.sub(r"(?<=[A-Za-z])(?={})".format(RTL_RE.pattern), " ", text)
    text = re.sub(r"(?<={})(?=[A-Za-z])".format(RTL_RE.pattern), " ", text)
    text = re.sub(r"(الـ?)\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"({})\s+-\s*([A-Za-z])".format(RTL_RE.pattern), r"\1-\2", text)
    text = re.sub(r"([A-Za-z])\s+-\s*({})".format(RTL_RE.pattern), r"\1-\2", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def normalize_for_metric(text):
    text = text.replace(FSI, "").replace(PDI, "")
    text = unicodedata.normalize("NFKC", text)
    for ch in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        text = text.replace(ch, "-")
    return " ".join(text.split())


def count_matched_chars(pred, gt):
    pred = normalize_for_metric(pred)
    gt = normalize_for_metric(gt)
    n, m = len(pred), len(gt)
    if m == 0:
        return 0, 1

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        pi = pred[i]
        row = dp[i]
        next_row = dp[i + 1]
        for j in range(m):
            if pi == gt[j]:
                value = row[j] + 1
            else:
                value = row[j + 1] if row[j + 1] >= next_row[j] else next_row[j]
            next_row[j + 1] = value
    return dp[n][m], m


def char_recall(pred, gt):
    matched, total = count_matched_chars(pred, gt)
    return matched / total if total else 0.0


def load_labels(label_path, image_dir):
    if not label_path:
        return {}
    labels = {}
    image_dir = Path(image_dir).resolve()
    path = Path(label_path)
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            name, label = line.split("\t", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            name, label = parts
        name = name.strip()
        label = label.strip()
        labels[name] = label
        labels[str(Path(name))] = label
        labels[Path(name).name] = label
        labels[str((image_dir / name).resolve())] = label
    return labels


def find_label(labels, image_file):
    image_path = Path(image_file)
    for key in [str(image_path), str(image_path.resolve()), image_path.name]:
        if key in labels:
            return labels[key]
    return None


def parse_args():
    parser = utility.init_args()
    parser.add_argument(
        "--save_res_path",
        type=str,
        default="./output/rec/predicts_rtl_postprocess.txt",
        help="Path to save text results.",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="",
        help="Optional label txt. Each line: image_name + whitespace/tab + label.",
    )
    parser.add_argument(
        "--print_visual",
        type=utility.str2bool,
        default=False,
        help="Print raw visual-order text in terminal.",
    )
    parser.add_argument(
        "--isolate_terminal",
        type=utility.str2bool,
        default=True,
        help="Wrap terminal text with Unicode isolate marks to avoid bidi spillover.",
    )
    parser.add_argument(
        "--terminal_order",
        type=str,
        default="image",
        choices=["image", "logical", "both"],
        help="Text printed in terminal. image reverses block order and keeps each block readable.",
    )
    parser.add_argument(
        "--simple_terminal",
        type=utility.str2bool,
        default=False,
        help="Print only prediction and confidence in terminal.",
    )
    return parser.parse_args()


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    valid_image_file_list = []
    img_list = []

    logger = get_logger()
    text_recognizer = TextRecognizer(args, logger=logger)
    labels = load_labels(args.label_path, args.image_dir)

    logger.info(
        "RTL postprocess is enabled: model visual-order output will be converted "
        "to logical text for metrics and image-order text for display."
    )

    if args.warmup:
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        for _ in range(2):
            text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info(f"error in loading image:{image_file}")
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)

    rec_res, _ = text_recognizer(img_list)

    save_path = Path(args.save_res_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    recalls = []
    scores = []

    for image_file, result in zip(valid_image_file_list, rec_res):
        visual_text, score = result[:2]
        logical_text = visual_to_logical(visual_text)
        image_order_text = logical_to_readable_image_order(logical_text)
        gt_text = find_label(labels, image_file)
        recall = char_recall(logical_text, gt_text) if gt_text is not None else None
        scores.append(float(score))
        if recall is not None:
            recalls.append(recall)

        lines.append(f"image: {image_file}\n")
        lines.append(f"predict: {logical_text}\n")
        if args.print_visual:
            lines.append(f"visual: {visual_text}\n")
        if gt_text is not None:
            lines.append(f"label: {gt_text}\n")
            lines.append(f"recall: {recall:.6f}\n")
        lines.append(f"confidence: {float(score):.6f}\n")
        lines.append("\n")

        logical_for_terminal = FSI + logical_text + PDI if args.isolate_terminal else logical_text
        image_for_terminal = FSI + image_order_text + PDI if args.isolate_terminal else image_order_text
        if args.simple_terminal:
            print(f"predict: {logical_for_terminal}")
            print(f"confidence: {float(score):.6f}")
        elif args.print_visual:
            visual_for_terminal = (
                FSI + visual_text + PDI if args.isolate_terminal else visual_text
            )
            logger.info(
                "Predicts of {}: image_order='{}', logical='{}', visual='{}', score={:.6f}".format(
                    image_file,
                    image_for_terminal,
                    logical_for_terminal,
                    visual_for_terminal,
                    float(score),
                )
            )
        else:
            if args.terminal_order == "logical":
                text_for_terminal = logical_for_terminal
            elif args.terminal_order == "both":
                text_for_terminal = (
                    f"image_order='{image_for_terminal}', "
                    f"logical='{logical_for_terminal}'"
                )
            else:
                text_for_terminal = image_for_terminal

            if args.terminal_order == "both":
                msg = "Predicts of {}: {}, score={:.6f}".format(
                    image_file, text_for_terminal, float(score)
                )
            else:
                msg = "Predicts of {}: ('{}', {:.6f})".format(
                    image_file,
                    text_for_terminal,
                    float(score),
                )
            if recall is not None:
                msg += ", recall={:.6f}".format(recall)
            logger.info(msg)

    if scores:
        lines.append(f"average_confidence: {sum(scores) / len(scores):.6f}\n")
    if recalls:
        lines.append(f"average_recall: {sum(recalls) / len(recalls):.6f}\n")

    save_path.write_text("".join(lines), encoding="utf-8")
    logger.info(f"RTL postprocessed results are saved to {save_path}")


if __name__ == "__main__":
    main(parse_args())
