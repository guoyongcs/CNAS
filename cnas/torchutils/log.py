import os
import sys
import time
import typing
import logging
import argparse
import glob
import zipfile

import torch

from torch.utils.tensorboard import SummaryWriter

from .distributed import init, is_master, DummyClass

T = typing.TypeVar("T")


class LogExceptionHook(object):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(self, exc_type, exc_value, traceback):
        self.logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, traceback))


def get_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_directory", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def get_logger(name: str, output_directory: str, log_name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )
    if is_master():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if output_directory is not None:
            file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    logger.propagate = False
    return logger


def create_code_snapshot(name: str,
                         include_suffix: typing.List[str],
                         source_directory: str,
                         store_directory: str) -> None:
    if store_directory is None:
        return
    with zipfile.ZipFile(os.path.join(store_directory, "{}.zip".format(name)), "w") as f:
        for suffix in include_suffix:
            for file in glob.glob(os.path.join(source_directory, "**", "*{}".format(suffix)), recursive=True):
                f.write(file, os.path.join(name, file))


init()
args = get_args(sys.argv)
output_directory = args.output_directory
if is_master():
    if output_directory is not None:
        os.makedirs(args.output_directory, exist_ok=False)
    logger = get_logger("project", args.output_directory, "log.txt")
    sys.excepthook = LogExceptionHook(logger)
    create_code_snapshot("code", [".py"], ".", args.output_directory)
    summary_writer = SummaryWriter(args.output_directory)
else:
    logger = DummyClass()
    summary_writer = DummyClass()
