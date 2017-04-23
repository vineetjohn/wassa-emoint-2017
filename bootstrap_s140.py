from argparse import ArgumentParser

import sys

import logging

from data.options import Options
from processors.s140_processor import S140Processor

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='[%(asctime)s]: %(name)s : %(levelname)s : %(message)s'
)
log = logging.getLogger(__name__)


def main(argv):
    options = parse_args(argv)
    processor = S140Processor(options)
    processor.process()


def parse_args(argv):
    parser = ArgumentParser(prog="wassa-task")
    parser.add_argument('--input_file_path', metavar='Input File Path',
                        type=str, required=True)
    parser.add_argument('--lexicon_file_path', metavar='Lexicon File Path',
                        type=str, required=True)

    return parser.parse_args(argv, namespace=Options())


if __name__ == "__main__":
    main(sys.argv[1:])
