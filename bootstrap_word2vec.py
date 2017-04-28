from argparse import ArgumentParser

import sys

import logging

from data.options import Options
from processors.word2vec_processor import Word2VecProcessor

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='[%(asctime)s]: %(name)s : %(levelname)s : %(message)s'
)
log = logging.getLogger(__name__)


def main(argv):
    options = parse_args(argv)
    processor = Word2VecProcessor(options)
    processor.process()


def parse_args(argv):
    parser = ArgumentParser(prog="wassa-task")
    parser.add_argument('--training_data_file_path', metavar='Train Data File Path',
                        type=str, required=True)
    parser.add_argument('--test_data_file_path', metavar='Test Data File Path',
                        type=str, required=False)
    parser.add_argument('--wv_model_path', metavar='Word2Vec File Path',
                        type=str, required=True)

    return parser.parse_args(argv, namespace=Options())


if __name__ == "__main__":
    main(sys.argv[1:])
