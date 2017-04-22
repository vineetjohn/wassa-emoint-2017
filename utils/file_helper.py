from data.tweet import Tweet
from utils import text_cleaner


def read_input_data(input_file_path):

    with open(input_file_path) as input_file:
        for line in input_file:
            line = line.strip()
            array = line.split('\t')
            yield Tweet(array[0], text_cleaner.clean_str(array[1]),
                        array[2], float(array[3]))
