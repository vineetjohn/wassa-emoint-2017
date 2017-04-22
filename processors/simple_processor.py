import logging

from processors.abstract_processor import Processor

log = logging.getLogger(__name__)


class SimpleProcessor(Processor):

    def process(self):
        log.info("SimpleProcessor begun")

        log.info("SimpleProcessor ended")
