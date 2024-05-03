import logging


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    # root_logger=logging.getLogger()
    # for handler in root_logger.handlers[:]:
    #     root_logger.removeHandler(handler)
    # for filter in root_logger.filters[:]:
    #     root_logger.removeFilter(filter)
    # formatter = logging.Formatter("%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter)
    # file_handler = logging.FileHandler('debug.log')  # log file name
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)    
    # root_logger.addHandler(console_handler)
    # root_logger.addHandler(file_handler)
    