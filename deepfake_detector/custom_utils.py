import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )