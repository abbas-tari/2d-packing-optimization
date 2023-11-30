import logging

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(filename='app.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

def log_header(message: str):
    """Log and print a header message."""
    header = "#" * 40
    centered_message = message.center(40)
    for output in [logging.info, print]:
        output(header)
        output(centered_message)
        output(header)

def log_message(label: str, data: str):
    """Log and print a formatted message."""
    formatted_message = f"{label:<30} : {data:<10}"
    logging.info(formatted_message)
    print(formatted_message)
