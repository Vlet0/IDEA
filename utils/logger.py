import os
import sys
import logging
import csv
from datetime import datetime

def setup_logger(output_dir, name="experiment"):
    """Configures console and file logger."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def save_dict_to_csv(data_dict_list, output_csv_path):
    """Saves a list of dictionaries to a CSV file."""
    if not data_dict_list:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    fieldnames = list(data_dict_list[0].keys())
    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in data_dict_list:
            writer.writerow(row)
