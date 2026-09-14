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

def clean_dict_for_csv(d):
    """Filters dictionary values to ensure only serializable scalars are saved to CSV."""
    clean_d = {}
    for k, v in d.items():
        if hasattr(v, 'item'):  # PyTorch tensor or NumPy scalar
            try:
                clean_d[k] = v.item()
            except Exception:
                continue
        elif isinstance(v, (int, float, str, bool)) or v is None:
            clean_d[k] = v
    return clean_d

def save_dict_to_csv(data_dict_list, output_csv_path, mode='a'):
    """Saves a list of dictionaries to a CSV file robustly, ignoring tensors and non-scalars."""
    if not data_dict_list:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    clean_list = [clean_dict_for_csv(d) for d in data_dict_list]
    
    # Determine all fieldnames preserving order
    fieldnames = []
    seen = set()
    for d in clean_list:
        for k in d.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    file_exists = os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0
    
    if file_exists and mode == 'a':
        try:
            with open(output_csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            for h in existing_header:
                if h not in seen:
                    seen.add(h)
                    fieldnames.append(h)
        except Exception:
            pass

    write_header = (not file_exists) or (mode == 'w')
    with open(output_csv_path, mode=mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        for row in clean_list:
            writer.writerow(row)

def read_csv_to_dict_list(csv_path):
    """Reads an existing CSV file into a list of dictionaries for checkpoint/resume."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return []
    rows = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    try:
                        if '.' in v:
                            parsed[k] = float(v)
                        else:
                            parsed[k] = int(v)
                    except (ValueError, TypeError):
                        parsed[k] = v
                rows.append(parsed)
    except Exception as e:
        print(f"[Warning] Failed to read {csv_path}: {e}")
    return rows
