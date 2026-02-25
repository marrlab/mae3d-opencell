"""
Logging utilities for saving training logs to files.
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime


class TeeLogger:
    """
    Logger that writes to both console and file.
    """
    def __init__(self, log_file, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_file, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logger(output_dir, log_filename='train.log', rank=0):
    """
    Setup logging to both console and file.

    Args:
        output_dir: Directory to save log file
        log_filename: Name of log file
        rank: Process rank (only rank 0 creates logs)

    Returns:
        logger: Logger instance
        log_file_path: Path to log file
    """
    # Only rank 0 creates logs
    if rank != 0:
        return None, None

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = output_dir / f"{log_filename.replace('.log', '')}_{timestamp}.log"

    # Setup Python logging
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file_path}")
    logger.info("="*80)

    return logger, str(log_file_path)


def redirect_stdout_to_file(output_dir, log_filename='train_output.log', rank=0):
    """
    Redirect stdout to both console and file using Tee.

    Args:
        output_dir: Directory to save log file
        log_filename: Name of log file
        rank: Process rank (only rank 0 creates logs)

    Returns:
        tee_logger: TeeLogger instance (or None for non-rank-0 processes)
    """
    # Only rank 0 creates logs
    if rank != 0:
        return None

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = output_dir / f"{log_filename.replace('.log', '')}_{timestamp}.log"

    # Create tee logger
    tee = TeeLogger(log_file_path, mode='w')
    sys.stdout = tee
    sys.stderr = tee

    print(f"[Logger] Redirecting stdout/stderr to {log_file_path}")
    print("="*80)

    return tee


def save_config_to_file(config, output_dir, filename='config.yaml', rank=0):
    """
    Save configuration to a YAML file in output_dir.

    Args:
        config: OmegaConf config object
        output_dir: Directory to save config
        filename: Name of config file
        rank: Process rank (only rank 0 saves)
    """
    if rank != 0:
        return

    from omegaconf import OmegaConf

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / filename
    OmegaConf.save(config, config_path)
    print(f"[Logger] Saved config to {config_path}")
