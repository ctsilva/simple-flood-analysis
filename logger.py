#!/usr/bin/env python3
"""
Logging utilities for flood analysis toolkit.

This module provides centralized logging configuration and utilities
for consistent logging across all modules in the toolkit.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional
from config import LoggingConfig, EnvironmentConfig


class FloodAnalysisLogger:
    """
    Centralized logger for flood analysis toolkit.
    
    Provides consistent logging configuration and utilities for all modules.
    """
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def setup_logging(cls, 
                     log_level: Optional[str] = None,
                     log_file: Optional[Path] = None,
                     console_output: bool = True) -> None:
        """
        Configure logging for the entire toolkit.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            console_output: Whether to output logs to console
        """
        if cls._configured:
            return
        
        # Determine log level
        if log_level is None:
            log_level = LoggingConfig.get_log_level()
        
        # Determine log file
        if log_file is None:
            if not EnvironmentConfig.is_debug_mode():
                log_file = LoggingConfig.get_log_file()
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            LoggingConfig.DEFAULT_LOG_FORMAT,
            LoggingConfig.DEFAULT_DATE_FORMAT
        )
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            try:
                # Ensure log directory exists
                log_file.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)  # Always log everything to file
                root_logger.addHandler(file_handler)
                
            except Exception as e:
                # If file logging fails, at least log to console
                if console_output:
                    logging.warning(f"Could not set up file logging to {log_file}: {e}")
        
        cls._configured = True
        
        # Log the configuration
        logger = cls.get_logger("FloodAnalysisLogger")
        logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific module.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.setup_logging()
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def log_function_entry(cls, logger: logging.Logger, func_name: str, **kwargs) -> None:
        """
        Log function entry with parameters.
        
        Args:
            logger: Logger instance
            func_name: Function name
            **kwargs: Function parameters to log
        """
        if kwargs:
            params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            logger.debug(f"Entering {func_name}({params})")
        else:
            logger.debug(f"Entering {func_name}()")
    
    @classmethod
    def log_function_exit(cls, logger: logging.Logger, func_name: str, result: any = None) -> None:
        """
        Log function exit with optional result.
        
        Args:
            logger: Logger instance
            func_name: Function name
            result: Function result to log (optional)
        """
        if result is not None:
            logger.debug(f"Exiting {func_name} -> {result}")
        else:
            logger.debug(f"Exiting {func_name}")
    
    @classmethod
    def log_progress(cls, logger: logging.Logger, current: int, total: int, operation: str = "") -> None:
        """
        Log progress information.
        
        Args:
            logger: Logger instance
            current: Current progress count
            total: Total items to process
            operation: Description of operation
        """
        percentage = (current / total * 100) if total > 0 else 0
        msg = f"Progress: {current:,}/{total:,} ({percentage:.1f}%)"
        if operation:
            msg = f"{operation} - {msg}"
        logger.info(msg)
    
    @classmethod
    def log_memory_usage(cls, logger: logging.Logger, description: str = "") -> None:
        """
        Log current memory usage.
        
        Args:
            logger: Logger instance
            description: Optional description of the operation
        """
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            msg = f"Memory usage: {memory_mb:.1f} MB"
            if description:
                msg = f"{description} - {msg}"
            
            # Warn if memory usage is high
            max_memory = EnvironmentConfig.get_max_memory_mb()
            if memory_mb > max_memory:
                logger.warning(f"{msg} (exceeds threshold of {max_memory} MB)")
            else:
                logger.debug(msg)
                
        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            logger.debug(f"Could not check memory usage: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return FloodAnalysisLogger.get_logger(name)


def setup_logging(log_level: Optional[str] = None,
                 log_file: Optional[Path] = None,
                 console_output: bool = True) -> None:
    """
    Convenience function to setup logging.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
    """
    FloodAnalysisLogger.setup_logging(log_level, log_file, console_output)


# Auto-configure logging on import
if not FloodAnalysisLogger._configured:
    FloodAnalysisLogger.setup_logging()