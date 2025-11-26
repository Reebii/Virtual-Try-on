import logging
import sys
from pathlib import Path

class UnicodeSafeFormatter(logging.Formatter):
    """Custom formatter that handles Unicode characters safely."""
    
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # Remove or replace problematic Unicode characters
            record.msg = self._safe_message(record.msg)
            return super().format(record)
    
    def _safe_message(self, message):
        """Replace Unicode emojis with text equivalents."""
        replacements = {
            '✅': '[OK]',
            '🔍': '[SEARCH]',
            '🎨': '[ART]',
            '🍌': '[BANANA]',
            '💾': '[SAVE]',
            '💰': '[MONEY]',
            '⚠️': '[WARN]',
            '❌': '[ERROR]',
            '⏱️': '[TIME]',
            '🛡️': '[SHIELD]',
            '💳': '[CARD]'
        }
        
        if isinstance(message, str):
            for emoji, text in replacements.items():
                message = message.replace(emoji, text)
        return message

def setup_logger():
    """Setup logger with Unicode-safe formatting."""
    logger = logging.getLogger('virtual_tryon')
    logger.setLevel(logging.INFO)
    
    # Create formatter with safe Unicode handling
    formatter = UnicodeSafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = Path('virtual_tryon.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()