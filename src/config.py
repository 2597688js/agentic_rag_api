import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for loading and accessing YAML configuration."""

    _config = None

    def __init__(self, config_path: Optional[str] = None):
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None):
        """Load configuration from YAML file and override with environment variables."""
        if config_path is None:
            # Default to config.yaml in the config folder (next to this file)
            current_dir = Path(__file__).parent
            config_path = current_dir / "config" / "config.yaml"
            logger.info(f"Config file loaded from {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
                logger.info(f"Configuration loaded successfully from {config_path}")
            
            # Override with environment variables
            self._override_with_env_vars()
            
            return self._config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise

    def _override_with_env_vars(self):
        """Override configuration values with environment variables."""
        # OpenAI configuration - API key is required
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required. Please set it before running the application.")
        self._config['openai']['api_key'] = openai_api_key
        
        # API configuration
        if os.getenv('ENVIRONMENT'):
            self._config['api']['environment'] = os.getenv('ENVIRONMENT')
        if os.getenv('LOG_LEVEL'):
            self._config['api']['log_level'] = os.getenv('LOG_LEVEL')
        if os.getenv('CORS_ORIGINS'):
            self._config['api']['cors_origins'] = os.getenv('CORS_ORIGINS').split(',')
        if os.getenv('ALLOWED_HOSTS'):
            self._config['api']['allowed_hosts'] = os.getenv('ALLOWED_HOSTS').split(',')
        
        logger.info("Configuration overridden with environment variables")

    def get_value(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value by section and key."""
        try:
            return self._config.get(section, {}).get(key, default)
        except Exception as e:
            logger.error(f"Error getting config value for {section}.{key}: {e}")
            return default

# # Global instance
# config = ConfigManager()

# print(config._config)