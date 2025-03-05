# tests/conftest.py
import sys
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine.*never awaited")
# Add project root to path
sys.path.insert(0, '.')
# Chonkie warnings
warnings.filterwarnings("ignore", message=".*chunk_overlap is getting deprecated.*", module="chonkie.*")

# LiteLLM warnings
warnings.filterwarnings("ignore", message=".*open_text is deprecated.*", module="litellm.*")

# Coroutine warnings
warnings.filterwarnings("ignore", message=".*coroutine.*never awaited.*")