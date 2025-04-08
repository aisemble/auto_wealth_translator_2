"""
Test script for DeepL API integration.
"""

import sys
print(f"Python path: {sys.path}")

try:
    import deepl
    print("DeepL module imported successfully!")
    
    # Test initialization of the translator
    translator = deepl.Translator("test_auth_key")
    print("DeepL Translator class initialized successfully!")

except ImportError as e:
    print(f"Failed to import DeepL: {str(e)}")
except Exception as e:
    print(f"Error occurred: {str(e)}") 