#!/usr/bin/env python3
"""
Installation and Configuration Validator for ResearchAgent
"""

import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.12 or higher."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print(f"   ⚠️  Python 3.12 or higher is required")
        return False


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("ResearchAgent - Installation & Configuration Validator")
    print("=" * 70)
    
    result = check_python_version()
    
    if result:
        print("\n🎉 Basic checks passed!")
        return 0
    else:
        print("\n⚠️  Some checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
