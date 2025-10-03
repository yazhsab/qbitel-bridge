#!/usr/bin/env python3
"""
Pre-commit hook to detect hardcoded credentials in Python files.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


# Patterns to detect hardcoded credentials
CREDENTIAL_PATTERNS = [
    # Hardcoded passwords
    (r'password\s*[:=]\s*["\'](?!.*\$\{)([^"\']{3,})["\']', 'Hardcoded password'),
    
    # API keys and tokens
    (r'api[_-]?key\s*[:=]\s*["\']([^"\']{16,})["\']', 'Hardcoded API key'),
    (r'token\s*[:=]\s*["\']([^"\']{16,})["\']', 'Hardcoded token'),
    
    # AWS credentials
    (r'aws[_-]?access[_-]?key[_-]?id\s*[:=]\s*["\']([A-Z0-9]{20})["\']', 'AWS Access Key'),
    (r'aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*["\']([A-Za-z0-9/+=]{40})["\']', 'AWS Secret Key'),
    
    # Database connection strings with embedded passwords
    (r'postgresql://[^:]+:([^@]+)@', 'Database password in connection string'),
    (r'mysql://[^:]+:([^@]+)@', 'Database password in connection string'),
    
    # JWT secrets
    (r'jwt[_-]?secret\s*[:=]\s*["\']([^"\']{8,})["\']', 'Hardcoded JWT secret'),
    
    # Encryption keys
    (r'encryption[_-]?key\s*[:=]\s*["\']([^"\']{8,})["\']', 'Hardcoded encryption key'),
    
    # Generic secrets
    (r'secret\s*[:=]\s*["\']([^"\']{8,})["\']', 'Hardcoded secret'),
]

# Allowed patterns (exceptions)
ALLOWED_PATTERNS = [
    r'password\s*[:=]\s*["\'][\'"]\s*#.*environment',  # Empty with comment about env var
    r'password\s*[:=]\s*os\.getenv',  # Loading from environment
    r'password\s*[:=]\s*get_secret',  # Loading from secrets manager
    r'password\s*[:=]\s*\$\{',  # Template variable
    r'#.*password.*example',  # Comment with example
    r'#.*password.*test',  # Comment for testing
    r'""".*password.*"""',  # Docstring
    r"'''.*password.*'''",  # Docstring
]

# Files to exclude
EXCLUDE_PATTERNS = [
    r'test_.*\.py$',
    r'.*_test\.py$',
    r'.*/tests/.*',
    r'.*/examples/.*',
    r'check_credentials\.py$',
]


def is_excluded(filepath: str) -> bool:
    """Check if file should be excluded from scanning."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, filepath):
            return True
    return False


def is_allowed_pattern(line: str) -> bool:
    """Check if line matches an allowed pattern."""
    for pattern in ALLOWED_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    return False


def check_file(filepath: str) -> List[Tuple[int, str, str]]:
    """
    Check a file for hardcoded credentials.
    
    Returns:
        List of (line_number, issue_type, line_content) tuples
    """
    issues = []
    
    if is_excluded(filepath):
        return issues
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            # Skip if line is allowed
            if is_allowed_pattern(line):
                continue
            
            # Check each credential pattern
            for pattern, issue_type in CREDENTIAL_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append((line_num, issue_type, line.strip()))
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
    
    return issues


def main():
    """Main function."""
    files_to_check = sys.argv[1:]
    
    if not files_to_check:
        print("No files to check")
        return 0
    
    total_issues = 0
    
    for filepath in files_to_check:
        if not filepath.endswith('.py'):
            continue
        
        issues = check_file(filepath)
        
        if issues:
            print(f"\n❌ {filepath}:")
            for line_num, issue_type, line_content in issues:
                print(f"  Line {line_num}: {issue_type}")
                print(f"    {line_content}")
            total_issues += len(issues)
    
    if total_issues > 0:
        print(f"\n{'='*80}")
        print(f"❌ Found {total_issues} potential hardcoded credential(s)")
        print(f"{'='*80}")
        print("\nSecurity Best Practices:")
        print("1. Use environment variables for sensitive data")
        print("2. Use secrets management (Vault, AWS Secrets Manager, Azure Key Vault)")
        print("3. Never commit credentials to version control")
        print("4. Use configuration files with empty/placeholder values")
        print("\nExample:")
        print('  password: str = ""  # Set via DATABASE_PASSWORD environment variable')
        print('  password = os.getenv("DATABASE_PASSWORD")')
        print('  password = get_secret("database_password")')
        return 1
    
    print("✅ No hardcoded credentials detected")
    return 0


if __name__ == '__main__':
    sys.exit(main())