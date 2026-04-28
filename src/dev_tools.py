"""
This module provides development tools for GodDev, including JavaScript syntax validation.
It leverages external tools like Node.js and ESLint to perform comprehensive checks,
ensuring code quality and correctness within the development lifecycle.
"""

import subprocess
import json
import os
import sys

def validate_js_syntax(js_code: str) -> dict:
    """
    Validates JavaScript code syntax using ESLint via a Node.js subprocess.

    This function executes a Node.js script that uses the ESLint CLIEngine
    to lint the provided JavaScript code string. It passes the JavaScript
    code as a command-line argument to the Node.js script.

    It returns a dictionary indicating whether the code is valid according
    to ESLint's 'error' level rules and a list of any detected error messages.

    Args:
        js_code: A string containing the JavaScript code to validate.

    Returns:
        A dictionary with two keys:
        - 'valid': A boolean indicating if the JavaScript code is syntactically valid
                   and passes ESLint checks at 'error' severity (True for no errors, False otherwise).
        - 'errors': A list of strings, where each string is an ESLint error message
                    or a diagnostic message from the validation process.

    Raises:
        RuntimeError: If Node.js or ESLint is not found or fails to execute,
                      or if there's an unexpected error during subprocess execution.
    """
    node_script_path = os.path.join(os.path.dirname(__file__), 'eslint_runner.js')
    
    try:
        with open(node_script_path, 'r') as f:
            node_script_template = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Node.js script file not found at {node_script_path}")

    node_exe = os.getenv('NODE_BIN', 'node')
    
    try:
        cmd = [node_exe, '-e', node_script_template, js_code]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30
        )
        
        try:
            eslint_results = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"Failed to parse ESLint output: {result.stdout}")
        
        errors = []
        for file_result in eslint_results:
            for message in file_result.get('messages', []):
                if message.get('severity') == 2:
                    errors.append(f"Line {message.get('line', '?')}: {message.get('message', 'Unknown error')}")
        
        valid = len(errors) == 0
        
        return {
            'valid': valid,
            'errors': errors
        }
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Node.js script execution timed out after 30 seconds")
    except FileNotFoundError:
        raise RuntimeError(f"Node.js executable '{node_exe}' not found. Please ensure Node.js is installed.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during JavaScript validation: {str(e)}")