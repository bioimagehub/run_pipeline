#!/usr/bin/env python3
"""
CLI Definition Generator for BIPHUB Pipeline Designer

This intelligent tool generates JSON CLI definition files from Python scripts by:
1. Parsing argparse help output
2. Loading definitions from old YAML pipeline configs
3. Setting outputFilesDefined=False to defer output discovery to GUI runtime

DESIGN PHILOSOPHY:
- We're building a GUI to AVOID command-line prompts
- Therefore, we don't ask for test inputs during generation
- Output files are auto-discovered when nodes run in the GUI for the first time
- The GUI will show: "Output files not yet defined. Run this node to auto-discover outputs."

Usage:
    python generate_cli_definition.py --script path/to/script.py --no-interactive
    python generate_cli_definition.py --from-yaml path/to/config.yaml --no-interactive
    python generate_cli_definition.py --scan-missing

Author: BIPHUB
License: MIT
"""

import argparse
import subprocess
import json
import sys
import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CLIDefinitionGenerator:
    """Generate CLI definition JSON files from Python scripts."""
    
    # Standard category mappings
    CATEGORY_PATTERNS = {
        'Segmentation': ['segment', 'mask', 'threshold'],
        'Image Processing': ['convert', 'filter', 'drift', 'merge', 'transform'],
        'Measurement': ['measure', 'quantif', 'distance', 'calculate'],
        'Tracking': ['track'],
        'Visualization': ['plot', 'roi', 'visual'],
        'Input/Output': ['file', 'selector', 'copy', 'delete', 'rename'],
    }
    
    # Standard colors by category
    CATEGORY_COLORS = {
        'Segmentation': '#c586c0',
        'Image Processing': '#569cd6',
        'Measurement': '#4ec9b0',
        'Tracking': '#ce9178',
        'Visualization': '#f48771',
        'Input/Output': '#4ec9b0',
        'Utilities': '#858585',
    }
    
    # Standard icons by category
    CATEGORY_ICONS = {
        'Segmentation': '🔬',
        'Image Processing': '🎨',
        'Measurement': '📊',
        'Tracking': '📈',
        'Visualization': '📉',
        'Input/Output': '📁',
        'Utilities': '🔧',
    }
    
    def __init__(self, repo_root: str = None):
        """Initialize the generator."""
        self.repo_root = repo_root or self._find_repo_root()
        
    def _find_repo_root(self) -> str:
        """Find the repository root by looking for run_pipeline.go."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / 'run_pipeline.go').exists():
                return str(parent)
        return str(Path.cwd())
    
    def parse_help_output(self, script_path: str, environment: str = None) -> Dict:
        """
        Parse argparse help output from a Python script.
        
        Args:
            script_path: Path to the Python script
            environment: Optional UV environment to run in (e.g., "uv@3.11:convert_to_tif")
        
        Returns:
            Dict with parsed argument information
        """
        logger.info(f"Parsing help output from: {script_path}")
        
        try:
            # Construct command based on environment
            if environment and environment.startswith('uv@'):
                # Parse UV environment format: uv@3.11:convert_to_tif
                parts = environment.split(':')
                if len(parts) == 2:
                    env_name = parts[1]
                    # Use uv run with the specified environment
                    cmd = ['uv', 'run', '--with', env_name, 'python', script_path, '--help']
                    logger.info(f"Running with UV environment: {env_name}")
                else:
                    # Fallback to regular python if format is wrong
                    cmd = ['python', script_path, '--help']
            else:
                # No environment specified, use current python
                cmd = ['python', script_path, '--help']
            
            # Try to run the script with --help
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # Increased timeout for UV environment setup
            )
            
            if result.returncode != 0:
                logger.warning(f"Script exited with code {result.returncode}")
                return self._parse_help_text(result.stdout + result.stderr)
            
            return self._parse_help_text(result.stdout)
            
        except subprocess.TimeoutExpired:
            logger.error("Script took too long to respond to --help")
            return {}
        except Exception as e:
            logger.error(f"Error running script: {e}")
            return {}
    
    def _parse_help_text(self, help_text: str) -> Dict:
        """Parse argparse help text into structured data."""
        arguments = []
        
        # Split into lines and find argument definitions
        lines = help_text.split('\n')
        current_arg = None
        in_options_section = False
        
        for line in lines:
            # Detect when we enter the options/arguments section
            if line.strip().lower() in ['options:', 'optional arguments:', 'positional arguments:']:
                in_options_section = True
                continue
            
            # Stop if we hit a new section (like examples, environment variables, etc.)
            if in_options_section and line and not line[0].isspace() and ':' in line:
                # New section header detected
                break
            
            # Match argument flags - improved regex to handle modern argparse format
            # Matches: "  --flag VALUE" or "  --flag" or "  -f, --flag VALUE"
            # Key insight: metavar (VALUE/PATTERN) comes RIGHT after flag with 1-2 spaces max
            # Descriptions come after MANY spaces (typically 13+ spaces for alignment)
            flag_match = re.match(r'\s+(-\w|--[\w-]+)(?:,?\s+(--[\w-]+))?\s{0,2}([A-Z_]+)?\s*(.*)?', line)
            
            if flag_match and in_options_section:
                # Save previous argument
                if current_arg:
                    # Skip help flag
                    if current_arg['flag'] not in ['-h', '--help']:
                        arguments.append(current_arg)
                
                short_flag = flag_match.group(1)
                long_flag = flag_match.group(2)
                metavar = flag_match.group(3) or ''  # The VALUE/PATTERN part
                rest = flag_match.group(4) or ''
                
                # Clean up description - remove leading metavar if it appears
                if metavar and rest.strip().startswith(metavar):
                    rest = rest.strip()[len(metavar):].strip()
                
                # Prefer long flag if available
                flag = long_flag if long_flag else short_flag
                
                # Skip help flags
                if flag in ['-h', '--help']:
                    current_arg = None
                    continue
                
                # Parse type and default from description
                arg_type = 'string'
                default = ''
                required = False
                
                # Check the whole line for required/default indicators
                full_line_lower = line.lower()
                if 'required' in full_line_lower:
                    required = True
                
                # Look for default value in description
                default_match = re.search(r'\(default:\s*([^)]+)\)', rest, re.IGNORECASE)
                if default_match:
                    default = default_match.group(1).strip()
                
                # Infer type from flag name and metavar
                flag_lower = flag.lower()
                
                # Check for action flags (store_true, store_false)
                is_action_flag = (not metavar and 
                                 ('deprecated' in rest.lower() or 
                                  'enable' in rest.lower() or 
                                  'disable' in rest.lower() or
                                  flag.startswith('--no-') or
                                  'action' in rest.lower()))
                
                if is_action_flag:
                    arg_type = 'bool'
                elif 'folder' in flag_lower or 'dir' in flag_lower:
                    arg_type = 'path'
                elif 'pattern' in flag_lower:
                    arg_type = 'glob_pattern'
                elif 'path' in flag_lower:
                    arg_type = 'path'
                elif 'extension' in flag_lower:  # File extensions are strings, not paths
                    arg_type = 'string'
                elif 'file' in flag_lower and 'files' not in flag_lower and metavar:
                    arg_type = 'path'
                elif '{' in rest and '}' in rest:  # Choices like {None,max,sum}
                    arg_type = 'string'  # Dropdown/choice type
                elif not metavar:  # No value expected - it's a boolean flag
                    arg_type = 'bool'
                elif 'INT' in metavar or 'int' in rest.lower():
                    arg_type = 'int'
                elif 'FLOAT' in metavar or 'float' in rest.lower():
                    arg_type = 'float'
                
                current_arg = {
                    'flag': flag,
                    'type': arg_type,
                    'required': required,
                    'default': default,
                    'description': rest.strip()
                }
            elif current_arg and line.strip() and in_options_section:
                # Continuation of description (indented line)
                if line.startswith('  ') and not line.strip().startswith('-'):
                    # Clean up the continuation line
                    continuation = line.strip()
                    if current_arg['description']:
                        current_arg['description'] += ' ' + continuation
                    else:
                        current_arg['description'] = continuation
        
        # Don't forget the last argument
        if current_arg and current_arg['flag'] not in ['-h', '--help']:
            arguments.append(current_arg)
        
        return {'arguments': arguments}
    
    def load_from_yaml_config(self, yaml_path: str, script_basename: str = None) -> Optional[Dict]:
        """
        Load CLI definition from an old YAML pipeline config.
        
        Args:
            yaml_path: Path to YAML config file
            script_basename: Optional script basename to search for (e.g., "convert_to_tif.py")
        
        Returns:
            Dict with CLI definition or None if not found
        """
        logger.info(f"Loading from YAML config: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'run' not in config:
            logger.warning("No 'run' section found in YAML")
            return None
        
        for step in config['run']:
            commands = step.get('commands', [])
            
            # Find the script path in commands
            script_cmd = None
            for i, cmd in enumerate(commands):
                if isinstance(cmd, str) and cmd.endswith('.py'):
                    script_cmd = cmd
                    break
            
            if not script_cmd:
                continue
            
            # Check if this matches the requested script
            if script_basename:
                if Path(script_cmd).name != script_basename:
                    continue
            
            # Extract arguments from commands
            arguments = []
            i = 0
            while i < len(commands):
                cmd = commands[i]
                
                if isinstance(cmd, dict):
                    # Format: {--flag: value}
                    for flag, value in cmd.items():
                        arguments.append({
                            'flag': flag,
                            'value': str(value),
                            'type': self._infer_type_from_value(value),
                            'socket_side': 'output' if 'output' in flag else 'input'
                        })
                i += 1
            
            return {
                'name': step.get('name', ''),
                'environment': self._convert_to_uv_env(step.get('environment', '')),
                'script': script_cmd,
                'arguments': arguments
            }
        
        return None
    
    def _convert_to_uv_env(self, env_name: str) -> str:
        """
        Convert conda environment name to UV format.
        
        Args:
            env_name: Conda environment name (e.g., "segment_threshold")
            
        Returns:
            UV format environment string (e.g., "uv@3.11:segment_threshold")
        """
        if not env_name:
            return ''
        
        # If already in UV format, return as-is
        if env_name.startswith('uv@'):
            return env_name
        
        # Convert conda env name to UV format with Python 3.11
        return f'uv@3.11:{env_name}'
    
    def _infer_type_from_value(self, value) -> str:
        """Infer argument type from its value."""
        value_str = str(value).lower()
        
        if value_str in ['true', 'false']:
            return 'bool'
        elif '*' in value_str or '?' in value_str:
            return 'glob_pattern'
        elif any(x in value_str for x in ['folder', 'dir', '.tif', '.py']):
            return 'path'
        
        try:
            int(value)
            return 'int'
        except:
            pass
        
        try:
            float(value)
            return 'float'
        except:
            pass
        
        return 'string'
    
    def discover_output_files(self, script_path: str, arguments: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Run the script with test data and discover all created files.
        
        Args:
            script_path: Path to the Python script
            arguments: List of argument definitions with flags and defaults
        
        Returns:
            Tuple of (updated arguments list, list of discovered output files)
        """
        print("\n" + "="*60)
        print("Test Run - Output File Discovery")
        print("="*60)
        print("\nTo discover all files created by this CLI, we need to run it")
        print("with real test data.")
        
        # Ask for ONE test dataset path
        print("\nPlease provide a test input file or pattern:")
        print("(Press Enter to skip test run)")
        test_input_path = input("Test input: ").strip()
        
        if not test_input_path:
            logger.info("No test data provided, skipping test run")
            return arguments, []
        
        print("\n✓ Auto-generating test values for all arguments...")
        print("="*60)
        
        test_inputs = {}
        monitored_folders = set()
        
        # Add the input path folder to monitored folders
        input_folder = str(Path(test_input_path).resolve().parent)
        if input_folder and input_folder != '.':
            monitored_folders.add(input_folder)
        
        # Auto-generate test values for all arguments
        for arg in arguments:
            flag = arg['flag']
            arg_type = arg['type']
            default_val = arg.get('defaultValue', '')
            socket_side = arg.get('socketSide', 'input')
            
            # Skip output arguments - we're trying to discover them
            if socket_side == 'output':
                continue
            
            # Auto-generate test value
            test_value = self._generate_test_value(flag, arg_type, default_val, test_input_path)
            
            if test_value:
                test_inputs[flag] = test_value
                print(f"  {flag} = {test_value}")
                
                # Track folders mentioned in arguments for monitoring
                if arg_type in ['path', 'glob_pattern']:
                    try:
                        # Extract folder from path
                        if '*' in test_value:
                            folder = str(Path(test_value).parent.resolve())
                        else:
                            path_obj = Path(test_value)
                            if path_obj.suffix:  # It's a file
                                folder = str(path_obj.parent.resolve())
                            else:  # It's a folder
                                folder = str(path_obj.resolve())
                        
                        if folder and folder != '.':
                            monitored_folders.add(folder)
                            # Also monitor parent for outputs
                            if 'output' in flag.lower():
                                monitored_folders.add(str(Path(folder).parent.resolve()))
                    except Exception as e:
                        logger.debug(f"Could not extract folder from {test_value}: {e}")
        
        print(f"\n✓ Monitoring {len(monitored_folders)} folders for new files")
        print("="*60)
        print("\nRunning test execution...")
        
        # Convert paths to absolute and resolve
        monitored_folders = {str(Path(f).resolve()) for f in monitored_folders if f and f != '.'}
        
        # Snapshot files before execution in all monitored folders
        files_before = {}
        for folder in monitored_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                files_before[folder] = set(folder_path.rglob('*'))
            else:
                files_before[folder] = set()
        
        # Build and run command
        cmd = ['python', str(script_path)]
        for flag, value in test_inputs.items():
            cmd.extend([flag, value])
        
        print(f"\nCommand: {' '.join(cmd)}")
        print("\nExecuting (this may take a moment)...\n")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.repo_root
            )
            
            if result.returncode != 0:
                print(f"⚠ Script exited with code {result.returncode}")
                print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            else:
                print("✓ Script executed successfully")
            
            # Snapshot files after execution
            files_after = {}
            for folder in monitored_folders:
                folder_path = Path(folder)
                if folder_path.exists():
                    files_after[folder] = set(folder_path.rglob('*'))
                else:
                    files_after[folder] = set()
            
            # Find all new files across all monitored folders
            all_new_files = []
            for folder in monitored_folders:
                new_files = files_after[folder] - files_before[folder]
                for f in new_files:
                    if f.is_file():
                        try:
                            rel_path = f.relative_to(Path(folder).parent)
                            all_new_files.append({
                                'path': str(rel_path),
                                'folder': folder,
                                'name': f.name,
                                'size': f.stat().st_size
                            })
                        except ValueError:
                            all_new_files.append({
                                'path': str(f),
                                'folder': folder,
                                'name': f.name,
                                'size': f.stat().st_size
                            })
            
            if all_new_files:
                print(f"\n{'='*60}")
                print(f"Discovered {len(all_new_files)} new files:")
                print(f"{'='*60}")
                
                # Group by folder
                by_folder = {}
                for file_info in all_new_files:
                    folder = file_info['folder']
                    if folder not in by_folder:
                        by_folder[folder] = []
                    by_folder[folder].append(file_info)
                
                for folder, files in by_folder.items():
                    print(f"\nIn {folder}:")
                    for f in files:
                        size_kb = f['size'] / 1024
                        print(f"  • {f['name']} ({size_kb:.1f} KB)")
                
                # Ask user to verify and categorize output files
                print(f"\n{'='*60}")
                print("Output File Categorization")
                print(f"{'='*60}")
                print("\nWe'll now add these as output arguments.")
                print("For each file pattern, specify the argument flag and description.")
                
                # Group files by extension to suggest patterns
                by_extension = {}
                for file_info in all_new_files:
                    ext = Path(file_info['name']).suffix
                    if ext not in by_extension:
                        by_extension[ext] = []
                    by_extension[ext].append(file_info)
                
                new_output_args = []
                for ext, files in by_extension.items():
                    print(f"\n--- Files with extension '{ext}' ({len(files)} files) ---")
                    
                    # Show examples
                    for i, f in enumerate(files[:3]):
                        print(f"  Example: {f['name']}")
                    if len(files) > 3:
                        print(f"  ... and {len(files) - 3} more")
                    
                    # Suggest flag name based on file content
                    suggested_flag = self._suggest_output_flag(files[0]['name'])
                    
                    print(f"\nSuggested flag: {suggested_flag}")
                    flag = input(f"Output flag name (or Enter to use suggested, 's' to skip): ").strip()
                    
                    if flag.lower() == 's':
                        continue
                    
                    if not flag:
                        flag = suggested_flag
                    
                    if not flag.startswith('--'):
                        flag = '--' + flag
                    
                    description = input(f"Description for {flag}: ").strip()
                    if not description:
                        description = f"Output files with extension {ext}"
                    
                    # Determine output path pattern
                    # Use the folder from first file
                    output_folder = files[0]['folder']
                    
                    new_output_args.append({
                        'flag': flag,
                        'type': 'path',
                        'socketSide': 'output',
                        'isRequired': False,
                        'defaultValue': output_folder,
                        'description': description,
                        'validation': 'create_if_missing',
                        'userOverride': False
                    })
                
                # Add new output arguments to the arguments list
                arguments.extend(new_output_args)
                
                logger.info(f"Added {len(new_output_args)} output arguments")
                return arguments, [f['path'] for f in all_new_files]
            else:
                print("\n⚠ No new files detected. The script may not have run correctly,")
                print("or it may not create files with the provided test inputs.")
                return arguments, []
                
        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out (>120s)")
            print("\n✗ Test execution timed out")
            return arguments, []
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            print(f"\n✗ Test execution failed: {e}")
            return arguments, []
    
    def _generate_test_value(self, flag: str, arg_type: str, default_val: str, test_input_path: str) -> str:
        """
        Auto-generate reasonable test values for arguments.
        
        Args:
            flag: Argument flag (e.g., '--input-pattern')
            arg_type: Type of argument (path, int, float, string, etc.)
            default_val: Default value if any
            test_input_path: The test input path provided by user
        
        Returns:
            Generated test value as string
        """
        flag_lower = flag.lower()
        
        # Input-related flags - use the test input path
        if 'input' in flag_lower or 'search' in flag_lower:
            if 'pattern' in flag_lower:
                # For glob patterns, create pattern from test path
                test_path = Path(test_input_path)
                if '*' in test_input_path:
                    return test_input_path  # Already a pattern
                else:
                    # Create pattern: ./folder/*.ext
                    return str(test_path.parent / f"*.{test_path.suffix.lstrip('.')}")
            else:
                return test_input_path
        
        # Use default if provided for non-input args
        if default_val and 'input' not in flag_lower:
            return str(default_val)
        
        # Output-related flags
        if 'output' in flag_lower:
            if 'folder' in flag_lower or 'dir' in flag_lower:
                return './test_output'
            elif 'extension' in flag_lower or 'ext' in flag_lower:
                return 'tif'
            elif 'file' in flag_lower or 'name' in flag_lower:
                return 'output.tif'
            else:
                return './test_output'
        
        # Channel-related flags
        if 'channel' in flag_lower:
            if default_val:
                return str(default_val)
            return '0'
        
        # Threshold flags
        if 'threshold' in flag_lower or 'thresh' in flag_lower:
            return '0.5'
        
        # Size-related flags
        if 'size' in flag_lower or 'width' in flag_lower or 'height' in flag_lower:
            return '10'
        
        # Delimiter flags
        if 'delimiter' in flag_lower or 'delim' in flag_lower:
            if default_val:
                return str(default_val)
            return '__'
        
        # Method/algorithm flags
        if 'method' in flag_lower or 'algorithm' in flag_lower or 'mode' in flag_lower:
            if default_val:
                return str(default_val)
            return 'default'
        
        # Type-based defaults
        if arg_type == 'int':
            return '1'
        elif arg_type == 'float':
            return '1.0'
        elif arg_type == 'bool':
            return ''  # Don't include boolean flags by default
        elif arg_type in ['path', 'glob_pattern']:
            return './test_data'
        else:
            return ''  # Empty for unknown types
    
    def _suggest_output_flag(self, filename: str) -> str:
        """Suggest an output flag name based on filename."""
        name = Path(filename).stem
        
        # Common patterns
        if 'metadata' in name.lower():
            return '--output-metadata'
        elif 'mask' in name.lower():
            return '--output-masks'
        elif 'result' in name.lower():
            return '--output-results'
        elif 'data' in name.lower():
            return '--output-data'
        else:
            return '--output-folder'
    
    def interactive_refinement(self, definition: Dict) -> Dict:
        """
        Interactively refine the CLI definition with user input.
        
        Args:
            definition: Initial CLI definition
        
        Returns:
            Refined CLI definition
        """
        print("\n" + "="*60)
        print("CLI Definition - Interactive Refinement")
        print("="*60)
        
        # Review basic info
        print(f"\nName: {definition.get('name', 'Unnamed')}")
        if input("Change name? (y/n): ").lower() == 'y':
            definition['name'] = input("New name: ")
        
        print(f"\nScript: {definition.get('script', '')}")
        
        # Review category
        category = definition.get('category', 'Utilities')
        print(f"\nCategory: {category}")
        print("Available categories:", ', '.join(self.CATEGORY_PATTERNS.keys()))
        if input("Change category? (y/n): ").lower() == 'y':
            definition['category'] = input("New category: ")
            category = definition['category']
        
        # Set color and icon based on category
        definition['color'] = self.CATEGORY_COLORS.get(category, '#858585')
        definition['icon'] = self.CATEGORY_ICONS.get(category, '🔧')
        
        print(f"Icon: {definition['icon']}, Color: {definition['color']}")
        
        # Review arguments
        print(f"\n{'='*60}")
        print(f"Arguments ({len(definition.get('arguments', []))}):")
        print(f"{'='*60}")
        
        for i, arg in enumerate(definition.get('arguments', [])):
            print(f"\n[{i+1}] {arg['flag']}")
            print(f"    Type: {arg.get('type', 'string')}")
            print(f"    Side: {arg.get('socketSide', 'input')}")
            print(f"    Required: {arg.get('isRequired', False)}")
            print(f"    Default: {arg.get('defaultValue', '')}")
            print(f"    Description: {arg.get('description', '')[:60]}")
            
            if input("    Modify? (y/n): ").lower() == 'y':
                arg['socketSide'] = input(f"      Side (input/output) [{arg.get('socketSide', 'input')}]: ") or arg.get('socketSide', 'input')
                arg['isRequired'] = input(f"      Required? (y/n) [{arg.get('isRequired', False)}]: ").lower() == 'y'
                new_desc = input(f"      Description: ")
                if new_desc:
                    arg['description'] = new_desc
        
        print(f"\n{'='*60}")
        print("Refinement complete!")
        print(f"{'='*60}\n")
        
        return definition
    
    def infer_category(self, script_name: str) -> str:
        """Infer category from script name."""
        script_lower = script_name.lower()
        
        for category, patterns in self.CATEGORY_PATTERNS.items():
            if any(pattern in script_lower for pattern in patterns):
                return category
        
        return 'Utilities'
    
    def generate_definition(
        self,
        script_path: str,
        from_yaml: str = None,
        interactive: bool = True
    ) -> Dict:
        """
        Generate a complete CLI definition.
        
        Args:
            script_path: Path to the Python script
            from_yaml: Optional path to YAML config to load from
            interactive: Whether to use interactive refinement
        
        Returns:
            Complete CLI definition dictionary with outputFilesDefined=False
        """
        script_path = Path(script_path)
        script_name = script_path.stem
        
        # Start with basic structure
        definition = {
            'id': script_name,
            'name': script_name.replace('_', ' ').title(),
            'icon': '🔧',
            'color': '#858585',
            'description': f"CLI tool: {script_name}",
            'environment': 'uv@3.11:' + script_name,
            'executable': 'python',
            'script': f"standard_code/python/{script_path.name}",
            'helpCommand': f"python standard_code/python/{script_path.name} --help",
            'arguments': [],
            'version': '1.0.0',
            'author': 'BIPHUB',
            'lastParsed': datetime.now().isoformat(),
            'outputFilesDefined': False  # Outputs not yet discovered - will be auto-discovered on first run
        }
        
        # Infer category
        category = self.infer_category(script_name)
        definition['category'] = category
        definition['color'] = self.CATEGORY_COLORS.get(category, '#858585')
        definition['icon'] = self.CATEGORY_ICONS.get(category, '🔧')
        
        # SECONDARY SOURCE: Load YAML only for metadata and better default values
        environment = None  # Environment to use for running --help
        yaml_values = {}  # Store YAML values for later use as better defaults
        
        if from_yaml:
            yaml_data = self.load_from_yaml_config(from_yaml, script_path.name)
            if yaml_data:
                logger.info("Loaded metadata and default values from YAML config")
                # Extract metadata
                if 'name' in yaml_data:
                    definition['name'] = yaml_data['name']
                if 'environment' in yaml_data:
                    definition['environment'] = yaml_data['environment']
                    environment = yaml_data['environment']  # Use this for running --help
                
                # Store YAML argument values for later use (only as better defaults)
                if 'arguments' in yaml_data:
                    for arg in yaml_data['arguments']:
                        flag = arg['flag']
                        yaml_values[flag] = arg.get('value', '')
                    logger.info(f"Extracted {len(yaml_values)} default values from YAML for fallback use")
        
        # PRIMARY SOURCE: Parse help output - this is the source of truth
        # Use environment from YAML if available
        help_data = self.parse_help_output(str(script_path), environment=environment)
        
        # Build argument list FROM HELP OUTPUT ONLY
        if help_data.get('arguments'):
            logger.info(f"Parsed {len(help_data['arguments'])} arguments from help output (primary source)")
            for arg in help_data['arguments']:
                flag = arg['flag']
                
                # Use help default if available, otherwise try YAML value as fallback
                default_value = arg.get('default', '')
                if not default_value and flag in yaml_values:
                    default_value = yaml_values[flag]
                    logger.debug(f"Using YAML value as better default for {flag}: {default_value}")
                
                definition['arguments'].append({
                    'flag': flag,
                    'type': arg.get('type', 'string'),
                    'socketSide': 'output' if 'output' in flag else 'input',
                    'isRequired': arg.get('required', False),
                    'defaultValue': default_value,
                    'description': arg.get('description', ''),
                    'validation': self._infer_validation(arg),
                    'userOverride': False
                })
        else:
            logger.warning("No arguments found in help output - cannot generate definition")
            if yaml_values:
                logger.warning(f"YAML had {len(yaml_values)} arguments, but they were IGNORED (help output is source of truth)")
        
        # Note: We skip test run during generation to avoid command-line prompts
        # Output files will be auto-discovered when the node is first run in the GUI
        logger.info("Output files will be auto-discovered on first run in GUI")
        
        # Interactive refinement (optional)
        if interactive and not from_yaml:
            # Only offer interactive mode if not loading from YAML
            # (YAML configs are already verified)
            definition = self.interactive_refinement(definition)
        
        return definition
    
    def _infer_validation(self, arg: Dict) -> str:
        """Infer validation rules from argument info."""
        flag = arg['flag'].lower()
        arg_type = arg.get('type', 'string')
        
        if 'input' in flag and arg_type == 'path':
            return 'must_exist'
        elif 'output' in flag and arg_type == 'path':
            return 'create_if_missing'
        
        return ''
    
    def save_definition(self, definition: Dict, output_path: str):
        """Save CLI definition to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(definition, f, indent=4)
        
        logger.info(f"Saved CLI definition to: {output_path}")
    
    def scan_for_missing_definitions(self, python_dir: str, cli_defs_dir: str) -> List[str]:
        """
        Scan for Python scripts without CLI definitions.
        
        Args:
            python_dir: Directory containing Python scripts
            cli_defs_dir: Directory containing CLI definition JSONs
        
        Returns:
            List of script paths without definitions
        """
        python_dir = Path(python_dir)
        cli_defs_dir = Path(cli_defs_dir)
        
        # Get all Python scripts
        scripts = [f for f in python_dir.glob('*.py') if not f.name.startswith('_')]
        
        # Get all existing definition IDs
        existing_ids = set()
        for json_file in cli_defs_dir.rglob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    existing_ids.add(data.get('id', ''))
            except:
                pass
        
        # Find missing
        missing = []
        for script in scripts:
            if script.stem not in existing_ids:
                missing.append(str(script))
        
        return missing


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate CLI definition JSON files for Pipeline Designer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from old YAML config (recommended)
  python generate_cli_definition.py --from-yaml pipeline_configs/convert_to_tif.yaml \\
                                     --script standard_code/python/convert_to_tif.py \\
                                     --no-interactive
  
  # Generate from script alone (will need output discovery later)
  python generate_cli_definition.py --script standard_code/python/convert_to_tif.py \\
                                     --no-interactive
  
  # Scan for missing definitions and generate them
  python generate_cli_definition.py --scan-missing
  
  # Specify output directory
  python generate_cli_definition.py --script standard_code/python/mask_measure.py \\
                                     --no-interactive \\
                                     --output-dir pipeline-designer/cli_definitions/Measurement
        """
    )
    
    parser.add_argument('--script', help='Path to Python script to generate definition for')
    parser.add_argument('--from-yaml', help='Load initial definition from YAML config')
    parser.add_argument('--scan-missing', action='store_true', 
                       help='Scan for scripts without CLI definitions')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Skip interactive refinement')
    parser.add_argument('--output-dir', 
                       default='pipeline-designer/cli_definitions',
                       help='Output directory for JSON files')
    parser.add_argument('--repo-root', help='Repository root directory')
    
    args = parser.parse_args()
    
    generator = CLIDefinitionGenerator(repo_root=args.repo_root)
    
    if args.scan_missing:
        # Scan for missing definitions
        python_dir = Path(generator.repo_root) / 'standard_code' / 'python'
        cli_defs_dir = Path(generator.repo_root) / 'pipeline-designer' / 'cli_definitions'
        
        missing = generator.scan_for_missing_definitions(str(python_dir), str(cli_defs_dir))
        
        if missing:
            print(f"\nFound {len(missing)} scripts without CLI definitions:")
            for i, script in enumerate(missing, 1):
                print(f"  {i}. {Path(script).name}")
            
            print("\nGenerate definitions for these? (y/n): ", end='')
            if input().lower() == 'y':
                for script in missing:
                    print(f"\n{'='*60}")
                    print(f"Processing: {Path(script).name}")
                    print(f"{'='*60}")
                    
                    # Try to find matching YAML
                    yaml_path = None
                    yaml_dir = Path(generator.repo_root) / 'pipeline_configs'
                    script_name = Path(script).stem
                    for yaml_file in yaml_dir.glob('*.yaml'):
                        yaml_data = generator.load_from_yaml_config(str(yaml_file), Path(script).name)
                        if yaml_data:
                            yaml_path = str(yaml_file)
                            print(f"Found matching YAML: {yaml_file.name}")
                            break
                    
                    # Generate definition
                    definition = generator.generate_definition(
                        script,
                        from_yaml=yaml_path,
                        interactive=not args.no_interactive
                    )
                    
                    # Save to appropriate category folder
                    category = definition['category'].replace(' ', ' ')
                    output_path = Path(args.output_dir) / category / f"{definition['id']}.json"
                    generator.save_definition(definition, str(output_path))
        else:
            print("\nNo missing CLI definitions found!")
    
    elif args.script:
        # Generate single definition
        script_path = Path(args.script)
        if not script_path.exists():
            logger.error(f"Script not found: {args.script}")
            sys.exit(1)
        
        definition = generator.generate_definition(
            args.script,
            from_yaml=args.from_yaml,
            interactive=not args.no_interactive
        )
        
        # Save
        category = definition['category'].replace(' ', ' ')
        output_path = Path(args.output_dir) / category / f"{definition['id']}.json"
        generator.save_definition(definition, str(output_path))
        
        print(f"\nCLI definition generated successfully!")
        print(f"  Output: {output_path}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
