"""
Universal ImageJ Script Runner using PyImageJ
Runs ImageJ macros (.ijm) or ImageJ2 scripts (.py, .js, .groovy) with flexible arguments

MIT License - BIPHUB, University of Oslo

Usage:
    python run_imagej_script.py --script path/to/macro.ijm --arg1 value1 --arg2 value2
    python run_imagej_script.py --script path/to/script.py --input /data --output /results
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_flexible_args() -> tuple[str, Dict[str, Any], Optional[str], bool]:
    """
    Parse command-line arguments with flexible unknown arguments.
    
    Returns:
        Tuple of (script_path, arguments_dict, imagej_path, headless)
    """
    parser = argparse.ArgumentParser(
        description="Run ImageJ macros or scripts from Python using PyImageJ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ImageJ macro with parameters
  python run_imagej_script.py --script simple_threshold.ijm --input /data --suffix .tif --minsize 35000

  # Run ImageJ2 Python script
  python run_imagej_script.py --script analysis.py --channel 0 --threshold 100

  # With ImageJ path specified
  python run_imagej_script.py --script macro.ijm --imagej-path "C:/Fiji.app" --input /data

All unknown arguments are passed directly to the ImageJ script as parameters.
        """
    )
    
    parser.add_argument(
        '--script',
        type=str,
        required=True,
        help='Path to ImageJ macro (.ijm) or ImageJ2 script (.py, .js, .groovy)'
    )
    
    parser.add_argument(
        '--imagej-path',
        type=str,
        default=None,
        help='Path to ImageJ/Fiji installation (optional, will use default if not specified)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run ImageJ in headless mode (default: True)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Parse known args and capture all unknown args
    args, unknown = parser.parse_known_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert script path to absolute path
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        logger.error(f"Script file not found: {script_path}")
        sys.exit(1)
    
    # Parse unknown arguments into a dictionary
    script_args = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        
        if arg.startswith('--'):
            # Long option format
            key = arg[2:]  # Remove '--'
            
            # Check if next item is a value or another flag
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                i += 2
            else:
                # Flag without value, treat as True
                value = 'true'
                i += 1
            
            script_args[key] = value
        elif arg.startswith('-'):
            # Short option format (not recommended but supported)
            key = arg[1:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('-'):
                value = unknown[i + 1]
                i += 2
            else:
                value = 'true'
                i += 1
            script_args[key] = value
        else:
            # Positional argument (skip)
            logger.warning(f"Skipping positional argument: {arg}")
            i += 1
    
    return str(script_path), script_args, args.imagej_path, args.headless


def format_macro_arguments(args_dict: Dict[str, Any]) -> str:
    """
    Format arguments dictionary into ImageJ macro argument string.
    
    ImageJ macros expect arguments as: "key1=value1,key2=value2,..."
    
    Args:
        args_dict: Dictionary of argument key-value pairs
    
    Returns:
        Formatted argument string
    """
    arg_pairs = []
    for key, value in args_dict.items():
        # Convert paths to forward slashes for ImageJ compatibility
        if isinstance(value, str) and ('\\' in value or ':' in value):
            value = value.replace('\\', '/')
        arg_pairs.append(f"{key}={value}")
    
    return ",".join(arg_pairs)


def run_imagej_macro(script_path: str, args_dict: Dict[str, Any], 
                     imagej_path: Optional[str] = None, headless: bool = True) -> None:
    """
    Run an ImageJ macro or script using PyImageJ.
    
    Args:
        script_path: Path to the ImageJ macro or script file
        args_dict: Dictionary of arguments to pass to the script
        imagej_path: Optional path to ImageJ/Fiji installation
        headless: Whether to run in headless mode
    """
    try:
        import imagej
        logger.info("Initializing ImageJ...")
        
        # Initialize ImageJ - if imagej_path provided, use it; otherwise auto-download
        if imagej_path:
            logger.info(f"Using ImageJ from: {imagej_path}")
            ij = imagej.init(imagej_path, mode='headless' if headless else 'interactive')
        else:
            logger.info("Using Fiji (will auto-download if needed)")
            # Use Fiji for full plugin support
            ij = imagej.init('sc.fiji:fiji', mode='headless' if headless else 'interactive')
        
        logger.info(f"ImageJ version: {ij.getVersion()}")
        
        # Read the script file
        script_path_obj = Path(script_path)
        logger.info(f"Reading script: {script_path}")
        
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
        
        # Determine script type by extension
        extension = script_path_obj.suffix.lower()
        
        if extension == '.ijm':
            # ImageJ macro - use ij.py.run_macro
            logger.info("Running ImageJ macro (.ijm) via ij.py.run_macro()")
            
            if args_dict:
                logger.info(f"Macro parameters: {args_dict}")
            
            # Run the macro with args - PyImageJ handles parameter injection
            result = ij.py.run_macro(script_content, args_dict)
            
        elif extension in ['.py', '.js', '.groovy', '.clj', '.bsh']:
            # ImageJ2 script - use ij.py.run_script
            logger.info(f"Running ImageJ2 script ({extension}) via ij.py.run_script()")
            
            if args_dict:
                logger.info(f"Script parameters: {args_dict}")
            
            # Map extension to language name
            lang_map = {
                '.py': 'python',
                '.js': 'javascript', 
                '.groovy': 'groovy',
                '.clj': 'clojure',
                '.bsh': 'beanshell'
            }
            language = lang_map.get(extension, extension[1:])
            
            # Run the script with parameters
            result = ij.py.run_script(language, script_content, args_dict)
            
        else:
            logger.error(f"Unsupported script type: {extension}")
            logger.error("Supported types: .ijm, .py, .js, .groovy, .clj, .bsh")
            sys.exit(1)
        
        logger.info("Script execution completed successfully")
        
        # Check for outputs in result
        if result is not None:
            # Try to get all outputs
            try:
                outputs = result.getOutputs()
                if outputs and len(outputs) > 0:
                    logger.info(f"Script produced {len(outputs)} output(s):")
                    for key in outputs.keySet():
                        value = result.getOutput(key)
                        logger.info(f"  {key}: {value}")
            except:
                logger.info(f"Script result: {result}")
        
    except ImportError:
        logger.error("PyImageJ is not installed!")
        logger.error("Install it with: uv sync --group imagej")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running ImageJ script: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    logger.info("=== ImageJ Script Runner ===")
    
    # Parse arguments
    result = parse_flexible_args()
    script_path = result[0]
    script_args = result[1]
    imagej_path = result[2]
    headless = result[3]
    
    logger.info(f"Script: {script_path}")
    logger.info(f"Arguments: {script_args}")
    logger.info(f"Headless: {headless}")
    
    # Run the script
    run_imagej_macro(script_path, script_args, imagej_path, headless)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
