"""Simple test to see what's failing"""
import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print()

print("Checking Java...")
print("JAVA_HOME:", os.environ.get('JAVA_HOME', 'Not set'))
print("PATH includes java:", 'java' in os.environ.get('PATH', '').lower())
print()

print("Step 1: Import jpype module (not starting JVM yet)...")
sys.stdout.flush()

import jpype
print("Success! jpype module loaded")
print("jpype version:", jpype.__version__)
print()

print("Step 2: Find JVM path...")
sys.stdout.flush()
jvm_path = jpype.getDefaultJVMPath()
print(f"JVM path: {jvm_path}")
print(f"JVM exists: {os.path.exists(jvm_path)}")
print()

print("Step 3: Start JVM...")
sys.stdout.flush()
jpype.startJVM(jvm_path, "-Xmx512M", convertStrings=False)
print("JVM started!")
print()

print("All tests passed!")
