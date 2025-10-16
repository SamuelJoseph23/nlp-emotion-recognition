import os
import sys

print("="*60)
print("SETUP DIAGNOSIS")
print("="*60)

# Check current directory
print(f"\nCurrent directory: {os.getcwd()}")

# Check if data folder exists
print(f"\nData folder exists: {os.path.exists('data')}")

# List contents of data folder
print("\nContents of data folder:")
if os.path.exists('data'):
    files = os.listdir('data')
    if files:
        for f in files:
            print(f"  - {f}")
    else:
        print("  (empty)")
else:
    print("  (folder doesn't exist)")

# Check if required files exist
required_files = ['data/train.txt', 'data/test.txt', 'data/val.txt']
print("\nRequired dataset files:")
for file in required_files:
    exists = os.path.exists(file)
    status = "✓ Found" if exists else "✗ Missing"
    print(f"  {status}: {file}")

# Check Python packages
print("\nChecking installed packages:")
try:
    import pandas
    print(f"  ✓ pandas {pandas.__version__}")
except:
    print("  ✗ pandas not installed")

try:
    import numpy
    print(f"  ✓ numpy {numpy.__version__}")
except:
    print("  ✗ numpy not installed")

try:
    import nltk
    print(f"  ✓ nltk {nltk.__version__}")
except:
    print("  ✗ nltk not installed")

try:
    import matplotlib
    print(f"  ✓ matplotlib {matplotlib.__version__}")
except:
    print("  ✗ matplotlib not installed")

print("\n" + "="*60)
