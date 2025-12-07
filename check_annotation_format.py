"""
Quick script to check annotation file format and diagnose issues.
"""

import sys

def check_annotation_format(filepath):
    """Check the format of an annotation file."""
    print(f"Checking file: {filepath}")
    print("="*60)
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        print("\nFirst 10 lines:")
        print("-"*60)
        
        for i, line in enumerate(lines[:10], 1):
            # Show the line with visible tabs and special characters
            visible_line = line.replace('\t', '→').replace('\n', '↵').replace(' ', '·')
            print(f"{i:3d}: {visible_line}")
        
        print("\nColumn analysis (first line):")
        print("-"*60)
        
        if lines:
            first_line = lines[0].strip()
            
            # Try different separators
            separators = {
                '\t': 'tab',
                ' ': 'space',
                ',': 'comma',
                '|': 'pipe'
            }
            
            for sep, name in separators.items():
                parts = first_line.split(sep)
                if len(parts) > 1:
                    print(f"{name:10s}: {len(parts)} columns")
                    if len(parts) <= 5:
                        for j, part in enumerate(parts):
                            print(f"  Column {j}: '{part}'")
        
        # Check if coordinates look numeric
        print("\nNumeric check (lines 1-5):")
        print("-"*60)
        for i, line in enumerate(lines[:5], 1):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    print(f"Line {i}: {parts[0][:30]:30s} → ({x}, {y})")
                except ValueError as e:
                    print(f"Line {i}: ERROR - {e}")
                    print(f"  Parts: {parts}")
        
        return True
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_annotation_format.py <annotation_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    check_annotation_format(filepath)
