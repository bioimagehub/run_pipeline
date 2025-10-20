import json
import glob

files = sorted(glob.glob('E:/Oyvind/BIP-hub-test-data/drift_correction/comprehensive_tests/test_report_*.json'))
if files:
    with open(files[-1]) as f:
        data = json.load(f)
    
    print("RMS ERRORS (lower is better, <0.1 is excellent):")
    print("="*60)
    for r in data['results'][:6]:
        print(f"{r['test_name']:<30} {r['algorithm']:<30}")
        print(f"  RMS Error:  {r['accuracy']['rms_error']:.4f} px")
        print(f"  Max Error:  {r['accuracy']['max_error']:.4f} px")
        print()
