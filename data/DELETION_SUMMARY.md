# Files/Directories Deletion Summary

## Space Savings Analysis

### Directories to Delete:
1. **data/repos/** - 11.92 GB
   - Contains 500 cloned GitHub repositories
   - Terraform files already extracted to data/terraform_files/
   - Safe to delete: YES

2. **data/scans/** - 14.52 MB
   - Contains raw Checkov and tfsec JSON outputs
   - Results already aggregated in file_labels.jsonl
   - Safe to delete: YES (can be regenerated if needed)

### Files to Delete:
3. **data/terraform_files_manifest.jsonl.backup** - ~21 MB
   - Backup file, not needed
   - Safe to delete: YES

### Total Space Savings: ~12 GB

## Verification Complete
- ✓ file_labels.jsonl exists (1,500 entries)
- ✓ datasets/train.jsonl exists
- ✓ datasets/val.jsonl exists
- ✓ datasets/test.jsonl exists

## Files Kept (Required):
- data/terraform_files/ - Extracted Terraform files
- data/file_labels.jsonl - Labeled dataset
- data/datasets/ - Train/val/test splits

## Deletion Date
Generated: $(Get-Date)

