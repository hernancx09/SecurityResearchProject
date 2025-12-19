# Axon InfraLinter – Research Notes

## Motivation

Traditional rule-based security scanners (Checkov, tfsec) excel at detecting known misconfiguration patterns but have limitations:

- **Limited context understanding**: Rules are pattern-matched and may miss subtle issues that require understanding code intent
- **Novel patterns**: New misconfiguration types require manual rule updates
- **False positives**: Overly strict rules can flag benign configurations

Large language models (LLMs) offer potential advantages:

- **Contextual reasoning**: Can understand relationships between resources and configuration intent
- **Generalization**: May detect misconfigurations not explicitly covered by rules
- **Natural language explanations**: Can provide human-readable justifications

## Research Questions

1. **Can a fine-tuned code LLM match or outperform rule-based scanners** on a curated Terraform security dataset?
2. **For which categories of misconfigurations** does the LLM provide the most value (e.g., IAM policies, encryption settings, network configurations)?
3. **What is the runtime and resource cost** of LLM-based scanning compared to existing tools?

## Implementation Lessons Learned

### Scanner Exit Code Handling

**Issue**: tfsec uses non-standard exit codes:
- Exit code `0`: Scan succeeded, no issues found
- Exit code `1`: Scan succeeded, issues found (this is **success**, not failure!)
- Exit code `2+`: Actual errors (tool not found, invalid input, etc.)

**Learning**: Initially misinterpreting exit code 1 as failure caused us to silently drop valid security findings. This taught me the importance of reading tool documentation carefully and testing edge cases.

**Solution**: Explicitly check `if result.returncode >= 2` to treat only codes ≥2 as errors.

### Dataset Balancing

**Issue**: Real-world security datasets are naturally imbalanced - insecure examples are relatively rare compared to secure configurations.

**Learning**: Without balancing, simple baselines (e.g., "always predict secure") achieve deceptively high accuracy. This masks the true performance differences between models.

**Solution**: Enforce equal numbers of secure and insecure examples in train/val/test splits. This makes evaluation metrics more meaningful and highlights where the LLM actually helps.

### Manifest Files for Traceability

**Issue**: When debugging mislabeled examples or analyzing results, we needed to trace findings back to their source repositories.

**Learning**: Storing comprehensive metadata (`github_repos_manifest.jsonl`, `terraform_files_manifest.jsonl`) makes the pipeline auditable and debuggable. This is especially important for research reproducibility.

**Solution**: Each stage produces a manifest file that links outputs back to inputs, preserving the full data lineage.

### LoRA vs. Full Fine-Tuning

**Decision**: Use LoRA (Low-Rank Adaptation) instead of full fine-tuning.

**Rationale**:
- **Memory efficiency**: LoRA trains only a small number of parameters, making it feasible on consumer GPUs
- **Faster training**: Fewer parameters to update means faster iterations
- **Modularity**: Can easily swap adapters or combine multiple adapters

**Tradeoff**: LoRA may have slightly lower performance than full fine-tuning, but the efficiency gains are worth it for this research prototype.

## Key Findings (To Be Filled)

*This section should be updated with your actual experimental results.*

### Performance Comparison

- **LLM vs. Checkov**: [Your findings]
- **LLM vs. tfsec**: [Your findings]
- **Where LLM helps most**: [Specific misconfiguration types]

### Failure Modes

- **Hallucinations**: Cases where LLM incorrectly flags non-existent issues
- **Missed detections**: Known misconfigurations that LLM failed to catch
- **False positives**: Secure configurations incorrectly flagged as insecure

### Runtime Analysis

- **Inference time per file**: [LLM vs. scanners]
- **Resource requirements**: [GPU memory, CPU usage]
- **Scalability**: [How does cost scale with corpus size?]

## Future Work

- **Larger models**: Evaluate with CodeLlama 13B or 34B variants
- **Ensemble approaches**: Combine LLM predictions with rule-based scanners
- **Few-shot learning**: Test if LLM can learn from fewer examples
- **Explanation quality**: Evaluate whether LLM explanations are more helpful than rule IDs

