> **🔧 Fork Notice**: This repository is a fork of [EasyLM](https://github.com/young-geng/EasyLM)

# Open Cabrita 🤖🇧🇷🇵🇹

**Open Cabrita** is a comprehensive research archive documenting our systematic investigation into Portuguese language model development. This repository preserves our complete experimental journey, including successful models, failed attempts, and critical insights gained from training across multiple architectures (LLaMA, Gemma) and scales (2B-7B parameters).

> **📚 Archive Note**: This repository serves as a research archive and historical record of our Portuguese LLM experiments conducted between 2023-2024. All methodologies, scripts, and findings are preserved for reproducibility and future research.

📄 **Research Paper**: [Open Cabrita: Challenges and Opportunities for Portuguese Language Models](https://arxiv.org/abs/2308.11878)

## 📋 Table of Contents

- [Research Summary](#research-summary)
- [Successful Model](#successful-model)
- [Experimental Timeline](#experimental-timeline)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Training Methodology](#training-methodology)
- [Usage Patterns](#usage-patterns)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Research Summary

This archive documents our systematic investigation into Portuguese language model development, spanning 18 months of experimentation (2023-2024). Our research methodology combined theoretical foundations with empirical validation, resulting in both successful models and valuable insights from controlled failures.

**Original Research Objectives:**
- 🎯 **Primary Goal**: Develop high-quality Portuguese language models through systematic scaling experiments
- 🔬 **Scientific Approach**: Document the complete experimental process, including failures and their causes
- � **Scaling Investigation**: Explore optimal architectures and scales for Portuguese LLM development
- 🌍 **Community Impact**: Provide open-source methodologies to accelerate Portuguese NLP research

**Actual Research Outcomes:**
- **✅ Successful Model**: Open Cabrita 3B - A high-performing Portuguese language model
- **� Unexpected Discovery**: Data quality emerged through failure analysis as the critical limiting factor
- **� Architecture Insights**: Different model architectures exhibit varying tolerance to data quality issues
- **📚 Methodological Contribution**: Complete reproducible pipeline and systematic failure documentation

## Successful Model

**Open Cabrita 3B** represents our most successful experimental outcome - a Portuguese language model that achieved excellent performance metrics and stable training characteristics.

### Model Specifications

| Specification | Value | Notes |
|--------------|-------|--------|
| **Architecture** | Modified LLaMA-3B | Portuguese-optimized adaptations |
| **Parameters** | 3.0B | Optimal scale for data quality alignment |
| **Context Length** | 2048 tokens | Standard context window |
| **Training Steps** | 400K steps | Converged with stable loss |
| **Training Duration** | ~3 weeks | Single TPU v3-8 pod |
| **Final Status** | ✅ **Production Ready** | Successfully completed training |

### Performance Characteristics
- **🚀 Training Stability**: Achieved smooth loss convergence without instabilities
- **🇧🇷🇵🇹 Language Quality**: Generates coherent Portuguese text with proper grammar and syntax
- **📊 Benchmark Performance**: Outperforms base 3B model on Portuguese evaluation tasks

## Experimental Timeline

Our research followed a systematic experimental methodology, documenting both successful outcomes and controlled failures for scientific completeness.

### Experimental Phases

#### Phase 1: Foundation Establishment (Q3-Q4 2023)
**Objective**: Establish baseline capabilities with smaller-scale models

| Model | Architecture | Parameters | Outcome | Key Insights |
|-------|-------------|------------|---------|--------------|
| **Open Cabrita 3B** | LLaMA-3B | 3.0B | ✅ **Success** | Established optimal configuration |

**Findings**: Successfully demonstrated feasibility of Portuguese LLM training when base model pretraining quality aligned with available Portuguese corpus quality (OpenLLaMA-3B's pretraining quality matched our Portuguese data quality).

#### Phase 2: Scaling Investigation (Q1-Q2 2024)
**Objective**: Investigate scaling properties and architectural variations

| Model | Architecture | Parameters | Outcome | Primary Challenge |
|-------|-------------|------------|---------|------------------|
| **LLaMA2 7B Variant** | LLaMA2-7B | 7.0B | ❌ **Failed** | Convergence instability |
| **Gemma 2B** | Gemma-2B | 2.0B | ❌ **Failed** | Training instabilities |
| **Gemma 7B** | Gemma-7B | 7.0B | ❌ **Failed** | Data quality sensitivity |

**Critical Discovery Through Failure Analysis**: Larger and more sophisticated models exhibited increased sensitivity to what we hypothesize may be quality mismatches between their original pretraining data and our Portuguese corpus, suggesting that data quality alignment—rather than preprocessing quality in isolation—could be a fundamental factor for Portuguese LLM scaling success.

## Dataset & Preprocessing

Our systematic scaling experiments unexpectedly revealed data quality as the primary limiting factor, leading us to investigate preprocessing requirements that were not initially the focus of our research.

### Primary Dataset
- **Source**: MC4 Portuguese corpus (Common Crawl)
- **Size**: 145GB of Portuguese web text
- **Coverage**: Comprehensive representation of Portuguese language variants (PT-BR, PT-PT)
- **Initial Assessment**: Appeared suitable for scaling experiments

### Preprocessing Evolution (Driven by Failure Analysis)

#### Initial Approach (Standard Processing)
- **Method**: Conventional web text cleaning (standard practice for scaling experiments)
- **Expectation**: Sufficient for Portuguese LLM scaling
- **Reality**: Led to training failures in more advanced models (LLaMA2 and Gemma)
- **Key Learning**: Standard preprocessing inadequate for Portuguese scaling

#### Failure-Driven Investigation
Our systematic failures across multiple architectures and scales led us to believe that data quality was the potential root cause.

### Critical Research Finding (Emergent)
**The Data Quality Mismatch Hypothesis**: Through systematic failure analysis, our best explanation for the observed patterns is that models originally pretrained on higher-quality datasets (LLaMA2, Gemma) may suffer significant knowledge degradation when continued pretraining uses lower-quality data, even if that data is domain-specific. We hypothesize that the quality gap between original pretraining data and Portuguese continued pretraining data caused these models to lose more general knowledge than they gained Portuguese-specific capabilities. While we didn't investigate this deeply, this hypothesis best explains why more sophisticated models failed while simpler models (OpenLLaMA-3B) succeeded.

## Training Methodology

Our training approach combined established best practices with Portuguese-specific optimizations, systematically validated across multiple experimental runs.

### Successful Configuration (Open Cabrita 3B)

#### Architecture Specifications
- **Base Model**: LLaMA-3B with Portuguese adaptations
- **Tokenizer**: Custom SentencePiece vocabulary (52K tokens) optimized for Portuguese morphology
- **Position Encoding**: RoPE (Rotary Position Embedding) for enhanced context modeling
- **Normalization**: Layer normalization for training stability

#### Training Hyperparameters
- **Precision**: bf16 mixed precision (memory efficiency without stability loss)
- **Sequence Length**: 2048 tokens (optimal context window)
- **Global Batch Size**: 256 (empirically determined optimal for 3B scale)
- **Learning Rate Schedule**: Cosine decay with warmup (critical for convergence stability)
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)

#### Infrastructure Configuration
- **Hardware**: Google Cloud TPU v3-8 pods
- **Training Framework**: JAX/Flax with pjit for distributed training
- **Monitoring**: Comprehensive metrics via Weights & Biases
- **Checkpointing**: Every 5,000 steps with full state recovery capability

#### Optimization Strategies
- **Gradient Checkpointing**: Effective memory reduction without performance impact
- **Data Pipeline Optimization**: Prefetching and parallel loading for TPU efficiency
- **Learning Rate Tuning**: Fine-grained schedule optimization critical for Portuguese data
- **Portuguese-Specific Tokenization**: Custom vocabulary significantly improved efficiency


### Failed Training Analysis

#### LLaMA2 (7B) and Gemma (2B & 7B) Model Experiments 
- **Original Hypothesis**: Modern architectures would handle Portuguese scaling better
- **Primary Issue**: Training instabilities and poor convergence across models
- **Our Best Hypothesis**: Their high-quality pretraining data made them particularly sensitive to the quality gap with our Portuguese corpus
- **Caveat**: This remains our best guess based on systematic comparison of outcomes across different base models

## Usage Patterns

### For Reproduction
1. **Environment Setup**: Use `scripts/tpu_vm_setup.sh`
2. **Data Preparation**: Follow `src/download_and_process_dataset.py`
3. **Training Execution**: Use configurations in `training_scripts/`
4. **Monitoring**: Integrate with provided W&B configurations

## Acknowledgments

This research was enabled by the foundational work of the open-source machine learning community. We acknowledge and thank the following projects and contributors:

### Core Framework Dependencies
* **EasyLM Training Framework** - [young-geng](https://github.com/young-geng/EasyLM) - Provided the robust training infrastructure that enabled our systematic experimentation
* **JAX LLaMA Implementation** - [JAX_llama](https://github.com/Sea-Snell/JAX_llama) - High-quality LLaMA implementation in JAX/Flax
* **Transformers Library** - [Hugging Face](https://huggingface.co/docs/transformers/) - JAX/Flax model implementations and utilities
* **MLX Utilities** - [mlxu](https://github.com/young-geng/mlxu) - Essential JAX utilities for distributed training
* **JAXSeq Framework** - [JAXSeq](https://github.com/Sea-Snell/JAXSeq) - Methodological inspiration for systematic LLM research

### Research Infrastructure
* **Google Cloud Platform** - TPU v3-8 access through research credits program
* **Weights & Biases** - Comprehensive experiment tracking and analysis platform
* **Common Crawl Foundation** - MC4 Portuguese dataset access

## License

This project is released under the same license terms as the original EasyLM framework to maintain compatibility and enable community contributions. 