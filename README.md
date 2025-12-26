# Semantic Refinement and Hierarchy-Aware Inference for Weakly Supervised Multi-Label Text Classification
This study addresses the challenge of Hierarchical Multi-Label Text Classification (H-MLTC) in a transductive
setting with no labeled training data. We propose a robust self-training framework that integrates Large
Language Models (LLMs) for semantic class enrichment and synthetic data generation. Our approach proceeds
in distinct phases: (1) semantic initialization using MPNet and LLM-generated descriptions, (2) high-precision
silver label generation via a gap-based heuristic, and (3) a refinement phase that addresses extreme class
imbalance through "starved class" augmentation and focal loss. Furthermore, we demonstrate that a top-down
beam search inference strategy significantly outperforms bottom-up approaches by enforcing taxonomic
consistency. Our method achieves a Sample F1 score of 0.63578 on the Amazon Product Review dataset[1],
showing substantial improvement over the baseline (0.39180). We also analyze the limitations of Graph Neural
Networks (GNNs) in this specific semantic space, providing empirical evidence of reduced distinctiveness.
Our code and data are available at: https://github.com/All4Nothing/20252R0136DATA30400.
Additional Key Words and Phrases: Hierarchical Classification, Weakly Supervised Learning, Data Augmenta-
tion, LLM, Self-Training
