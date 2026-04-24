# ABES Cognitive Evaluation Report

**Date**: 2026-03-05T03:40:24.976959+00:00  
**Model**: unknown  
**Total prompts**: 1000  
**Pass rate**: 82.5% (95% CI: [0.800, 0.848])  
**Mean cosine similarity**: 0.875 +/- 0.181  
**Ecology violations**: 0  

## Domain Breakdown

| Domain | Passed | Rate | Mean cos | Violations |
|--------|--------|------|----------|------------|
| episodic_memory | 121/125 | 96.8% | 0.933 | 0 |
| language_comprehension | 88/125 | 70.4% | 0.826 | 0 |
| reasoning | 99/125 | 79.2% | 0.849 | 0 |
| selective_attention | 107/125 | 85.6% | 0.889 | 0 |
| self_correction | 103/125 | 82.4% | 0.905 | 0 |
| semantic_memory | 116/125 | 92.8% | 0.930 | 0 |
| social_cognition | 73/125 | 58.4% | 0.747 | 0 |
| working_memory | 118/125 | 94.4% | 0.924 | 0 |

## Failed Prompts

| ID | Domain | Construct | Similarity | Details |
|----|--------|-----------|------------|---------|
| 88 | episodic_memory | temporal_ordering | 0.699 | similarity 0.699 < threshold 0.7 |
| 93 | episodic_memory | temporal_ordering | 0.682 | similarity 0.682 < threshold 0.7 |
| 95 | episodic_memory | temporal_ordering | 0.655 | similarity 0.655 < threshold 0.7 |
| 98 | episodic_memory | temporal_ordering | 0.685 | similarity 0.685 < threshold 0.7 |
| 180 | semantic_memory | property_retrieval | 0.594 | similarity 0.594 < threshold 0.7 |
| 182 | semantic_memory | property_retrieval | 0.650 | similarity 0.650 < threshold 0.7 |
| 203 | semantic_memory | source_discrimination | 0.828 | forbidden similarity 0.651 > threshold 0.6 |
| 209 | semantic_memory | source_discrimination | 0.629 | similarity 0.629 < threshold 0.7 |
| 228 | semantic_memory | knowledge_update | 0.814 | forbidden similarity 0.800 > threshold 0.6 |
| 231 | semantic_memory | knowledge_update | 0.584 | similarity 0.584 < threshold 0.7 |
| 234 | semantic_memory | knowledge_update | 0.602 | similarity 0.602 < threshold 0.7 |
| 243 | semantic_memory | knowledge_update | 0.876 | forbidden similarity 0.790 > threshold 0.6 |
| 246 | semantic_memory | knowledge_update | 0.850 | forbidden similarity 0.775 > threshold 0.6 |
| 253 | working_memory | multi_item_encoding | 0.558 | similarity 0.558 < threshold 0.7 |
| 280 | working_memory | item_retrieval | 0.660 | similarity 0.660 < threshold 0.7 |
| 300 | working_memory | item_retrieval | 0.595 | similarity 0.595 < threshold 0.7 |
| 360 | working_memory | updating | 0.658 | similarity 0.658 < threshold 0.7 |
| 363 | working_memory | updating | 0.503 | similarity 0.503 < threshold 0.7 |
| 364 | working_memory | updating | 0.425 | similarity 0.425 < threshold 0.7 |
| 365 | working_memory | updating | 0.699 | similarity 0.699 < threshold 0.7 |
| 378 | selective_attention | target_encoding | 0.597 | similarity 0.597 < threshold 0.7 |
| 404 | selective_attention | distractor_filtering | 0.485 | similarity 0.485 < threshold 0.7 |
| 413 | selective_attention | distractor_filtering | 0.693 | similarity 0.693 < threshold 0.7 |
| 414 | selective_attention | distractor_filtering | 0.573 | similarity 0.573 < threshold 0.7 |
| 415 | selective_attention | distractor_filtering | 0.540 | similarity 0.540 < threshold 0.7 |
| 429 | selective_attention | inhibition | 0.406 | similarity 0.406 < threshold 0.7 |
| 430 | selective_attention | inhibition | 0.578 | similarity 0.578 < threshold 0.7 |
| 459 | selective_attention | relevance_gating | 0.642 | similarity 0.642 < threshold 0.7 |
| 463 | selective_attention | relevance_gating | 0.539 | similarity 0.539 < threshold 0.7 |
| 478 | selective_attention | focused_retrieval | 0.685 | similarity 0.685 < threshold 0.7 |
| 479 | selective_attention | focused_retrieval | 0.680 | similarity 0.680 < threshold 0.7 |
| 480 | selective_attention | focused_retrieval | 0.662 | similarity 0.662 < threshold 0.7 |
| 485 | selective_attention | focused_retrieval | 0.660 | similarity 0.660 < threshold 0.7 |
| 493 | selective_attention | focused_retrieval | 0.644 | similarity 0.644 < threshold 0.7 |
| 494 | selective_attention | focused_retrieval | 0.404 | similarity 0.404 < threshold 0.7 |
| 495 | selective_attention | focused_retrieval | 0.677 | similarity 0.677 < threshold 0.7 |
| 499 | selective_attention | focused_retrieval | 0.460 | similarity 0.460 < threshold 0.7 |
| 500 | selective_attention | focused_retrieval | 0.367 | similarity 0.367 < threshold 0.7 |
| 503 | language_comprehension | scalar_implicature | 0.753 | forbidden similarity 0.705 > threshold 0.6 |
| 505 | language_comprehension | scalar_implicature | 0.603 | similarity 0.603 < threshold 0.7; forbidden similarity 0.600 > threshold 0.6 |
| 507 | language_comprehension | scalar_implicature | 0.559 | similarity 0.559 < threshold 0.7 |
| 509 | language_comprehension | scalar_implicature | 0.582 | similarity 0.582 < threshold 0.7 |
| 511 | language_comprehension | scalar_implicature | 0.266 | similarity 0.266 < threshold 0.7 |
| 513 | language_comprehension | scalar_implicature | 0.482 | similarity 0.482 < threshold 0.7 |
| 515 | language_comprehension | scalar_implicature | 0.242 | similarity 0.242 < threshold 0.7 |
| 517 | language_comprehension | scalar_implicature | 0.703 | forbidden similarity 0.757 > threshold 0.6 |
| 519 | language_comprehension | scalar_implicature | 0.525 | similarity 0.525 < threshold 0.7; forbidden similarity 0.646 > threshold 0.6 |
| 521 | language_comprehension | scalar_implicature | 0.688 | similarity 0.688 < threshold 0.7; forbidden similarity 0.615 > threshold 0.6 |
| 523 | language_comprehension | scalar_implicature | 0.432 | similarity 0.432 < threshold 0.7 |
| 525 | language_comprehension | scalar_implicature | 0.562 | similarity 0.562 < threshold 0.7 |

*...and 125 more failures omitted.*