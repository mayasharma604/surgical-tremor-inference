# surgical-tremor-inference

# ML-Based Surgical Intent Inference

## Overview
Robotic surgical systems receive noisy control signals that include involuntary hand tremor and sensor artifacts. While surgeons internally know their intended motion, robotic platforms must infer intent from noisy time-series data.

This project explores whether a lightweight machine learning model can infer a surgeon’s intended trajectory using temporal context, enabling tremor suppression without introducing excessive lag or overriding deliberate motion.

## Core Idea
Rather than filtering tremor using static signal processing, we frame tremor suppression as an intent inference problem:
- Input: Noisy hand/tool trajectories
- Output: Estimated intended trajectory

A temporal ML model learns motion patterns across time, distinguishing voluntary motion from involuntary oscillations.

## Model
- Architecture: LSTM-based sequence model
- Input: Sliding window of 2D positions
- Output: Next intended position
- Baseline comparison: Raw signal vs ML inference

## Data
Synthetic trajectories are generated and corrupted with realistic tremor models (high-frequency oscillations + noise). This allows controlled evaluation of intent recovery performance.

## Results
The ML model produces smoother, more stable trajectories while preserving intentional direction changes, outperforming static filtering in scenarios requiring fine motor precision.

## Limitations
- Synthetic data
- No force or tactile feedback
- Not validated for clinical use

## Future Work
- Real surgeon data
- Hybrid ML + control systems
- Integration with robotic tool simulators

## Disclaimer
This project is a research prototype and not intended for clinical deployment.


surgical-intent-inference/
│
├── data/
│   ├── generate_synthetic.py
│   └── dataset.py
│
├── models/
│   └── intent_lstm.py
│
├── train.py
├── evaluate.py
├── visualize.py
│
├── requirements.txt
└── README.md
