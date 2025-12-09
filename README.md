<div align="center">

![Chess Game Animation](https://media.giphy.com/media/iCZyBnPBLr0dy/giphy.gif)

# ♟️ Beyond The Board

**Deep Learning Chess Coaching System**

*Neural network-powered position evaluation and move suggestion*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)

</div>

---

## What is Beyond The Board?

Beyond The Board is a chess AI that evaluates positions and suggests moves using deep learning instead of traditional search algorithms. Think of it as a neural network alternative to Stockfish—trained on millions of positions to understand chess strategy, tactics, and checkmate patterns.

The system provides real-time position analysis through a REST API, making it easy to integrate into chess applications, training tools, or analysis platforms.

---

## 🧠 Three Specialized Models

| Model | Architecture | Specialty | Best For |
|-------|-------------|-----------|----------|
| **The Coach** | ResNet (3 blocks) | Balanced evaluation | General position analysis |
| **The Tactician** | CNN + 200k mate games | Checkmate patterns | Tactical puzzles, forced mates |
| **The Mastermind** | Transformer | Pure positional understanding | Complex strategic positions |

Each model accepts different input formats—the API automatically detects and routes appropriately.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-org/beyond-the-board.git
cd beyond-the-board
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn beyond_the_board.api.chess:app --host 0.0.0.0 --port 8000
```

### Python SDK Usage

```python
import requests

# Analyze a position (FEN notation)
fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Use The Coach for balanced analysis
response = requests.get(f"http://localhost:8000/predict?fen={fen}")

# Use The Tactician for tactical positions
response = requests.get(f"http://localhost:8000/predict_moves?fen={fen}")

# Use The Mastermind for deep positional analysis
response = requests.get(f"http://localhost:8000/predict_more?fen={fen}")
```

### REST API Response

```json
{
  "current_eval": 0.40,
  "to_move": "white",
  "moves": [
    {
      "move_san": "Nf3",
      "eval_after": 0.50,
      "eval_change": 0.09,
      "improves": "True"
    }
  ]
}
```

---

## 📊 Training Pipelines

### V1 Pipeline (Baseline)
Three-step progressive training: foundation → self-play → final coach model.

### V2 Pipeline (Advanced) ✅
Enhanced architecture with MCTS self-play, Squeeze-and-Excitation attention, and alpha-beta pruning for mate search up to 6 moves.

| Metric | V1 | V2 | Improvement |
|--------|----|----|-------------|
| Evaluation MAE | 1.76 | <1.0 | 52% better |
| Mate-in-1 | 3.7% | 95%+ | 25x better |
| Mate-in-2 | — | 75% | — |
| Mate-in-3 | — | 60% | — |
| Inference | ~100ms | 50-100ms | CPU optimized |

**Key innovations:** Multi-task learning, focal loss for hard examples, temperature-based sampling, uncertainty quantification.

---

## 📁 Project Structure

```
beyond_the_board/
├── models/
│   ├── pipe1/          # V1 pipeline (step1.py, step2.py, step3.py)
│   └── pipe2/          # V2 pipeline with MCTS & attention
├── api/
│   └── chess.py        # FastAPI endpoints with auto-detection
├── notebooks/          # Training & analysis notebooks
└── weights/
    ├── optimus_prime.keras
    ├── shallow_blue.keras
    └── big_brother.keras
```

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow/Keras
- **API:** FastAPI + Uvicorn
- **Chess Logic:** python-chess
- **Deployment:** Docker
- **Training Data:** Lichess database
- **Ground Truth:** Stockfish evaluations

---

## API Endpoints

| Endpoint | Model | Input Features |
|----------|-------|----------------|
| `GET /Optimus_Prime` | The Coach | Board + 40 metadata |
| `GET /Shallow_Blue` | The Tactician | Board + 15 metadata |
| `GET /Big_Brother` | The Mastermind | Board only |

All endpoints accept a `fen` query parameter and return the same response format.

---

<div align="center">

**Built with ♟️ by the Beyond The Board Team**

[Documentation](docs/) · [Report Bug](issues/) · [Request Feature](issues/)

</div>
