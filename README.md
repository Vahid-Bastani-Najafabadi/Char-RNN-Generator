# Character-Level RNN Word Generator (PyTorch)

A tiny character-level generative language model built with **PyTorch**.  
Trained on four toy words, it learns next-character probabilities and generates **novel words** via proportional sampling.

> **Training set**
> ```
> winnipeg
> manitoba
> winnitoba
> manipeg
> ```

---

## ✨ Features

- Builds a **character dictionary** (+ `EOW` end token) and one-hot encodings  
- Implements a **Vanilla RNN** (`torch.nn.RNN`) with an MLP head  
- Trains using **CrossEntropyLoss** with character-level accuracy tracking  
- Experiments with **Adam, RMSprop, SGD** across different learning rates  
- Explores **weight decay** with Adam optimizer  
- Generates **50 new words** by proportional sampling  

---

## ⚙️ Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- NumPy
- Matplotlib
- torchinfo

Install dependencies:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```bash
torch
numpy
matplotlib
torchinfo
```

---
## 🚀 Quick Start

Run the script:
```bash
python char_rnn.py
```
This will:
1. Build the dictionary and one-hot encodings
2. Train the RNN with different optimizers and learning rates
3. Plot **loss** and **accuracy** curves
4. Generate and print **50 new words**
---
## 🧠 Model

- **Core**: `nn.RNN(input_size=K, hidden_size=128, batch_first=True, nonlinearity='tanh')`
- **Head**: `Linear(128→128) → ReLU → Linear(128→K)`
- **Loss**: Cross-entropy at each time step
- **Training**: Teacher forcing with true next character as input
- **Generation**: Starts with a random character, samples successive characters until `EOW`
---
## 📈 Experiments
The script trains the RNN with:
- **Adam**, **RMSprop**, **SGD** optimizers
- Learning rates from **0.05 → 0.0001**
- **Weight decay sweep** with Adam (`1e-9 → 1e-3`)

For each, it plots:
- Training **Loss vs Epoch**
- Training **Accuracy vs Epoch**

---
## 🔤 Word Generation

The trained model generates 50 new words:
- Picks a random start character
- Samples next characters from `softmax(logits)`
- Stops on `EOW`
- Repeats to produce 50 words

Example output (varies by run):
```bash
Generated Words:
['manipeg', 'winnitoba', 'manipeba', 'winnioba', ...]
```
---
## 📝 Reproducibility

To make runs deterministic, set seeds at the top of the script:
```bash
import torch, numpy as np, random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
```
Save outputs:
```bash
np.savetxt("outputs/generated_words.txt", new_words, fmt="%s")
plt.savefig("plots/adam_lr_sweep.png", dpi=200, bbox_inches="tight")
```
---
## ⚡ Known Notes

- Print statement inside training loop needs correct quotes:
```bash
print(f"Epoch {epoch+1}, lr: {optimizer.param_groups[0]['lr']}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
```
- Prefer `torch.softmax(logits, dim=-1)` instead of manual exp/sum(exp) for stability.
- Add a **max length cutoff** (e.g., 20) during generation to avoid rare infinite loops.
- Small typos in plot titles (“Optimzier” → “Optimizer”, “RMSProbp” → “RMSprop”).
---

## 🚀 Possible Extensions

- Try `nn.LSTM` or `nn.GRUA`
- Add **dropout** to **regularize**
- Introduce a **temperature** parameter for sampling creativity
- Split code into `model.py`, `train.py`, `generate.py` for cleaner structure
- Add CLI arguments (optimizer, lr, epochs, etc.) using `argparse`
---

## 📄 License

This project is licensed under the MIT [License](https://github.com/Vahid-Bastani-Najafabadi/Char-RNN-Generator/blob/main/LICENSE) – see the LICENSE file for details.
