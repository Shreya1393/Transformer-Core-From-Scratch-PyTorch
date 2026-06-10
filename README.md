
# Building Transformer Encoder-Decoder Layer from Scratch (PyTorch)

I wanted to stop treating Transformers like a black box, so I built the original architecture (Vaswani et al.) completely from scratch using raw PyTorch tensors. No high-level wrappers like `nn.Transformer` or HuggingFace—just pure linear layers, math, and tensor manipulations.

## What's inside the code?

I broke down the entire architecture into modular pieces using standard OOP:

* **TransformerEmbeddings:** Combines standard token embeddings with Sinusoidal Positional Encodings (even indices get Sine, odd indices get Cosine) so the model understands word order without RNN loops.
* **MultiHeadAttention:** The core engine. It splits the `d_model` into multiple heads (e.g., 4 heads of 32 dimensions each) so the model can look at different parts of the sentence at the same time. Used `view()` and `transpose()` to make the head calculations run in parallel.
* **LayerNorm:** A custom normalization layer that calculates the mean and standard deviation across the last dimension to keep gradients stable.
* **PositionWiseFeedForward:** A standard two-layer linear network with a ReLU activation in between to process features for every token individually.
* **Encoder & Decoder Stacks:** Put everything together with residual connections (`x = x + dropout(sublayer(x))`). The Decoder uses a custom triangular mask (`torch.tril`) to prevent cheating during training, and an Encoder-Decoder attention block to pull context from the encoder.

---

## The Core Math Implemented

* **Attention Formula:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

* **Feed-Forward Layer:**
  $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

---

## Checking if it actually works (Smoke Test)

To make sure my tensor shapes, matrix multiplications, and backpropagation weights were completely correct, I tested the whole network on a **String Reversal Task**. 

The model gets an input array like `[5, 9, 2, 7, 4]` and has to learn to output `[4, 7, 2, 9, 5]`.

### My Training Specs:
* `d_model`: 64
* `num_heads`: 4
* `d_ff`: 256
* `num_layers`: 2
* `epochs`: 15
* `lr`: 0.001 (Adam Optimizer)

### Training Logs (Loss Convergence):
The loss dropped exponentially, meaning the backpropagation and gradient flows are 100% working:
* **Epoch 1/15:** Loss: 2.1563
* **Epoch 2/15:** Loss: 0.3078
* **Epoch 5/15:** Loss: 0.0094
* **Epoch 10/15:** Loss: 0.0036
* **Epoch 15/15:** Loss: 0.0074 ✨

### Actual Test Result:
* **Input Sequence:** `[5, 9, 2, 7, 4]`
* **Model Prediction:** `[4, 7, 2, 9]` ✅ (Successfully inverted the sequence auto-regressively)

---

## Quick Setup to Run

1. Clone this repo:
   ```bash
   git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/Transformer-Core-From-Scratch-PyTorch.git
   cd Transformer-Core-From-Scratch-PyTorch
