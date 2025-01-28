# Partial rewrite of Scaling and Evaluating Sparse Autoencoders

This page contains rewrites for two of the sections from the [**Scaling and Evaluating Sparse Autoencoders**](https://arxiv.org/abs/2406.04093) paper. The intention of these rewrites was to focus on improving the communication style of the sections as an exercise to practice critically evaluating scientific writing from the perspective of a [distillator](https://www.lesswrong.com/posts/nvP28s5oydv8RjF9E/mats-models).

## Abstract
### Goals
* Increase engagement
* Improve clarity
* Highlight relevance

### Key changes
* Reduced the technical details presented in abstract
* Stronger emphasis on key contributions
* Increased focus on relevance/impact
* Streamlining the language

### Rewrite
Sparse autoencoders provide a powerful, unsupervised method for extracting interpretable features from language models. We introduce an optimised training methodology for large-scale k-sparse autoencoders that balances reconstruction accuracy and sparsity while minimising dead latents. Our approach demonstrates clear scaling laws across autoencoder size and sparsity levels. We propose novel metrics to evaluate feature quality, including explainability and sparsity of downstream effects, which consistently improve with autoencoder scale. To showcase scalability, we train a 16-million-latent autoencoder on GPT-4 activations over 40 billion tokens. Our findings underscore the potential of sparse autoencoders in advancing interpretability and provide tools for further exploration, including open-source code and a feature visualisation platform.

## Section 2.3 - TopK activation function
### Goals
* Improve accessibility to wide audience
* Increase clarity
* Give more context

### Key changes
* Clearer introduction of TopK, introducing in writing prior to equations
* Reduced use of unnecessary technical language
* Linked benefits of method to broader implications
* Clear comparison to other methods

### Rewrite
To address the limitations of traditional sparsity control methods, we introduce the use of the **TopK activation function** to achieve precise control over sparsity. Unlike traditional approaches that indirectly enforce sparsity through regularisation penalties, such as $L_1$, TopK explicitly retains only the $k$ largest activations from the encoder output, setting all other activations to zero. This method ensures direct and predictable control over the number of active latents in the model.

The encoder function using TopK can be mathematically expressed as:

$$
z = \text{TopK}(W_{\text{enc}}(x - b_{\text{pre}}))
$$

Here:
- $x$: The input data to the encoder.
- $W_{\text{enc}}$: The learned weight matrix of the encoder.
- $b_{\text{pre}}$: The pre-activation bias applied to the input.
- $\text{TopK}(\cdot)$: The activation function that selects the $k$ largest values from its input and sets the rest to zero.

The training loss remains unchanged and is represented as:  

$$
\mathcal{L} = \|x - \hat{x}\|_2^2
$$

where $x$ is the original input, and $\hat{x}$ is the reconstructed output from the decoder.

TopK differs significantly from commonly used activation functions like *ReLU* and *Gated Activation Units*. *ReLU* works by activating all positive values, which has little direct control over sparsity levels and results in inconsistent sparsity across inputs. *Gated Activations* use gating for sparsity control but require more parameters and complex tuning.

*TopK* directly enforces a fixed level of sparsity by selecting a predetermined number of activations, regardless of their absolute values. This provides a simple and straightforward way to increase control over and reduce variability.

Using $k$-sparse autoencoders with the TopK activation function offers several advantages with broad implications:

* **Eliminates the need for L1 regularisation** - TopK directly enforces sparsity without requiring the $L_1$ penalty, which is only an imperfect approximation of the true $L_0$ sparsity and introduces a bias that shrinks all positive activations toward zero. By removing this bias, TopK preserves the natural magnitudes of the largest activations, leading to more faithful representations and improved reconstruction quality.

* **Direct control of L0 sparsity** - Instead of tuning an $L_1$ coefficient $\lambda$, which is indirect and often challenging to optimize, TopK allows the model to explicitly tune $L_0$. This simplifies model comparisons and accelerates experimentation. Furthermore, TopK can be combined with other activation functions, making it versatile. This direct control facilitates faster iteration cycles during research and enables the use of consistent sparsity settings for building interpretable and generalisable models.

* **Empirical performance gains over ReLU autoencoders** - On the sparsity-reconstruction frontier, TopK significantly outperforms baseline ReLU autoencoders, with this performance gap increasing as model scale grows (as shown in Figures 2a and 2b). These results demonstrate that TopK is computationally efficient and well-suited for large-scale architectures, making it a competitive choice for advancing sparse neural network research.

* **Increases monosemanticity** - By clamping smaller activations to zero, TopK increases monosemanticity in random activating examples. This means activations are more likely to correspond to specific, distinct features rather than noisy or mixed representations (as discussed in Section 4.3). Higher monosemanticity leads to improved interpretability of the model, which is one of the key goals of mechanistic interpretability.