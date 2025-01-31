# General notes on Scaling and Evaluating Sparse Autoencoders

This page contains my general notes on the [**Scaling and Evaluating Sparse Autoencoders**](https://arxiv.org/abs/2406.04093) paper as part of [a research exercise](../README.md).

## L0 and k
$L_0$ and $k$ are related concepts but they are not exactly the same.

### L0 Norm
$L_0$ refers to the number of non-zero elements in a vector or a set. In the context of sparsity in machine learning, the $L_0$ norm is the count of non-zero values in a given vector. This indicates how many features/activations are being used in a model (and is used to quantify the sparsity of the model).

### k (k-sparsity)
$k$ typically refers to the number of top activations that are allowed to remain non-zero in the context of a sparse activation function.
Using k-sparse activation indicates that only the top $k$ activations are kept while all other activations are set to zero.

## k-sparse activation
A k-sparse activation refers to a sparsity mechanism in neural networks where only the top $k$ largest activations (in terms of magnitude) are retained in a layer's output, and all other activations are set to zero. This method enforces a specific level of sparsity in the activations, ensuring that only the most significant features or signals are propagated to the next layer.

### Summary of process
1. A layer (e.g., a dense or convolutional layer) computes its activations as usual based on the input and learned weights.
2. The activations are ranked by their absolute values (magnitude).
3. Only the $k$ largest activations are kept; all others are set to zero.
4. The resulting sparse vector, with only $k$ nonzero elements, is passed to the next layer.

Mathematically, this can be expressed as:

$\text{Output}(x)_i = \begin{cases} x_i, & \text{if } x_i \text{ is among the top } k \text{ values in } |x| \\ 0, & \text{otherwise.} \end{cases}$

Here, $x_i$ represents the activations, and $|x|$ denotes their absolute values.

## Encode and decoder

The encoder and decoder are represented as:

$$
\mathbf{z} = \text{TopK}(\mathbf{W}_{\text{enc}}(\mathbf{x}_l - \mathbf{b}_{\text{dec}}) + \mathbf{b}_{\text{enc}})
$$

where $\mathbf{x}_l$ is the input activation, $\mathbf{b}_{dec}$ is the decoder bias, $\mathbf{W}_{\text{enc}}$ is the encoder matrix to transform into the latent representation, $\mathbf{b}_{enc}$ is the encoder bias and $\text{TopK}$ is the Top-k activation function.

$$
\hat{\mathbf{x}}_l = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{dec}} = \sum w_i f_i
$$

where $\mathbf{W}_{\text{dec}}$ is the decoder matrix to map from sparse/latent representation to the original activation space, $\mathbf{b}_{dec}$ is the decoder bias, $f_i$ are the basis vectors and $w_i$ are the weights from $\mathbf{z}$.

The decoder bias acts as a baseline/default activation pattern. This is subtracted from the input activations $before$ encoding as it acts as a centering operation. If we directly applied the encoder matrix to the input activations​, the encoding might be influenced by the presence of a strong bias term, which could reduce the effectiveness of sparsity enforcement. Removing the decoder bias​ before encoding ensures that the encoder is primarily learning the deviation of the input activations​ from the baseline rather than encoding absolute activations.


## References from 'Related work' section

### Lee Sharkey, 2022:
[Taking features out of superposition with sparse autoencoders](https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)
Investigates the challenge of superposition in neural networks, where multiple features are entangled within single neurons. Uses sparse autoencoders to disentangle features into more interpretable/monosemantic activations.

### Bricken et al., 2023:
[Towards monosemanticity: Decomposing language models with dictionary learning](https://transformer-circuits.pub/2023/monosemantic-
features/index.html) 
Introduces a method using dictionary learning to decompose language models into more interpretable components. Aims to identify underlying features that contribute to the model's predictions, to improve transparency/understanding of model behaviour.

### Yun et al., 2021:
[Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors.](https://arxiv.org/abs/2103.15949)

