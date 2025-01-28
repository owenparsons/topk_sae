# Strengths and improvements in Scaling and Evaluating Sparse Autoencoders

As outlined in the [repository's README](../README.md), I read through the [**Scaling and Evaluating Sparse Autoencoders**](https://arxiv.org/abs/2406.04093) paper with the intention of focusing on it's communication style from the perspective of a [distillator](https://www.lesswrong.com/posts/nvP28s5oydv8RjF9E/mats-models). My thoughts on the strengths/improvements regarding the article's communication style are captured below.

## What is effective about this paper's communication, and why?

After reading the paper, I found the following aspects particularly effective in communicating the motivations, approach and results of the paper:
* **The paper had a very clear structure and outlined the main contributions well** - The paper effectively organises content into distinct sections: methodology, scaling laws, and evaluation. It clearly enumerates its contributions upfront, aiding quick comprehension. It also gives a summary in the introduction section that details the main contributions of the other sections.
* **There are detailed but clear explanations of the methodologies** - Fairly complex concepts like k-sparse autoencoders and scaling laws are well-explained, and supported by both equations and diagrams. I'm a big fan of using both formal and visual approaches to conveying conceptual information as it improves accessibility for different types of reader as well as allowing for readers to access the paper at different levels of depth (quick skim vs deep read).
* **Good balance of conciseness and depth** - The writing throughout the paper generally strikes a good balance between conciseness and providing sufficient depth. The writing avoids being overly verbose while explaining technical concepts, which makes the article more engaging and focused.
* **Well structured progression of concepts** - The paper builds on the concepts it introduces well, starting with foundation concepts such as limitations with using the $L_1$ penalty and the benefits of k-sparse activation before building up to more complex ideas like scaling laws and alignment of evaluation metrics. This layered approach helps to tell a story and guide the reader.
* **Highlights key insights clearly** - The authors spotlight the most impactful results and insights in a very clear, structured way (e.g. bullet points for benefits of k-sparse autoencoders in sections 2.3).

## What could be improved about the way this paper communicates?

While I generally felt the writing communication was a strength of the paper, and thoroughly enjoyed reading it, I identified the following as potential areas for improvement:
* **Slightly disjointed repetition across sections** - Some of the discussions, such as those on scaling laws and sparsity trade-offs, crop up multiple times in slightly different forms. At times the repetition was slightly awkward and gave a slightly choppy feeling to the paper that distracted from the material being presented. Streamlining those repeated points could tighten up the narrative and cut down on redundancy.

* **Unclear prioritization of results** - Several different findings are presented in the paper and it’s sometimes unclear which results are most important/critical. A brief summary at the end of key sections, which highlights the most impactful results from that section, would help to guide readers.

* **Underexplored practical implications** - The implications of the findings, specifically around how larger sparse autoencoders might be used in real-world tasks, are only lightly touched upon in the paper. While the potential impact of the results presented here may be obvious to some, explicitly stating the impact of the findings would make the results more relatable.

* **Non-intuitive visualizations** - While the general use of visualisations are a strength of the paper, a number of them are overly complex and present information in a way that isn’t immediately obvious. This requires more deliberate thought and interpretation to understand what is being conveyed, which could interrupt the flow of the reader.