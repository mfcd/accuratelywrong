Despite wide adoption in the industry, our understanding of deep learning is still lagging.

[20], nicely summarized by [21], identifies four research branches:

 - **Non-Convex Optimization**: we deal with a non-convex function, yet SGD works. Why does SGD even converge?
 - **Overparametrization and Generalization**:  how can Deep Neural Networks avoid the curse of dimensionality?

> Theorists have long assumed networks with hundreds of thousands of neurons and orders of magnitude more individually weighted connections between them should suffer from a fundamental problem: over-parameterization [19]

 -  **Role of Depth**:  How does depth help a neural network to converge? What is the link between depth and generalization? [21]
 -  **Generative Models**: Why do Generative Adversarial Networks (GANs) work so well? What theoretical properties could we use to stabilize them or avoid mode collapse? [21]

In this series of articles, we will focus on two areas [4]:

1. the analysis of the topology of the loss function
2. the width or flatness of the minima

 
#  Loss Function Topology
## The starting point: Spin Glasses
  
The spin glass model seems a great starting point for a very fertile research direction. Should we go down this path?  [4] warns prudence: 
> this theoretical picture, grounded on the topology of the loss function, has several shortcomings. First, the ultimate metric in machine learning is the generalisation accuracy rather than minimising the loss function, which only captures how well the model fits the training data  [...]
> Second, it is known empirically that deeper minima have lower generalisation accuracy than shallower minima

Before continuing our work on loss function topology, let's have a look at what is meant with flatness of minima.

# Flatness of Minima
Consider the cartoon energy landscape (of the empirical loss function) in figure X discussed in [5]. [5] notes that under a Bayesian prior on the parameters, say a Gaussian of a fixed variance at locations $x_{robust}$ and $x_{non−robust}$ respectively, the wider local minimum has a higher marginalized likelihood than the sharp valley on the right. In other words, [5] says that parameters that lie in wider local minima like  $x_{robust}$, which may possibly have a higher loss than the global minimum, should generalize better than the ones that are simply at the global minimum [9]. 

What is the definition of width? [6] provides an overview of definitions of width (flatness). Definitions include:

 - a flat minimum as "a large connected region in weight space where the error remains approximately constant" [7]
 - definitions that look at the local curvature of the loss function around the critical point. In particular, the information encoded in the eigenvalues of the Hessian (spectral norms, traces) [5]
 - local Entropy [5]
 
 ## Local entropy
 I find the idea of local entropy extremely compelling and extremely pedagogical in explaining the idea of "flatness" of minima. 




Local entropy, from a practical perspective, tempers weights, something that is attempted with explicit regularization: weight regularization (decay), dropout and batch norm regularization.

## Flatness = generalization?
In the previous section, looking at local entropy, we tried to get a feeling for the attempts to define what a good solution could look like. There is still no consensus on the best definition of flatness. [6] pointed out that 

> However, when following several definitions of flatness, we have shown
> that the conclusion that flat minima should generalize better than
> sharp ones cannot be applied as is without further context. Previously
> used definitions fail to account for the complex geometry of some
> commonly used deep architectures. In particular, the
> non-identifiability of the model induced by symmetries, allows one to
> alter the flatness of a minimum without affecting the function it
> represents. Additionally the whole geometry of the error surface with
> respect to the parameters can be changed arbitrarily under different
> parametrizations. In the spirit of [18]  our work indicates that more
> care is needed to define flatness to avoid degeneracies of the
> geometry of the model under study. Also such a concept can not be
> divorced from the particular parametrization of the model or input
> space.

In other words, a precise definition of flatness, which can be use to build necessary and sufficient conditions for generalization, is an active topic of research [17].

Furthermore, while optimizing explicitly for flatness might help (as we saw in the case of Local Entropy), it appears from empirical evidence that DNN and SGD can get the work done, even without too much help from regularization (xxx see screenshot from [16])

This encourages the study of the topology of deep learning empirical loss functions. We did not waste our time understanding the work of Choromanska et al. [1]. The $H$-spherical spin glass model is extremely useful starting point and creating a link between the work of deep learning researchers, statistical physics and random matrix theory. 

[1] relies in relatively crude assumptions as the authors themselves point out in [11]. However, research continued to progress: [12] shows that with mild over-parameterization and dropout-like noise, training error for a neural network with one hidden layer and piece-wise linear activation is zero at every local minimum. [13] shows, under less restrictive assumptions, that every local minimum is a global minimum and that every critical point that is not a global minimum is a saddle point. Results in [12] and [13] have received empirical evidence: see for example [14] where the authors argue that *as a first-order characterization, we believe that the landscape of empirical risk is simply a collection of (hyper) basins that each has a flat global minima*

In the next article, we will look at [12], [13] and at the work by Tomaso Poggio's work at MIT ([14], 
[15]); we will try to draw a connection between research on loss function topology, flatness of minima and the role of over-parametrization.

xxx picture from [15].






# References

[1] Choromanska, Anna, et al. "The loss surfaces of multilayer networks." _Artificial Intelligence and Statistics_. 2015, [link](http://proceedings.mlr.press/v38/choromanska15.pdf)
[2] Torben Krüger, Handout, Graduate seminar: Random Matrices, Spin Glasses and Deep Learning [link](https://sites.google.com/site/torbenkruegermath/home/graduate-seminar-random-matrices-spin-glasses-deep-learning)
[3]  Auffinger, Antonio, Gérard Ben Arous, and Jiří Černý. "Random matrices and complexity of spin glasses." *Communications on Pure and Applied Mathematics* 66.2 (2013): 165-201 [link](https://arxiv.org/pdf/1003.1129.pdf) 
[4] Zhang, Yao, et al. "Energy–entropy competition and the effectiveness of stochastic gradient descent in machine learning." _Molecular Physics_ (2018): 1-10 [link](https://arxiv.org/pdf/1803.01927.pdf)
[5] Chaudhari, Pratik, et al. "Entropy-sgd: Biasing gradient descent into wide valleys." _arXiv preprint arXiv:1611.01838_(2016). [link](https://arxiv.org/pdf/1611.01838.pdf)
[6] Dinh, Laurent, et al. "Sharp minima can generalize for deep nets." _arXiv preprint arXiv:1703.04933_ (2017) [link](https://arxiv.org/pdf/1703.04933.pdf)
[7] Hochreiter, Sepp, and Jürgen Schmidhuber. "Flat minima." _Neural Computation_ 9.1 (1997): 1-42. [link](http://www.bioinf.jku.at/publications/older/3304.pdf)
[8] Martin, Charles "Why deep learning works: perspectives from theoretical chemistry", MMDS 2016 [link](http://mmds-data.org/presentations/2016/s-martin.pdf)
[9] ] G. E. Hinton and D. Van Camp, in Proceedings of the sixth annual conference on Computational learning theory (ACM, 1993) pp. 5–13.
[10] Merhav, Neri. "Statistical physics and information theory." _Foundations and Trends® in Communications and Information Theory_ 6.1–2 (2010): 1-212. [link](http://webee.technion.ac.il/people/merhav/papers/p138f.pdf)
[11] Choromanska, Anna, Yann LeCun, and Gérard Ben Arous. "Open problem: The landscape of the loss surfaces of multilayer networks." _Conference on Learning Theory_. 2015 [link](http://proceedings.mlr.press/v40/Choromanska15.pdf)
[12] Soudry, Daniel, and Yair Carmon. "No bad local minima: Data independent training error guarantees for multilayer neural networks." _arXiv preprint arXiv:1605.08361_ (2016) [link](https://arxiv.org/pdf/1605.08361.pdf)
[13] Kawaguchi, Kenji. "Deep learning without poor local minima." _Advances in Neural Information Processing Systems_. 2016. [link](https://pdfs.semanticscholar.org/f843/49b38c61467f3ba8501a3fcff61c87a505e3.pdf)
[14] Liao, Qianli, and Tomaso Poggio. "Theory II: Landscape of the Empirical Risk in Deep Learning." _arXiv preprint arXiv:1703.09833_ (2017) [link](https://arxiv.org/pdf/1703.09833.pdf)
[15] Zhang, Chiyuan, et al. _Theory of deep learning iii: Generalization properties of sgd_. Center for Brains, Minds and Machines (CBMM), 2017 [link](https://dspace.mit.edu/bitstream/handle/1721.1/107841/CBMM-Memo-067-v3.pdf?sequence=6)
[16] Ben Recht talk at ICLR 2017, [link](https://iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2017:recht_iclr2017.pdf)
[17] Sathiya Keerthi, Interplay between Optimization and Generalization in Deep Neural Networks, 3rd annual Machine Learning in the Real World Workshop, [link](http://www.keerthis.com/Optimization_and_Generalization_Keerthi_Criteo_November_08_2017.pptx)
[18] Swirszcz, Grzegorz, Wojciech Marian Czarnecki, and Razvan Pascanu. "Local minima in training of deep networks." (2016) [link](https://pdfs.semanticscholar.org/6496/93b2437378699945d36e8203d79a93454273.pdf)
[19] Chris Edwards, Deep Learning Hunts for Signals Among the Noise, Communications of the ACM, June 2018, Vol. 61 No. 6, Pages 13-14 [link](https://cacm.acm.org/magazines/2018/6/228030-deep-learning-hunts-for-signals-among-the-noise/fulltext)
[20] Sanjeev Arora, Toward Theoretical
Understanding of Deep Learning, ICML 2018 [link](https://www.dropbox.com/s/qonozmne0x4x2r3/deepsurveyICML18final.pptx?dl=0)
[21] Arthur Pesah, Recent Advances for a Better Understanding of Deep Learning − Part I [link](https://towardsdatascience.com/recent-advances-for-a-better-understanding-of-deep-learning-part-i-5ce34d1cc914)
