# Introduction

>OK, Deep Learning has outlived its usefulness as a buzz-phrase.
Deep Learning est mort. Vive Differentiable Programming! Yan LeCun

In my previous post, while discussing the importance of DSLs in ML and AI, we mentioned the idea of Software 2.0, introduced in [this article]((https://medium.com/@karpathy/software-2-0-a64152b37c35)
) by Andrej Karpathy:

> Software 2.0 is written in neural network weights. No human is involved in writing this code because there are a lot of weights (typical networks might have millions), and coding directly in weights is kind of hard (I tried). Instead, we specify some constraints on the behavior of a desirable program (e.g., a dataset of input output pairs of examples) and use the computational resources at our disposal to search the program space for a program that satisfies the constraints. In the case of neural networks, we restrict the search to a continuous subset of the program space where the search process can be made (somewhat surprisingly) efficient with backpropagation and stochastic gradient descent [[19]](https://medium.com/@karpathy/software-2-0-a64152b37c35)

*Deep Learning is eating software* [[13]](https://petewarden.com/2017/11/13/deep-learning-is-eating-software/). In this post, we'll dig deeper in this.

# Automatic Differentiation
What is Automatic Differentiation (AD)? There are countless of resources available online on the topic [[6]](https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation).
In a nutshell:

1. Forward Mode computes directional derivatives, also known as tangents. The directional derivative can be evaluated without explicitly computing the Jacobian [[4]](https://www-sop.inria.fr/tropics/ad/whatisad.html). In other words, one sweep of forward mode can calculate one column vector
of the Jacobian, `J ˙x`, where `˙x`  is a column vector of seeds [[7]](http://www.robots.ox.ac.uk/~tvg/publications/talks/autodiff.pdf)
2. Reverse Mode computes directional gradients, that is one sweep of reverse mode can calculate one row vector of the Jacobian, ¯yJ, where ¯y is a row vector of seeds [[7]](http://www.robots.ox.ac.uk/~tvg/publications/talks/autodiff.pdf)
3. The computational cost of one sweep forward or reverse is roughly
equivalent, but reverse mode requires access to intermediate
variables, requiring more memory.
4. Reverse mode AD is best suited for `F: R^n -> R`, while forward mode AD is best suited for `G: R -> R^m`. For other cases with `n > 1` and `m > 1`, the choice is non trivial.
5. **Backpropagationis merely a specialised version of automatic differentiation** [[2](https://idontgetoutmuch.wordpress.com/2013/10/13/backpropogation-is-just-steepest-descent-with-automatic-differentiation-2/)]: backpropagation is sometimes also known as reverse mode automatic differentiation [[3](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf)] for the historical background.

# Implementing Automatic Differentiation

The implementation of Automatic Differentiation is an interesting software engineering topic.
[[15]](http://www-sop.inria.fr/tropics/papers/TapenadeRef12.pdf) identifies two pricipal ways to implement Automatic Differentiation: 

> 1. Operator Overloading - if the language of P permits, one can replace the types of the
Foating-point variables with a new type that contains additional derivative information,
and overload the arithmetic operations for this new type so as to propagate this derivative
information along
> 2. Program Transformation - One can instead decide to explicitly build a new source
code that computes the derivatives. This is very similar to a compiler, except that it produces source code. This approach is more development-intensive than Operator Overloading, which is one
reason why Operator Overloading AD tools appeared earlier and are more numerous. [[15]](http://www-sop.inria.fr/tropics/papers/TapenadeRef12.pdf) 

[autograd](https://github.com/HIPS/autograd) is likely one of the most frequently used automatic differentiation libraries. [autograd](https://github.com/HIPS/autograd) is a great place to start:

> To compute the gradient, Autograd first has to record every transformation that was applied to the input as it was turned into the output of your function. To do this, Autograd wraps functions (using the function `primitive`) so that when they're called, they add themselves to a list of operations performed. Autograd's core has a table mapping these wrapped primitives to their corresponding gradient functions (or, more precisely, their vector-Jacobian product functions). To flag the variables we're taking the gradient with respect to, we wrap them using the `Box` class. You should never have to think about the Box class, but you might notice it when printing out debugging info.

> After the function is evaluated, Autograd has a graph specifying all operations that were performed on the inputs with respect to which we want to differentiate. This is the computational graph of the function evaluation. To compute the derivative, we simply apply the rules of differentiation to each node in the graph. [source](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)

This "boxing" is the OOD flavour of operator overloading. 

# Control Flow, In-Place Operations and Aliasing

## Control flow
It is crucial to note that Automatic Differentiation is applicable to code that contains control flow (branching, looping, ..). The possibility to have control flow is a key selling point of Deep Learning frameworks with Dynamic Computational graphs (ex: PyTorch, Chainer) - a capability oftern referred to as "Define and Run" [[18]](https://medium.com/intuitionmachine/pytorch-dynamic-computational-graphs-and-modular-deep-learning-7e7f89f18d1). However, control flow might result in code only piecewise differentiable, a significant complexity overhead [[4]](https://www-sop.inria.fr/tropics/ad/whatisad.html).

Another approach would be in the flavour of synthetic gradients [[22]](https://arxiv.org/abs/1608.05343?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI): 

> we can make backprop itself more efficient by introducing decoupled training modules with some synchronization mechanism between them, organized in a hierarchical fashion [[21]](https://blog.keras.io/the-future-of-deep-learning.html)


## In-Place operations
In-place operations, a necessary evil in algorithm design, pose an additionl hazard:

> In-place operations pose a hazard for automatic differentiation, because
an in-place operation can invalidate data that would be needed in the differentiation
phase. Additionally, they require nontrivial tape transformations to be performed. [[16]](https://openreview.net/pdf?id=BJJsrmfCZ)

[[16]](https://openreview.net/pdf?id=BJJsrmfCZ) provides an intuition of how PyTorch deals with in-place operations with invalidation.

> Every underlying storage of a variable is associated with a version
counter, which tracks how many in-place operations have been applied to the storage. When a
variable is saved, we record the version counter at that time. When an attempt to use the saved
variable is made, an error is raised if the saved value doesn’t match the current one. [[16]](https://openreview.net/pdf?id=BJJsrmfCZ)

```python
y = x.tanh()    # y._version == 0
y.add_(3)       # y._version == 1 
y.backward()    # ERROR: version mismatch in tanh_backward
```

## Aliasing
Aliasing also constitutes a technical challenge: 

```python
y = x[:2]
x.add_(3)    
y.backward()  
```
 
The in-place addition to x also causes some elements of y to be updated; thus, y’s
computational history has changed as well. Supporting this case is fairly nontrivial, so PyTorch
rejects this program, using an additional field in the version counter (see Invalidation paragraph) to determine that the data is shared [[16]](https://openreview.net/pdf?id=BJJsrmfCZ)

# Differentiable programming

https://www.facebook.com/yann.lecun/posts/10155003011462143

One way of viewing deep learning systems is “differentiable functional programming” [[8]](http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf). Deep Learning has a functional interpretation: 

1. Weight-tying or multiple applications of the same neuron
(e.g., ConvNets and RNNs) resemble function abstraction [[8]](http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf)
2. Structural patterns of composition resemble
higher-order functions (e.g., map, fold, unfold, zip) [[8]](http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf) [[12]](http://colah.github.io/posts/2015-09-NN-Types-FP/); 

> The most natural playground for exploring functional structures trained as deep learning networks would be a new language that can run back-propagation directly on functional programs. [[14]](https://www.edge.org/response-detail/26794)

One of the benefits of a higher level abstraction is the possibility to more easily design an infrastructure that tunes model parameters and model hyper-parameters [[10]](https://arxiv.org/pdf/1502.03492.pdf) having access to hypergradient:

>The availability of hypergradients allow you to do gradient-based optimization of gradient-based optimization, meaning that you can do things like optimizing learning rate and momentum schedules, weight initialization parameters, or step sizes and mass matrices in Hamiltonian Monte Carlo models. [[11]](http://hypelib.github.io/Hype/); 

> Gaining access to gradients with respect to hyperparamters
opens up a garden of delights. Instead of straining to eliminate hyperparameters from our models, we can embrace them, and richly hyperparameterize our models. Just as having a high-dimensional elementary parameterization gives a flexible model, having a high-dimensional  hyperparameterization gives flexibility over model classes, regularization, and training methods.[[10]](https://arxiv.org/pdf/1502.03492.pdf)

There are however deeper implications:

> It feels like a new kind of programming altogether, a kind of differentiable functional programming. One writes a very rough functional program, with these flexible, learnable pieces, and defines the correct behavior of the program with lots of data. Then you apply gradient descent, or some other optimization algorithm. The result is a program capable of doing remarkable things that we have no idea how to create directly, like generating captions describing images. [[9]](http://colah.github.io/posts/2015-09-NN-Types-FP/)

I like where this lines of thoughts goes: functinal programming means functional composability.

> Monolithic Deep Learning networks that are trained end-to-end as we typically find today are intrinsically immensely complex such that we are incapable of interpret its inference or behavior. There are recent research that have shown that an incremental training approach is viable. Networks have been demonstrated to work well by training with smaller units and then subsequently combining them to perform more complex behavior. [[20]](https://medium.com/intuitionmachine/why-teaching-will-be-the-sexiest-job-of-the-future-a-i-economy-b8e1c2ee413e)

# The road adhead of us

I am not sure if the term Differentiable Programming will stick around. The risk of confusion with [Differential *Dynamic* Programming](https://en.wikipedia.org/wiki/Differential_dynamic_programming) is high.

The idea, on the other hand, is intriguing. Very intriguing and I am very happy to see projects such as [Tensorlang](https://github.com/tensorlang/tensorlang) [[17]](https://medium.com/@maxbendick/designing-a-differentiable-language-for-deep-learning-1812ee480ff1)



# Resources

[1] Ullrich, Karen, Edward Meeds, and Max Welling. "Soft weight-sharing for neural network compression." arXiv preprint arXiv:1702.04008 (2017)

[2] Dominic, Steinitz, Backpropogation is Just Steepest Descent with Automatic Differentiation, [link](https://idontgetoutmuch.wordpress.com/2013/10/13/backpropogation-is-just-steepest-descent-with-automatic-differentiation-2/)

[3] Roger Grosse, Intro to Neural Networks and Machine Learning Lecture notes, [link](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L06%20Backpropagation.pdf)

[4] What is Automatic Differentiation ? [link](https://www-sop.inria.fr/tropics/ad/whatisad.html)

[5] The gradient and the directional derivative, [link](https://math.oregonstate.edu/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/grad/grad.html)

[6] Alexey Radul, Introduction to Automatic Differentiation, [link](https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation)

[7] Havard Berland, Automatic Differentiation, [link](http://www.robots.ox.ac.uk/~tvg/publications/talks/autodiff.pdf)

[8] Atılım Güneş Baydin, Differentiable Programming, [link](http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf)

[9] Christopher Olah, Neural Networks, Types, and Functional Programming, [link](http://colah.github.io/posts/2015-09-NN-Types-FP/)

[10] Maclaurin, Dougal, David Duvenaud, and Ryan Adams. "Gradient-based hyperparameter optimization through reversible learning." International Conference on Machine Learning. 2015.

[11] Hype: Compositional Machine Learning and Hyperparameter Optimization, [link](http://hypelib.github.io/Hype/)

[12] Christopher Olah, Neural Networks, Types, and Functional Programming, [link](http://colah.github.io/posts/2015-09-NN-Types-FP/)

[13] Pete Warden, Deep Learning is Eating Software, [link](https://petewarden.com/2017/11/13/deep-learning-is-eating-software/)

[14] David Dalrymple, Differentiable Programming, [link](https://www.edge.org/response-detail/26794)

[15] Hascoet, Laurent, and Valérie Pascual. ["The Tapenade Automatic Differentiation tool: principles, model, and specification."](http://www-sop.inria.fr/tropics/papers/TapenadeRef12.pdf) ACM Transactions on Mathematical Software (TOMS) 39.3 (2013): 20

[16] Paszke, Adam, et al. ["Automatic differentiation in PyTorch."](https://openreview.net/pdf?id=BJJsrmfCZ) (2017)

[17] Max Bendick, Designing a Differentiable Language for Deep Learning, [link](https://medium.com/@maxbendick/designing-a-differentiable-language-for-deep-learning-1812ee480ff1)

[18] Carlos Perez, PyTorch, Dynamic Computational Graphs and Modular Deep Learning, [link](https://medium.com/intuitionmachine/pytorch-dynamic-computational-graphs-and-modular-deep-learning-7e7f89f18d1)

[19] Andrej Karpathy, Software 2.0, [link](https://medium.com/@karpathy/software-2-0-a64152b37c35)

[20] Carlos Perez, Deep Teaching: The Sexiest Job of the Future, [link](https://medium.com/intuitionmachine/why-teaching-will-be-the-sexiest-job-of-the-future-a-i-economy-b8e1c2ee413e)

[21] François Chollet, The future of deep learning, [link](https://blog.keras.io/the-future-of-deep-learning.html)

[22] Jaderberg, Max, et al. ["Decoupled neural interfaces using synthetic gradients."](https://arxiv.org/abs/1608.05343?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=The%20Wild%20Week%20in%20AI) arXiv preprint arXiv:1608.05343 (2016).