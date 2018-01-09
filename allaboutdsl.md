# It's all about DSLs!

One of the most frequently asked question about my work is: what programming languages do you use?

_The short answer_: a mix of Scala, Python and F#

_The long answer_: applied machine learning is all about domain specific languages (DSL). DSLs are a hot topic in many of the tasks Machine Learning (ML) and Artificial Intelligence (AI) systems need to tackle.


## 1 Domain Language

DSLs are necessary to express concisely business logic.

### _Example: Trading_

```
let example = 
    trade {
        buy 100 "IBM" Shares At Max 45
        sell 40 "Sun" Shares At Min 24
        buy 25 "CISCO" Shares At Max 56 
    }
```
[Snippet source](https://github.com/dungpa/dsls-in-action-fsharp/blob/master/DSLCheatsheet.md), read more (Gosh2010, Frankau2009)

At the same time, ML and AI systems do not come set in stone.

- The underlying models reflect business and working hypothesis that might change over time
- Sensitivity analysis should not be only performed against model (hyper)parameters but also against business and working assumptions

DSLs come in handy to express complex business and working assumptions in a fluent, reads like English, way.

#TODO: find nice example


## 2 Mathematics

Once the problem has been formulated, it is then time to write a mathematical model. DSLs languages for statistics and mathematics have existed for decades: Matlab and R are extremely popular in the scientific community. DSL for statistics and mathematics embedded in general purpose programming language are enjoying increasing attention.


### _Example: Probabilistic programming (from Python)_

```
import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
```

[Snippet source](http://docs.pymc.io/notebooks/getting_started#A-Motivating-Example:-Linear-Regression), read more (Patil2010)


## 3 Querying (your database) and data manipulation
DSL are handy to write queries and manipulate data in a compact and expressive language.

A few examples that pop in my mind:

### _Example: F# Query expressions_
Query expressions look like our good, old, familiar SQL.

```
query {
    for student in db.Student do
    where (student.StudentID = 1)
    select student
    exactlyOne
}
```
[Snippet source](https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/query-expressions)

### _Example: Data Frames_ 
Data Frames, such as Pandas, provide syntactic sugar to perform data wrangling tasks such as [split, apply combine](https://pandas.pydata.org/pandas-docs/stable/groupby.html) and [pivots](http://nikgrozev.com/2015/07/01/reshaping-in-pandas-pivot-pivot-table-stack-and-unstack-explained-with-pictures/) (McKinney2010)

```
df.groupby('A').B.agg(['min', 'max'])
```

## 4 Under the hood: expressing computations

The need to have a domain specific language to reason about computations has been made evident by the work on Apache Spark. Spark runs, under the hood, a complex execution procedure that comprises several steps: the definition of a dataflow (logical plan), the definition of a DAG describing tasks and their execution (physical plan), job scheduling, .. [Read more](https://github.com/JerryLead/SparkInternals)

> TensorFlow and similar tools present themselves as “just libraries”, but they are extremely unusual ones. Most libraries provide a simple set of functions and data structures, not an entirely new programming system and runtime.  [Innes2017](https://julialang.org/blog/2017/12/ml&pl)

Why do we need such a language?

> The core reason for building new languages is simple: ML research has extremely high computational demands, and simplifying the modelling language makes it easier to add domain-specific optimisations and features  [Innes2017](https://julialang.org/blog/2017/12/ml&pl)

That's not all. Model complexity is growing exponentially. The work on DSLs that allow us represent, reason and analyze computations is currently extremely hot.

> models are becoming increasingly like programs, including ones that reason about other programs (e.g. program generators and interpreters), and with non-differentiable components like Monte Carlo Tree Search. It’s enormously challenging to build runtimes that provide complete flexibility while achieving top performance, but increasingly the most powerful models and groundbreaking results need both. [Innes2017](https://julialang.org/blog/2017/12/ml&pl)

In a Deep Learning context:

> An increasingly large number of people are defining the network procedurally in a data-dependant way (with loops and conditionals), allowing them to change dynamically as a function of the input data fed to them. It's really very much like a regular progam, except it's parameterized, automatically differentiated, and trainable/optimizable. Dynamic networks have become increasingly popular (particularly for NLP), thanks to deep learning frameworks that can handle them such as PyTorch and Chainer" [LeCun2017](https://www.facebook.com/yann.lecun/posts/10155003011462143)

This leads to argue for the birth of a new programming framework. Coming from a deep learning background, Andrej Karpathy wrote:

> Software 2.0 is written in neural network weights. No human is involved in writing this code because there are a lot of weights (typical networks might have millions), and coding directly in weights is kind of hard (I tried). Instead, we specify some constraints on the behavior of a desirable program (e.g., a dataset of input output pairs of examples) and use the computational resources at our disposal to search the program space for a program that satisfies the constraints. In the case of neural networks, we restrict the search to a continuous subset of the program space where the search process can be made (somewhat surprisingly) efficient with backpropagation and stochastic gradient descent (Karpathy2017)






## References

[Frankau2009] S. Frankau, D. Spinellis, N. Nassuphis and C. Burgard, Commercial uses: Going functional on exotic trades. Journal of Functional Programming, 19(1), 27-45. doi:10.1017/S0956796808007016, 2009

[Gosh2010] DSLs in Action, Debasish Ghosh, Manning Publications, November 2010

[Innes2017] On Machine Learning and Programming Languages, Innes et all, [https://julialang.org/blog/2017/12/ml&pl](https://julialang.org/blog/2017/12/ml&pl), 2017

[Karpathy2017] Karpathy, A., [https://medium.com/@karpathy/software-2-0-a64152b37c35](https://medium.com/@karpathy/software-2-0-a64152b37c35), 2017

[LeCun2017]LeCun, Y., [https://www.facebook.com/yann.lecun/posts/10155003011462143](https://www.facebook.com/yann.lecun/posts/10155003011462143), 2017

[McKinney2010] McKinney, W. (2010, June). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51-56). Austin, TX: SciPy.

[Patil2010] Patil, A., D. Huard and C.J. Fonnesbeck. (2010) PyMC: Bayesian Stochastic Modelling in Python. Journal of Statistical Software, 35(4), pp. 1-81
