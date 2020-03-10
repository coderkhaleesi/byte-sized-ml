---
layout: post
mathjax: true
title: "Conditional Probability and Conditional Expectation"
subtitle: "Basics of Conditional Probability so you handle it like a pro"
date: 2020-03-09 12:34:07 -0400
background: '/img/posts/prob.jpg'
---



<p>The reason I wrote this post is because I used to get routinely confused with finding expectations of conditional probability distributions. Part of the reason was that I didn't understand the notation properly, and part was the notation itself was pretty loose. But as I see, there are many books that use loose notation, so I decided it's up to me to get to the bottom of this and crank it out.</p>

<p>So today we are going to go on a really funn ride of conditional probability distributions. Don't worry, we are not going to use any obscure pdfs/pmfs. Just plain old rules of probability.</p>

<p>So the first question is - What is Conditional Probability? Well, it's important to understand it in words and in notation, both.

Let X, Y be jointly discrete random variables.
$$
    P_{x \vert y} (x \vert y)= P (X = x \vert Y = y) = \frac{P (x,y)}{P_{Y}(y)}
$$

In words, this is "If we restrict our sample space to pairs (x,y) with the value of y equal to the given value and then divide the original mass function $p_{y}(x,y)$ with $p(y)$, we get a pmf on the restricted space, which is the conditional probability mass function (in discrete case)."

Similarly, we can think about the continuous case although not as crudely. But for easy transference, we just take the same formula.

$$ f_{x \vert y} (x \vert y)= f(X = x \vert Y = y) = \frac{f(x,y)}{f_{Y}(y)} $$

So really think about the sample space getting restricted as you are already given the value of y. Now the function $f(x \vert y)$ is a probability distribution of the conditional measure of $X$ given $Y=y$. For continuous, just think in similar terms, just not Y=y (as not possible in discrete), but $Y \leq y$. Those familiar with measure theory might be cringing right now at my crude explanation, but I hope to get my hands dirty with measure theory someday. Right now, I got to finish the article and study for my assignment.</p>

<p>Now let's raise our expectations...I mean move to Conditional Expectation.

What comes to mind when we hear Expectation or see $E[X]$ in the world of probability? Something related to average value I suppose? It's actually an average of all the values the random variable can take weighted against the probability of that random variable taking those values. Simple, right?

So then what is this ? $$ E[X \vert Y] $$

Well, in words this the Expectation of the random variable <strong>X</strong> in the conditional probability measure for <strong>X given Y</strong>.
So in other words, it is the average of values that X takes weighted against the conditional probability of X given Y.</p>

<p>
More formally,

Discrete case:
$$
E[X \vert Y=y] = \sum_x P(X = x \vert Y = y) = \sum_x xP_{x\vert y}(x\vert y)
$$

Continuous case:
$$
E[X \vert Y=y] = \int_{-\infty}^{+\infty} xf_{x\vert y}(x\vert y) = \int_{-\infty}^{+\infty} \frac{xf(x, y)}{f_{Y}(y)}
$$


Now let's do a small exercise to consolidate our learning, and then we can all relax knowing we have learned something concrete and which is going to help us throughout our life. (Too cheesy?)

What is $E[E[X\vert Y]]$? (assume X and Y to be discrete random variables)

Let's try it out.
So from what we learned it, let's first write down the inner expectation.

$$
E[ \sum_x xP_{x\vert y}(x\vert y)]
$$
Now, since this is an average over X, this will be a function of y. So the outer expectation can be taken as the expectation of a function of y over y. (which we know how to calculate)

Let's write out the whole expression now,

$$
E[g(Y)] =  \sum_y \sum_x x p_{x\vert y}(x\vert y) p_{Y}(y)
\implies  \sum_y \sum_x x \frac{p(x, y)}{p_{Y}(y)} p_{Y}(y)
\implies \sum_y \sum_x x p(x, y)
\implies  \sum_x x \sum_y p(x, y)
\text{Hint: Marginal Distribution}
\implies \sum_x x p(x)
\implies E[X]
$$

Food for thought: From the above exercise, what we expect X to be after Y is the same as before. What does this tell you about X and Y?
</p>

<h4>References</h4>
<ul>
    <li> <a href="https://ocw.mit.edu/courses/mathematics/18-440-probability-and-random-variables-spring-2014/lecture-notes/MIT18_440S14_Lecture26.pdf">MIT Lecture Slides</a></li>
</ul>

<blockquote class="blockquote">Happy Coding!!!!</blockquote>
<p>Placeholder text by <a href="http://spaceipsum.com/">Space Ipsum</a>. Photographs by <a href="https://unsplash.com/">Unsplash</a>.</p>
