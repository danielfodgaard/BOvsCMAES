/*
* The Rubber Article Template.
*
* Here is a quick run-down of the template.
* Some example content has been added for you to see what the template looks like and how it works.
* Some features of this template are explained here, so you might want to check it out.
*/

#import "@preview/rubber-article:0.5.0": *


// Layout and styling
#show: article.with(
  cols: none, // Tip: use #colbreak() instead of #pagebreak() to seamlessly toggle columns
  eq-chapterwise: true,

  eq-numbering: "(1.1)",
  lang: "en",
  page-margins: 1.5in,
  page-paper: "a4",
)
#show link: underline
#set heading(numbering: none)
#set par(first-line-indent: 0em)

// Frontmatter
#maketitle(title: "BO vs CMA-ES", authors: ("Daniel Fodgaard \n daniel_fodgaard@u.nus.edu", ), date: datetime
  .today()
  .display("[month repr:long] [year]"))

// Actual Content starts here.
#let ip(x, y) = $lr(chevron.l #x, #y chevron.r)$

= Motivation
An optimiser never sees a problem directly; it sees one particular description of it. The same objective can be reported as a loss or as its logarithm, the same parameter can be measured in seconds or in milliseconds, and the same design can be written in coordinates that are rotated relative to each other. None of these choices changes where the optimum is, so ideally none of them should change how well an optimiser finds it.

Hansen’s tutorial on CMA-ES treats this as a design principle. It lists the transformations CMA-ES is invariant to: any order-preserving transformation of the objective, and rotations and rescalings of the search space, provided the starting state is transformed accordingly. I found the first of these striking. Because CMA-ES uses only the ranking of function values, f, log f and f⁴ are literally the same problem to it. Bayesian optimisation is built differently. It fits a Gaussian process to the values of f, so how f is reported changes the model, and potentially the search. Its default kernel also has one lengthscale per coordinate axis, which ties it to whatever coordinate system the user happened to choose.

This raises a practical question: do these invariances actually hold in the implementations people use (pycma and BoTorch), and when they don’t, what does it cost? In this report I test each invariance the same way: I run each method on a problem and on a transformed version of the same problem. The transformation changes neither the optimum nor the problem itself, so any difference in performance is caused by the method.

