Introduction
==================

FastGPLearn implements Genetic Programming in Python, with a ``scikit-learn`` inspired and compatible API.
And the fastgplearn applied the ``torch`` and ``numpy`` backend for fast calculated, make it accessible for ``CUDA`` .


While Genetic Programming (GP) can be used to perform a very wide variety of tasks, FastGPLearn is purposefully
constrained to solving symbolic regression problems.

Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression
that best describes a relationship. It begins by building a population of naive random formulas to represent
a relationship between known independent variables and their dependent variable targets in order to predict
new data. Each successive generation of programs is then evolved from the one that came before it by selecting
the fittest individuals from the population to undergo genetic operations.

The optional operator including (``'add'``, ``'sub'``, ``'mul'``, ``'div'``, ``'max'``, ``'min'``, ``'ln'``, ``'exp'``, ``'pow2'``, ``'pow3'``,
``'rec'``,  ``'sin'``, ``'cos'``).

FastGPLearn retains the familiar ``scikit-learn`` **fit**/**predict** API. You can get started with FastGPLearn as simply as:


>>> from fastgplearn.skflow import SymbolicRegressor
>>> est = SymbolicRegressor()
>>> est.fit(X_train, y_train)
>>> y_test_pred = est.predict(X_test)
>>> test_score = est.score(X_test,y_test)

FastGPLearn supports regression through the :py:class:`fastgplearn.skflow.SymbolicRegressor` ,
binary classification with the :py:class:`fastgplearn.skflow.SymbolicClassifier` .

**NOW, TRY IT !**



