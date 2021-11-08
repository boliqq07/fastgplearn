Introduction
==================

FastGPLearn implements Genetic Programming in Python, with a scikit-learn inspired and compatible API.
And the fastgplearn applied the ``torch`` and ``numpy`` backend for fast calculated, make it accessible for CUDA.


While Genetic Programming (GP) can be used to perform a very wide variety of tasks, fastgplearn is purposefully
constrained to solving symbolic regression problems.

Symbolic regression is a machine learning technique that aims to identify an underlying mathematical expression
that best describes a relationship. It begins by building a population of naive random formulas to represent
a relationship between known independent variables and their dependent variable targets in order to predict
new data. Each successive generation of programs is then evolved from the one that came before it by selecting
the fittest individuals from the population to undergo genetic operations.

FastGPLearn retains the familiar scikit-learn fit/predict API and works with the existing scikit-learn pipeline
and grid search modules. You can get started with fastgplearn as simply as:

>>> est = SymbolicRegressor()
>>> est.fit(X_train, y_train)
>>> y_pred = est.predict(X_test)

fastgplearn supports regression through the ``SymbolicRegressor``, binary classification with the ``SymbolicClassifier``.





