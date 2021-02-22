Machine Learning

- Make models that make predictions from past data/experience
- Use when:
  - problem too complex for hardcoded rules
  - unstudied/understudied phenomenon
  - automating some decision/prediction
  - problem is changing frequently
- Don&#39;t use
  - can&#39;t get right/enough data
  - does not require learning, or other ways to solve problem
  - cannot afford cost of a mistake
  - unethical
  - you need explanations
- Supervised Learning
  - learning from labeled data
  - classification vs regression
- Unsupervised Learning
  - learning from unlabeled data
  - clustering, dimensionality reduction
- Semi-supervised Learning
  - learning from partially labeled data
- Reinforcement Learning
  - agent can interact with environment, perform actions, and get rewards
- Transfer Learning
  - learning to repurpose an existing model for a new task

Workflow/Visualization

- Visualize data to potentially see what features are the best to use and how the data looks beforehand for what type of model would fit the data
- Define goals -\&gt; collect and process data -\&gt; train model -\&gt; evaluate model.
- ML cannot solve problems that require explanations of natural phenomena, such as physics

Feature Engineering Basics

- Want set of features that can be predictive but also have low correlation with each other
  - Feature Engineering **definition** : Taking raw data and turning it into a set of features that can be used in predictions and in our models.

● **How to know if you have enough data?**

○ **Plot learning curves to see how the model performs with different numbers of training examples**

○ **Try to have more examples than features (svm can work with a lot of features)**

● **Unclean Data**

○ **Data can have missing values or have noise (incorrect features)**

○ **Mitigation for missing features:**

■ **Remove missing value examples**

■ **Imputation: replace incorrect or missing values with something like the mean**

- **Numerical:**
  - Binning (aka bucketing): aggregate range of numerical values into discrete bins
  - Normalization: remap values onto a range such as [0,1] or [-1,1]
  - Standardization (aka z-score normalization): rescale feature values to follow a standard normal distribution (how? =\&gt; subtract the mean and divide by the standard deviation)
- **Categorical:**
  - One-hot encoding: turn a categorical feature with k possible values into a vector of k binary features with hamming weight 1
  - Ordinal encoding: if values are ordered/ranked, values can encoded in order (e.g., as integers)

Preprocessing Importance

- Want model to generalize well, if the model does not fit the data, it will not generalize/converge
- Removing missing/null values
- Normalize data, make it fit the model
  - **Standard scaler** : z score normalization (rescale to standard normal distribution, Process for z score norm: subtract mean and divide by standard deviation
  - (-1,1) **min max** normalization
  - 0,1 **normalization**
- **No free lunch:** No model is guaranteed to work without trying it, each model learns differently

Learning Algorithms

Need a loss/cost function, SVM has specific criterions

Also need an optimization procedure such as gradient descent

**Convex** loss functions: only a single global minimum

○ Better if a loss function is convex, because it guarantees a unique global minimum, which allows us to find the solution with least error.

Learning Types:

Supervised: Learning from labeled data

● Classification

○ Ex. SVM, KNN, Decision Tree Classifier

● Regression

○ Predicting a corresponding value/target (income based on gpa, etc)

○ Ex. SVM, Decision tree regressor, Elastic net, linear regression, Ridge Regression

● Sequence to Sequence, metric learning, learning to rank, etc

Unsupervised: Learning from unlabeled data

● Clustering, Dimensionality reduction

Semi-supervised: partially labeled data

Reinforcement Learning: learning to get the most rewards over time

Transfer Learning: repurposing an existing model

Classification:

Predicting a label

**Multiclass** : more than two classes , each class is mutually exclusive

- Done using SVM

● Samples can only be assigned one label

**Multi-label** : each instance can belong to more than one class, each label represents a different task, tasks are related

● Samples can be assigned more than one label!

**One class** : one class we want to distinguish from everything else

Parameters and Hyperparameters;

Parameters/weights determined by training data/optimization procedure

Hyper parameters set by engineer for the model, strategies include:

- Grid Search: to search all pairs for of hyperparameters in a set
- Random Search: randomly sample hyperparams

If we have an overfitted model, we should regularize to reduce complexity

- Regularization done through tuning hyperparameters- regularization constant, add cost to loss function in weights
- Can also lower complexity to lower overfittedness - lower variance, but higher bias

Question: Should data be rescaled before regularization?

● YES, min max or standardize the data before regularizing so that features are penalized equally

Train Test Validation Split

Dataset divided into 3 parts: training, test, validation

Validation used to tune hyperparameters

Generally a 70/15/15 split

Regression:

- Predict the corresponding value or target

Linear Regression:

w dot product x, prediction is

Find optimal parameters θ\* = (w\*, b\*)

**Goal** : minimize Mean Squared Error, not squared error because it&#39;s less convenient

Logistic Regression:

- Binary classification with a linear regression model
- Use regression to predict a probability (p)
- Softmax regression
  - train c classifiers to predict a logit score z for each class I then use softmax to combine into a probability distribution over c labels

![](RackMultipart20210222-4-1iry6bb_html_e772c797fdcfea9d.png)

Polynomial Regression:

Used when data is non linear, when model does not fit (high bias)

Adding features to linear models to capture non linearity

m features, want all combinations up to degree k (m+k)! / (m! k!)

Bias-Variance Tradeoff

![](RackMultipart20210222-4-1iry6bb_html_bee5cd246561d918.png)

(remember irreducible error when evaluating error on training and validation/test sets.

**Bias** - Error due to incorrect assumptions, **high = underfit**

**Variance** - Sensitivity to small variations in training data, **high = overfit**

Ideally want **low bias and low variance**

Lower bias: increase complexity, use more features

Lower variance: decrease complexity, have more training data

**Increasing model complexity =\&gt; lower bias**

**Decreasing model complexity =\&gt; lower variance**

**Generalization error** : overfit model = large generalization error (error on unseen data)

Examples:

1. Training error: 1%; validation error: 20% =\&gt; **low bias; high variance** (overfitted)

2. Training error: 20%; validation error: 21% =\&gt; **high bias; low variance** (underfitted; generalizes well)

3. Training error: 20%; validation error: 35% =\&gt; **high bias; high variance** (worst case)

4. Training error: 1%; validation error: 2% =\&gt; **low bias; low variance** (best case)

**Regularization** can be achieved through a **regularization constant** (hyperparameter) to reduce model complexity, good for overfitting

Decision Trees

- acyclic graph used to make predictions
- nonparametric model suitable for classification or regression
- tree splits to maximally distinguish between the classes
- gini impurity is a measure of this
- When to stop splitting?
  - all examples in leaves are classified
  - tree reaches some max depth
  - cannot find a feature to split
  - split does not significantly improve gini impurity or entropy
- Decision trees make no assumption about the data – so unless we control them, they tend to overfit!
- Advantages:
  - scales to large datasets easily
  - applicable to many supervised learning tasks
  - decisions are easy to understand/interpret
  - almost no data preprocessing/feature engineering required
- Drawbacks:
  - high variance
  - can lead to overfitting
  - optimal decision is an NP-complete problem

Random Forests

- Ensemble learning method, build many decision trees, use the results of all of them (avg) to make a prediction
- Utilize **bagging (more on this below)**, use a random set of features for all the different trees.
- Extra trees: even more random than forests, since they use random thresholds as well

Ensemble Learning

- Use multiple models to make predictions
  - **Hard voting:** predict **MODE** of all predictions
  - **Soft voting:** predict label with highest avg probability
    - Can only be used with percentages
  - For regressors: predict **avg/median** of all regression models
- **Bagging** : train many different models of the same type (sampling **with** replacement)
  - **Pasting** is **without** replacement
  - Applied to decision trees but can be applied to other classifiers and regressors.
  - **Lowers variance and reduces overfitting**
- **Boosting:** use many weak learners to make a strong learner (gradient boosting, adaboost)
- **Stacking:** alternative to regressors/classifiers, train model to do aggregation rather than average/majority vote in ensemble.
  - Train a meta model to combine predictions.

Metrics and Evaluation

- Use mean squared error, mean absolute error, or median absolute error for **regression\**
- **For classification: use confusion matrix**
  - False positive (type 1) - false alarms, also have true positives
  - False negative (type 2) - missed detections, also have true negatives
  - Use decision function/threshold
    - Train family of classifiers, assign scores, use decision threshold for prediction
  - Precision (positive predictive value). Recall (True positive rate)
    - Cant have both, have a **tradeoff**
      - **Navigate by setting a cutoff for either one** (optimize)
- **Many classification models actually train a family of classifiers!**

Gradient Descent

- Solves optimization problem (iterative procedure that updates parameters based on gradient)
- Minimize loss function L( **θ** ) where…
  - Theta is parameters/weights
  - n is learning rate (size of steps towards min), and ∇θL(θ) is the gradient of loss
  - Output is the parameter vector **θ**
- **Types:**
  - **Batch:** each iteration calculates gradient for the **whole dataset**
  - **Stochastic:** each iteration picks a random single example and calculates gradient for it
  - **Mini-batch Stochastic (SGD):** in each epoch, randomly shuffle the entire dataset and then process it in mini batches steps
  - (Learning schedule allows us to set learning rate for each iteration)
- Should always **rescale** features or else we might have trouble converging

Polynomial Regression:

- Type of linear regression, models relationship between independent and dependent variables by using an Nth degree polynomial.

Used when data is non linear, when linear model does not fit (high bias)

Adding features to linear models to capture non linearity

m features, want all combinations up to degree k: (m+k)! / (m! k!)

Bias-Variance Tradeoff

(remember irreducible error when evaluating error on training and validation/test sets)

**Bias** - Error due to incorrect assumptions, **high bias = underfit**

**Variance** - Sensitivity to small variations in training data, **high variance = overfit**

Ideally want **low bias and low variance**

Lower bias: increase complexity, use more features

Lower variance: decrease complexity, use more training data

**Increasing model complexity =\&gt; lower bias**

**Decreasing model complexity =\&gt; lower variance**

**Generalization error** : overfit model = large generalization error (error on unseen data)

Examples:

1. Training error: 1%; validation error: 20% =\&gt; **low bias; high variance** (overfitted)

2. Training error: 20%; validation error: 21% =\&gt; **high bias; low variance** (underfitted; generalizes well)

3. Training error: 20%; validation error: 35% =\&gt; **high bias; high variance** (worst case)

4. Training error: 1%; validation error: 2% =\&gt; **low bias; low variance** (best case)

Logistic Regression and Support Vector Machines (SVM)

- What are kernels in SVM?
  - Kernel trick: set of functions that take non-linear data and transform it to a form where it is linearly separable and can have a hyperplane applied to it
    - Transforms the dataset to a higher dimensional space
    - Data may not be linearly separable due to noise
- Should you use SVM with 100 data points (each with over 1000 features) (very complex)
  - **Yes** , SVM works well with small data sets, even if the entries are complex
- Binary classification with linear regression models?
  - Use Logistic regression
    - Still a classification model; want to minimize log loss to optimize params (use optimization for that)
- Two ways to transform multiclass classification into binary classification
  - One-vs-rest: train c (number of classes greater than 2) binary classifiers, f(i) used to classify i vs not i
  - One-vs-one: train c(c-1)/2 binary classifiers, f(i,j) used to classify class i vs class j
- What are support vectors in SVM?
  - Data points that lie the closest to the hyperplane
  - Most difficult to classify of all the data points

Decision Trees

- Nonparametric model, can be used in classification or regression
- Leaf values represent predictions, traverse from root branching with feature values to make predictions
- **Best Split** approaches
  - **Intuition:** Maximally distinguish between classes
  - **Metrics:** Gini Impurity, Entropy
- No assumptions made about data
  - Trees prone to overfitting without complexity control
- **Strategies to avoid overfitting in Decision Trees**
  - **Set a maximum depth**
  - **Restrict splitting**
  - **Prune branches after creation**

**Advantages**

- Scales to **large** data sets well
- Preprocessing/feature engineering is **very minimal**
  - No scaling needed, compatible with categorical and numerical features

**Disadvantages**

- **Optimal** Decision tree is NP Complete (no polynomial time solution)
- Complex trees overfit data very easily
- Do not do well with variation (high variance)
- Imbalance leads to bias

Random Forests

- Ensemble learning method, build many decision trees, use the results of all of them (avg) to make a prediction
- Utilize **bagging (more on this below)**, use a random set of features for all the different trees.
- Extra trees: even more random than forests, since they use random thresholds as well

Ensemble Learning

- Use multiple models to make predictions
  - **Hard voting:** predict **MODE** of all predictions
  - **Soft voting:** predict label with highest **AVG** probability
    - Can only be used with **percentages**
  - For regressors: predict **avg/median** of all regression models
- **Bagging** : train many different models of the same type (sampling **with** replacement)
  - **Pasting** is **without** replacement
  - Applied to decision trees but can be applied to other classifiers and regressors.
  - **Lowers variance and reduces overfitting**
- **Boosting:** use many weak learners to make a strong learner (gradient boosting, adaboost)
- **Stacking:** alternative to regressors/classifiers, train model to do aggregation rather than average/majority vote in ensemble.
  - Train a meta model to combine predictions from base models that fit on the training data

Metrics and Evaluation

- Use mean squared error, mean absolute error, or median absolute error for **regression**
- **For classification: use confusion matrix**
  - False positive (type 1) - false alarms, also have true positives
  - False negative (type 2) - missed detections, also have true negatives
  - Use decision function/threshold
    - Train family of classifiers, assign scores, use decision threshold for prediction
  - Precision (positive predictive value). Recall (True positive rate)
    - Cant have both, have a **tradeoff**
      - **Navigate by setting a cutoff for either one** (optimize)
    - Why is there a tradeoff? Explain the tradeoff
      - As threshold increases, precision increases and recall decreases
      - As thresh

Gradient Descent

- Solves **optimization** problem (iterative procedure that updates **parameters** based on gradient)
- Minimize loss function L( **θ** ) where…
  - Theta is parameters/weights
  - n is learning rate (size of steps towards min), and ∇θL(θ) is the gradient of loss
  - Output is the parameter vector **θ**
- **Types:**
  - **Batch:** each iteration calculates gradient for the **whole dataset**
  - **Stochastic:** each iteration picks a random single example and calculates gradient for it
  - **Mini-batch Stochastic (SGD):** in each epoch, randomly shuffle the entire dataset and then process it in mini batches steps (mix of batch and stochastic)
  - (Learning schedule allows us to set learning rate for each iteration)
    - Learning rate is the size of steps taken in the descent towards minimum
- Should always **rescale** features or else we might have trouble converging
- Effect of learning rate in gradient descent
  - High learning rate: model learns faster, but yields suboptimal weights
  - Small learning rate: takes much longer to run, but yields more optimal weights
