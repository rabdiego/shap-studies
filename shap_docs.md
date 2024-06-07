# [AI4Welness] SHAP Reference Documentation
> by Diego Rabelo

A quick guide on how to use tools from the SHAP library for explainabilty in machine learning tasks. For reading this document, we're expecting you to be already familiar with the concepts from Explainable AI (xAI).

## Summary
1. Shapley Values - The basics
2. SHAP plots
3. SHAP and the Explainer class
4. Different types of explainers
5. Working with Autogluon


## 1. Shapley Values - The basics

One of the main challenges for xAI is to be **model agnostic**, that is, work with any type of model of any implementation you could find. Despite all the differences between a SVM and a DNN, we can still define a machine learning model of arbitrary complexity as a function that receives an input (the attributes) and produces an output (the labels). As we don't necesseraly know how this function works (that's why we want a model agnostic approach), we could call it a **black box** function.

The way SHAP works with those black box functions is by computing the **shapley values** of them. The concept of shapley values comes from game theory, that is beyond the scope of this document, so let's just focus on some key points, using some analogies:

Imagine that you are the boss of four software developers (let's call them Diego, Said, Renê and Felipe) working in a single application, that can be rated in some way. You want to split a budget of value $X$ between them, based on the importance of each one to the final product quality that they're working on.  
Without any regards to work ethics and being a nice boss, and also imagining you have the powers of space-time bending, you have an idea. If I could exclude a worker $w_i$ from our staff $S$, and see how the quality of the product developed without such worker is, you could say how important this worker is to your project.  
If wee could apply this to every worker we have, then we could assign values $v_i$ to each worker $w_i$. And that's *almost* what shapley values are and how they're calculated.  
The thing is, if you excluded Renê from the original staff, the product's quality could drop significantly, but you can't just praise Renê for that. Maybe Diego and Said just hate each other, and Renê was the one that could distract them from fighting during work hours. So, if you instead excluded Renê and Said, the product's rating could be much bigger, if not even higher than the original value.  
What I'm saying is that those combinations of whose workers you should exclude from the staff aren't necesseraly linear independent, so you must check **all** possible combinations. And if you know some discrete mathematics, you can guess that this problem is already on the time complexity of $O(2^n)$. And that's the issue SHAP tries to solve, optimizing the computation of those shapley values.

But wait, what exactly does this have to do with explainable AI? Well, imagine that instead of *workers*, we're thinking of *attributes* from the data, and instead of a *product quality* we're thinking of a final *prediction* from a model. Now everything makes sense!

## 2. SHAP plots

We are currently using two types of plots, one being the **Beeswarm** plot, for global inference, and the **Waterfall** plot, for individual inference. See the original documentation for [beeswarm](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html) and [waterfall](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html) plots.

## 3. SHAP and the Explainer class 

The most basic way you can use SHAP, regardless of which model you are using, is with the Explainer class:

`explainer = shap.Explainer( ... )`

Here we initialized a explainer object (note that this won't calculate the shappley values yet). This is a perfect model-agnostic approach, as it takes as a parameter not a model object, but instead a method (function):

`explainer = shap.Explainer(model.predict, ... )`

In that way, you have the liberty of using any library or implementation of any model with SHAP!  
Now, we need to pass our data as a parameter too:

`explainer = shap.Explainer(model.predict, X_train[:, :100])`

As you may see, I only passed a subset of our train data into the explainer object, as it doesn't need all the data to work. This data passed is just a way for the explainer to "train", and not to calculate the shappley values. To do that, we can simply write:

`shap_values = explainer(X_train)`

And it will return an object with all the information needed for plotting waterfalls and beeswarm plots!

`shap.plots.beeswarm(shap_values)`  
`shap.plots.waterfall(shap_values[0])`

But if you try to use SHAP as I just did, you'll notice how long it takes to compute the shappley values. That's because we're using the basic `Explainer` class that tries to compute them by the traditional method, and as you remember, it is pretty slow. To solve this problem, SHAP gives us a plenty of different types of explainers to work better with some models, or to optimize if you really want to use an agnostic approach.

## 4. Different types of explainers

For this project, there are four types of explainers that we are insterested with, those being:

- `shap.KernelExplainer`, for model-agnostic approaches,
- `shap.TreeExplainer`, for tree-based models like XGBoost, sklearn's RandomForest and so on,
- `shap.LinearExplainer`, for sklearn's linear (and logistic) regression models, and
- `shap.DeepExplainer`, for PyTorch neural networks.

The first three explainers act similarly to `shap.Explainer`, but instead of passing a method, you pass the model object directly:

`model = xgb.XGBoostClassifier`  
`model.fit(X_train, y_train)` 
`explainer = shap.TreeExplainer(model, X_train[:, :100])`  
`shap_values = explainer(X_train)`

Unfortunetly, the DeepExplainer has some details that we need to address if we want it to work with PyTorch.

First, PyTorch doesn't work directly with Pandas dataframe, instead, it works with its proper Tensor class. For this reason, SHAP expects the data argument to be a Torch.Tensor object. So you need to convert the data before passing it to a DeepExplainer object:

`torch_data = torch.from_numpy(X_train.values)`

Note that we loose the attribute names doing that! This problem we will address later.  
With the data being Torch.Tensors, we can do:

`explainer = shap.DeepExplainer(model, torch_data)`  
`shap_values = explainer.shap_values(torch_data)`

Note that we are using a different method to compute the shappley values, and that changes everything. I didn't mentioned before, but for the other explainers, they compute not only the shappley values, but also the explanation object that will help the plotting methods for... plotting.  
There are a lot of information in those plots that aren't available in plain shappley values, such as the attribute's names, the attribute's range of values, and so on. We usually want them all the time, and for that reason, the majority of explainers gives us them without we direclty asking for.  

As PyTorch uses a different type of data, this isn't possible for us automatically, and then we need to do some things manually. There are four things we need to build an Explanation object, and luckly for us, we already have half of them: the shappley values, and the data. We still need to compute the base value and get the feature names.

The feature names are a simple matter, just get them from the original Pandas Dataframe stored in a list: `feat_names = list(X_train.columns)`  
As for the base value (the average model prediction for a dataset), we can simply store the model's prediction of the entire dataset in a tensor and get the mean of it:

`with torch.no_grad():`
`   outputs = model(x_train)`
`   probas = torch.softmax(outputs, 1)[:, 1]`
`   base_value = probas.mean().item()`

Finally, we can build the explanation object:

`exp = shap.Explanation(shap_values[1], data=torch_data.numpy(), base_values=base_value, feature_names=feat_names)`

And use it while plotting just as we were using the other `shap_values` objects.

Note that for autogluon's pytorch models for binary classification, they use two output neurons, one for each class. That explains why we are passing `shap_values[1]` for the explanation object, as they are the shappley values of the true class.

## 5. Working with autogluon

Finally, we need to address a crucial part that is using SHAP with Autogluon.

For choosing the best explainer to work with the model that autogluon choose as the best, we need to indentify it. To get the best model's name, you can use the following *gambiarra*:

`trainer = predictor._learner.load_trainer()`  
`model_name = trainer._get_best()`

Where the `predictor` variable is an autogluon predictor. After choosing the best explainer, we need to get the "raw" model object from the autogluon predictor: SHAP doens't understand that an autogluon predictor, or any object from the autogluon API, is a XGBoost or a PyTorch neural network. Because of that, we need to dig further into the autogluon predictor's pipeline to get the model that has the correct type to pass into SHAP:

`model_autogluon = trainer.load_model(model_name=trainer._get_best())`  
`raw_model = model_autogluon.model`

And then you can pass the `raw_model` variable into your explainer of choice.