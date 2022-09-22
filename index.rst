Cross-Framework Introspection
=============================

Cross-Framework Introspection is a tool for accessing introspection
methods for neural networks regardless of the framework in which the
neural network has been built. It is easily extendable and currently
supports models from TensorFlow 2.0 and PyTorch in combination with
methods from `Captum <captum.ai>`__ and
`tf-keras-vis <https://github.com/keisen/tf-keras-vis>`__. For more
details, see the `project
report <https://github.com/sfluegel05/metaNNvis/blob/main/results/report.md>`__.

Installation
------------

The latest version of metaNNvis can be installed via pip:

``pip install metaNNvis``

Usage
-----

For instructions on how to use cross-framework introspection and how to
extend it by new methods, see the `user
guide <https://github.com/sfluegel05/metaNNvis/blob/main/results/user_guide.ipynb>`__.

Available methods
-----------------

Cross-Framework Introspection currently supports most methods from
Captum and all methods from tf-keras-vis. The supported methods are:

============================= ======================
Method                        Category
============================= ======================
**Captum**                    
Integrated Gradients          primary, layer, neuron
Saliency                      primary
DeepLift                      primary, layer, neuron
GradientShap                  primary, layer, neuron
Input X Gradient              primary
Gradient X Activation         layer
Deconvolution                 primary, neuron
Feature Ablation              primary, layer, neuron
Feature Permutation           primary
Conductance                   layer, neuron
Layer Activation              layer
GradCAM                       layer
Neuron Gradient               neuron
**tf-keras-vis**              
Activation Maximization       feature visualization
Vanilla Saliency / SmoothGrad attribution
GradCAM                       attribution
GradCAM++                     attribution
ScoreCAM                      attribution
LayerCAM                      attribution
============================= ======================

Currently not supported are:

=========================== ======================
Method                      Category
=========================== ======================
**Captum**                  
DeepLiftShap                primary, layer, neuron
Guided Backpropagation      primary, neuron
Guided GradCAM              primary
Occlusion                   primary
Shapley Value Sampling      primary
Lime                        primary
KernelShap                  primary
Layer Relevance Propagation primary, layer
Internal Influence          layer
=========================== ======================
