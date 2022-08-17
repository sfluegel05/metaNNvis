Blog Report Cross-Framework-Introspection
---

# Introduction

Over the last years, neural networks have grown ever the more powerful and complex. Their structure allows them to be
adapted for a wide range of tasks in which the can provide great results.

One of their caveats however is the limited insight into how these results emerge. For most human users, the inside of a
neural network is little more than a blackbox which turns input data into, e.g., a classification or prediction.

This is a problem for many use cases. Starting with developers who want to improve their model's performance and need a
detailed understanding of the workings of each component in order to make the right changes. Up to end users in critical
settings, such as decision making in medicine, who have to verify a model's results and rely on adequate explanations to
draw the right conclusions.

Different methods, called introspection methods, have been developed to counteract this problem. They try to explain a
networks's behaviour either by attributing the results to certain parts of input or the network itself or alternatively
by visualizing features a network has learned. An example for the first case is the Grad-CAM method which allows to
visualize the contribution of different parts in the input toward a classification (see [Figure 1a](#figure1a)). Feature
visualization on the other hand can be achieved by optimizing a randomized input towards a certain goal such as
activating a particular neuron (see [Figure 1b](#figure1b)).


<div class="row" style="display: flex">
<div class="column" style="padding: 10px">
<img id="figure1a" src="report_images/gradcam_example.png" alt="Grad-CAM for the classes 'cat' and 'dog'" height=74%/>
<div text-align=center class="caption">Figure 1a: Grad-CAM for the classes 'cat' and 'dog'. Source: <a href="#selvaraju2017">Selvaraju et al., 2017</a></div>
</div>
<div class="column" style="padding: 10px;">
<img id="figure1b" src="report_images/feature_vis_example.png" alt="Feature Visualization for a single neuron using existing images (top) and optimization." height=74%/>
<div text-align=center class="caption">Figure 1b: Feature Visualization for a single neuron using existing images (top) and optimization. Source: <a href="#olah2017">(Olah et al., 2017)</a></div>
</div>
</div>

For the users' convenience, implementations of many of these methods have been packaged and made publicly available in
toolsets like [Captum](https://captum.ai/) or [tf-keras-vis](https://github.com/keisen/tf-keras-vis). One limitation of
these preimplemented introspection methods is that they depend on a specific framework
like [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

In practice, this may lead to situations like the following:
Imagine you have trained a model in TensorFlow and now want to apply the attribution method Integrated
Gradients [(Sudararajan, 2017)](#sundararajan2017) to it. You find out that this particular method is already
implemented in Captum, a PyTorch toolset, but not in the TensorFlow toolset tf-keras-vis. This means you have three
options: Rebuild your model in PyTorch so you can use the PyTorch toolset, implement the method yourself or not don't
use the method at all.

None of these options are optimal. It would be better by far to instead have a forth option that lets you use the
already existing Integrated Gradients implementation independent of the framework it was implemented in. Realising this
option is the idea behind Cross-Framework Introspection.

To achieve this, we have built a tool that acts as an interface between the user and the introspection methods. The user
just puts in their model and their favoured introspection method. Based on that, the tool selects an implementation,
and, if the framework of the model does not fit the implementation's framework, it translates the model and additional
arguments into the right framework. Finally, it executes the introspection method and returns the result.

Concretely, the tool supports methods from tf-keras-vis and Captum, but has also been designed to be extended to other
toolsets if needed.

In the following, we will describe the mentioned introspection toolset and the different components of the translation
process between PyTorch and TensorFlow. Afterwards, we will describe the tool's structure and functionality in more
detail. Finally, we will evaluate the tool and use it to compare different implementations of the same introspection
methods.

# Supported Toolsets

The goal of cross-framework introspection is to connect models and introspection methods from different frameworks.
Since TensorFlow, more precisely Tensorflow 2.0, and PyTorch are currently the most used machine learning frameworks,
methods and models from at least these two frameworks should be supported. Therefore, we have chosen one toolset from
each framework, Captum from PyTorch and tf-keras-vis from TensorFlow. We will take a closer look at them in the
following:

## Captum

- [(Narine et al., 2020)](#narine2020)
- open source, extendable, generic
- input / layer / neuron attribution
- gradient vs pertubation based
- not only applicable to image tasks
-

## tf-keras-vis

## Method selection

Especially Captum implements a large number of methods, 36 in total counting primary, layer and neuron variants
seperately. In order to limit the scope of this project, I have chosen prioritize the introspection methods provided by
Captum and tf-keras-vis. The result can be seen in the table below.

| Method  | Category  | Priority  |
| --- | --- | --- |
| **Captum** |||
| Integrated Gradients  |    primary, layer, neuron  | <span style="color:green">include</span> |
| Saliency  | primary | <span style="color:green">include</span> |
| DeepLift  | primary, layer, neuron  | <span style="color:green">include</span> |
| DeepLiftShap  | primary, layer, neuron  | <span style="color:red">leave out</span> |
| GradientShap  | primary, layer, neuron  | <span style="color:yellow">later</span> |
| Input X Gradient  | primary | <span style="color:green">include</span> |
| Gradient X Activation | layer | <span style="color:green">include</span> |
| Guided Backpropagation    | primary, neuron | <span style="color:red">leave out</span> |
| Guided GradCAM  | primary | <span style="color:red">leave out</span> |
| Deconvolution | primary, neuron | <span style="color:yellow">later</span> |
| Feature Ablation  | primary, layer, neuron  | <span style="color:green">include</span> |
| Occlusion | primary  | <span style="color:red">leave out</span> |
| Feature Permutation | primary | <span style="color:green">include</span> |
| Shapley Value Sampling  | primary | <span style="color:red">leave out</span> |
| Lime  | primary | <span style="color:red">leave out</span> |
| KernelShap  | primary | <span style="color:red">leave out</span> |
| LRP [todo ausschreiben] | primary, layer  | <span style="color:red">leave out</span> | 
| Conductance | layer, neuron | <span style="color:yellow">later</span> |
| Layer Activation  | layer | <span style="color:yellow">later</span> |
| Internal Influence  | layer | <span style="color:red">leave out</span> |
| GradCAM | layer | <span style="color:green">include</span> |
| Neuron Gradient | neuron  | <span style="color:yellow">later</span> |
| **tf-keras-vis** |||	
| Activation Maximization | feature visualization | <span style="color:green">include</span> |
| Vanilla Saliency / SmoothGrad | attribution | <span style="color:green">include</span> |
| GradCAM | attribution | <span style="color:green">include</span> |
| GradCAM++ | attribution | <span style="color:yellow">later</span> |
| ScoreCAM  | attribution | <span style="color:yellow">later</span> |
| LayerCAM  | attribution | <span style="color:yellow">later</span> |

_Include_ marks methods which are important and were implemented right away. Methods marked as _later_ are considered to
be useful and should be implemented as well, but are not considered essential to the project.
_Leave out_ is used for methods which are deemed not important for this project and thusly are not integrated into the
tool.

The aim of this selection is to capture a wide range of different approaches, for instance including both gradient-based
and perturbation-based methods. This also means that of similar method pairs like Feature Ablation and Occlusion,
usually only one method is included. Basic, often-used methods are given an advantage compared to more specialized ones
as well. For this reson, GradCAM for example has a higher priority than the GradCAM variants. Additionally, methods
which are known to fail sanity checks (todo cite) such as Guided Backpropagation have been excluded.

# Cross-Framework Introspection

To accomplish the task of making the selected methods from Captum accessible in TensorFlow and the tf-keras-vis methods
accessible in PyTorch we need a tool that has the following features:

- **Model translation from PyTorch to TensorFlow:** todo: add paragraphs
- **Model translation from TensorFlow to PyTorch**
- **Data Translation**
- **Interface for accessing introspection methods**

Beside these strictly neccessary features, there are a few additionally requirements for the tool:

- **Extendability**
- **Uniform interface**
- **Error handling**
- **Plotting of results**

To achieve these goals, we have decided on the structure shown in todo

<div class="row" style="display: flex">
<div class="column" style="padding: 10px;">
<img id="figure2" src="report_images/cfi_components_09-08-22.png" alt="Figure 2: Internal and external components of the cross-framework introspection tool." width=100%/>
<div text-align=center class="caption">Figure 2: Internal and external components of the cross-framework introspection tool.</div>
</div>
</div>

First, let's take a look at the external components shown on the lefthand side. These are required as part of the
translation process. Models are translated between PyTorch and TensorFlow via an indermediate representation,
the [Open Neural Network Exchange](https://onnx.ai/) (ONNX) format. ONNX has been developed specifically to be
interoperable with a wide range of different frameworks including PyTorch and TensorFlow. Another advantage are the
already existing libraries for conversion of models between ONNX and other frameworks, four of which are used here:
[**torch.onnx**](https://pytorch.org/docs/stable/onnx.html) is part of PyTorch and converts models from PyTorch to ONNX.
These ONNX models are subsequentially converted to the TensorFlow Keras format by [**
onnx2keras**](https://github.com/AxisCommunications/onnx-to-keras), a library developed
by [Axis Communications](https://www.axis.com/). However, because some changes were neccessary to fit this library into
the translation process, we are using a [forked version](https://github.com/sfluegel05/onnx-to-keras/tree/dev) of
onnx2keras.

For the translation process from Tensorflow to PyTorch, we employ [**
tensorflow-onnx**](https://github.com/onnx/tensorflow-onnx), a library directly provided by ONNX, and [**
onnx2torch**](https://pypi.org/project/onnx2torch/), which has been developed by [**enot.ai**](https://enot.ai/). All
mentioned libraries are accessible open-source.

Regarding the internal structure, note that the translation classes, as well as methods and toolsets have abstract
superclasses which provide an interface to the other functions. This ensures the tool's extensibility, since new
methods, toolsets or even complete frameworks can be added as new subclasses with minimal additional effort.

Now, let's take a look at how these components work together when a user wants to execute a Captum method with a
TensorFlow model. The execution of a tf-keras-vis method in combination with a PyTorch model works analogous. The
process is also described in [Figure 3](#figure3) (todo).



<div style="display:none">
notes:
- goals
- tool implementation (component / activity diagram?)
- relevant projects: onnx (torch.onnx, tf-onnx), onnx2keras, onnx2torch, tf-keras-vis, Captum
- refer to documentation for implementation details / usage
</div>

# Evaluation

- clever hans results
- activation maximization
- framework comparison
- tool limitations

# Conclusion

# Bibliography

- <div id="selvaraju2017">Selvaraju et al.: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Proceedings
  of the IEEE International Conference on Computer Vision (ICCV), 2017</div>
- <div id="olah2017"><a href="https://distill.pub/2017/feature-visualization/">Chris Olah, Alexander Mordvintsev, Ludwig Schubert: Feature Visualization, Distill, 2017</a></div>
- <div id="sundararajan2017"><a href="https://arxiv.org/abs/1703.01365">Mukund Sundararajan, Ankur Taly, Qiqi Yan: Axiomatic Attribution for Deep Networks, arXiv, 2017</a></div>
- <div id="narine2020"><a href="https://arxiv.org/abs/2009.07896">Kokhlikyan et al.: Captum: A Unified and Generic Model Interpretability Library for PyTorch</a></div>