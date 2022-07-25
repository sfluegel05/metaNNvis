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
visualize the contribution of different parts in the input toward a classification (see Figure 1). Feature visualization
on the other hand can be achieved by optimizing a randomized input towards a certain goal such as activating a
particular neuron (see Figure 2).


<div class="row" style="display: flex">
<div class="column" style="padding: 10px">
<img src="report_images/gradcam_example.png" alt="Grad-CAM for the classes 'cat' and 'dog'" height=74%/>
<div text-align=center class="caption">Figure 1a: Grad-CAM for the classes 'cat' and 'dog'. Source: [1]</div>
</div>
<div class="column" style="padding: 10px;">
<img src="report_images/feature_vis_example.png" alt="Feature Visualization for a single neuron using existing images (top) and optimization." height=74%/>
<div text-align=center class="caption">Figure 1b: Feature Visualization for a single neuron using existing images (top) and optimization. Source: [2]</div>
</div>
</div>

For the users' convenience, implementations of many of these methods have been packaged and made publicly available in
toolsets like [Captum](https://captum.ai/) or [tf-keras-vis](https://github.com/keisen/tf-keras-vis). One limitation of
these preimplemented introspection methods is that they depend on a specific framework
like [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

In practice, this may lead to situations like the following:
Imagine you have trained a model in TensorFlow and now want to apply the attribution method Integrated Gradients [3] to
it. You find out that this particular method is already implemented in Captum, a PyTorch toolset, but not in the
TensorFlow toolset tf-keras-vis. This means you have three options: Rebuild your model in PyTorch so you can use the
PyTorch toolset, implement the method yourself or not don't use the method at all.

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

# "Related Work"

- relevant projects: onnx (torch.onnx, tf-onnx), onnx2keras, onnx2torch, tf-keras-vis, Captum

# Cross-Framework Introspection

- goals
- tool implementation (component / activity diagram?)
- refer to documentation for implementation details / usage

# Evaluation

- clever hans results
- activation maximization
- framework comparison
- tool limitations

# Conclusion

# Bibliography

- [1] Selvaraju et. al.: Grad-cam: Visual explanations from deep networks via gradient-based localization, Proceedings
  of the IEEE International Conference on Computer Vision (ICCV), 2017
- [2] [Chris Olah, Alexander Mordvintsev, Ludwig Schubert: Feature Visualization, Distill, 2017](https://distill.pub/2017/feature-visualization/)
- [3] [Mukund Sundararajan, Ankur Taly, Qiqi Yan: Axiomaticc Attribution for Deep Networks, arXiv, 2017](https://arxiv.org/abs/1703.01365)