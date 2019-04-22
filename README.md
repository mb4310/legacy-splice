# Neural-Splicing
N.B This repo contains legacy code for splicing


In this repo, we introduce, implement, and analyze a novel method of transfer learning in deep neural architectures. 

### Introduction

Transfer learning has seen much success in the field of computer vision and more recently natural language processing. Specific approaches vary considerably, here we focus on extending a commonly used technique in which some of the weights and biases of a pre-trained model are loaded into a new architecture which is repurposed to some new tasks. This method has long since seen success in computer vision tasks and more recently in the field of natural language processing, popularized by Jeremy Howard and Sebastian Ruder in their paper introducing [ULMFiT ](https://arxiv.org/abs/1801.06146) and in Jeremy's [courses at fast.ai](https://course.fast.ai/) and later extended upon by folks at [google](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)and [open.ai](https://blog.openai.com/language-unsupervised/) all together resulting in many state-of-the-art NLP results.

Broadly speaking, this approach can be thought of as viewing a pretrained model as being composed of two pieces: a general-purpose *body* or *core* and a task-specific *head*. Thus far, applications of this approach involve replacing an existing *head* with a new one tailored for a specific task. Typically, in CV/NLP the core consists of the convolutional/recurrent components respectively, whereas the head is consist of pooling/flattening layers followed by a shallow multi-layer perceptron.

Here we investigate the question: why is in each of these cases the new component is added only **on top of** existing components? Can anything be gained by **splicing new components directly into the core architecture**?

Let us illustrate witha concrete example: suppose we has trained the following MLP for some regression task:

```
m = (nn.Sequential(nn.Linear(2,5), 
                   nn.ReLU(), 
                   nn.Linear(5,9), 
                   nn.ReLU(), 
                   nn.Linear(9,1)))
```

and one would like to add a layer *between* the first and second linear layer:

```
m_splice = (nn.Sequential(nn.Linear(2,5), 
                          nn.ReLU(), 
                          nn.Linear(5,5),      #Additional, un-trained component
                          nn.ReLU(), 
                          nn.Linear(5,9), 
                          nn.ReLU(), 
                          nn.Linear(9,1)))
```

Some observations: if the weights of the newly introduced layer are initialized randomly, then we can surely expect that model performance will collapse. Since the original layers were trained together, layer two (nn.Linear(5,9)) *expects* inputs to look a certain way, (e.g look like something layer one would have output). Essentially this degradation would be the result of an extreme case of covariate shift which would defeat the purpose of using the pre-trained weights.

On the other hand, if we initialize the bias to zero and the weights to identity matrix, then the new layer has *no effect* on the model initially. Then if we freeze the learning rate on all the original layers, we can think of learning the weights of our new component as learning a post-processing of layer one output to be more 'palatable' for the rest of the network to consume in a similar spirit as batch-norm is introduced to post-process the output of layers to have desirable sample statistics. In practice, we anneal the learning rate on the layers surrounding the splice gradually from zero to some stable level while annealing the learning rate on the spliced component along a one cycle schedule as introduced [here](https://arxiv.org/abs/1708.07120). By monitoring gradient norms through the added component(s) as well as the original layers, one can propose a metric for determining the extent to which the architecture has 'accepted' or 'rejected' the new components. 


### Goals

In this repo, we focus for now on convolutional and linear splicing and postpone experiments with splicing into recurrent architectures to a later date. Specifically, we are interested in exploring the following questions:

1) If we begin by training an architecture (A1) to some stop criterion (where reduction in loss is minimal, or overfitting begins to occur, etc.) and then splice in components to achieve architecture (A2), are we able to train (A2) further to achieve an overall lower loss (or else better performance by some specified metric)? 

2) Consider fully trained architecture (A2) above, along with full training history (pre- and post- splice, beginning from scratch). Make a copy of the architecture denoted (B2) and re-initialize weights randomly and train (B2) from scratch. We are interested:
* Does (B2) achieve worse, the same, or better *ultimate* performance (with a pre-defined stopping criterion specified a priori) as compared to (A2)
* How does the complete training history of (B2) compare to that of (A2)? 
        
       
An affirmative response to question (1) suggests a cheap, fast procedure for incrementally increasing a trained model's performance if slight increase in performance is necessary for an application. 

A positive response to question (2) suggests alternative paradigms for training deep neural networks, rather than initializing a very deep network completely at random, perhaps it is faster, more efficient, or ultimately result in a better performance if one begins with a shallower network and gradually adds depth, splicing in layers over time and training them gradually. 
