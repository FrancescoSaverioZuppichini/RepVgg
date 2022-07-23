# Implementing RepVGG in PyTorch
## Make your cnn > 100x faster!

Hello There!! Today we’ll see how to implement RepVGG in PyTorch proposed in [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/pdf/2101.03697.pdf)

Code is [here](https://github.com/FrancescoSaverioZuppichini/RepVgg), an interactive version of this article can be downloaded from [here](https://github.com/FrancescoSaverioZuppichini/RepVgg/blob/main/README.ipynb).

Let’s get started!

The paper proposed a new architecture that can be tuned after training to make it faster on modern hardware. And by faster I mean lighting fast, this idea was used by [Apple's MobileOne model].(https://arxiv.org/abs/2206.04040).

![img](images/fig1.png)


## Single vs Multi Branch Models

A lot of recent models use multi-branching, where the input is passed through different layers and then aggregated somehow (usually with addition).

![img](images/single_multi_branch.png)



This is great because it makes the multi-branch model an implicit ensemble of numerous shallower models. More specifically, *the model can be interpreted as an ensemble of 2^n models since every block branches the flow into two paths.*

Unfortunately, multi-branch models consume more memory and are slower compared to single branch ones. Let's create a classic `ResNetBlock` to see why.


https://gist.github.com/15f8c164d13fa7925db12b4e498f61d6

Storing the `residual` double memory consumption. This is also shown in the following image from the paper

![img](images/fig3.png)


The authors noticed that the **multi branch architecture is useful only at train time**. Thus, if we can have a way to remove it at test time we can improve the model speed and memory consumption.


### From Multi Branches to Single Branch

Consider the following situation, you have two branches composed of two `3x3` convs


https://gist.github.com/501c5d44d62018c9197ff5b32d1808c3


https://gist.github.com/0d8b5ee9116d9fe8669a9389c2333807




    torch.Size([1, 8, 5, 5])



Now, we can create one conv, let's call it `conv_fused` such that `conv_fused(x) = conv1(x) + conv2(x)`. Very easily, we can just sum up the `weight`s and the `bias` of the two convs! Thus we only need to run one `conv` instead of two.


https://gist.github.com/2e0ee5400f75688ddab5ce724329d86b

Let's see how much faster it is!


https://gist.github.com/4e1289c934383729fc584d18602d99ee

    conv1(x) + conv2(x) tooks 0.000421s
    conv_fused(x) tooks 0.000215s


Almost `50%` less (keep in mind this is a very naive benchmark, we will see a better one later on)

### Fuse Conv and BatchNorm

In modern network architectures, `BatchNorm` is used as a regularization layer after a convolution block. We may want to fused them together, so create a conv such that `conv_fused(x) = batchnorm(conv(x))`. The idea is to change the weights of `conv` in order to incorporation the shifting and scaling from `BatchNorm`. 

The paper explains it as follows:

![img](images/eq3.png)

The code is the following:


https://gist.github.com/42056dee938e5c694d5ea3caca64833f

Let's see if it works


https://gist.github.com/ad69227a26f3701a3958d7fa787523a6

yes, we fused a `Conv2d` and a `BatchNorm2d` layer. There is also an [article from PyTorch about this](https://pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html) 


So our goal is to fuse all the branches in one single conv, making the network faster! 

The author proposed a new type of block, called `RepVGG`. Similar to ResNet, it has a shortcut but it also has an identity connection (or better branch).


![img](images/fig2.png)

In PyTorch:


https://gist.github.com/7f4f2c56500b8c8ac2dffa6ddd37b2f1

#### Reparametrization

We have one `3x3` `conv->bn`, one `1x1` `conv-bn` and (somethimes) one `batchnorm` (the identity branch). We want to fused them together to create one single `conv_fused` such that `conv_fused` = `3x3conv-bn(x) + 1x1conv-bn(x) + bn(x)` or if we don't have an identity connection, `conv_fused` = `3x3conv-bn(x) + 1x1conv-bn(x)`.

Let's go step by step. To create `conv_fused` we have to:
- fuse the `3x3conv-bn(x)` into one `3x3conv`
- `1x1conv-bn(x)`, then convert it to a `3x3conv`
- convert the identity, `bn`, to a `3x3conv`
- add all the three `3x3conv`s

Summarized by the image below:

![img](images/fig4.png)

The first step it's easy, we can use `get_fused_bn_to_conv_state_dict` on `RepVGGBlock.block` (the main `3x3 conv-bn`).

The second step is similar, `get_fused_bn_to_conv_state_dict` on `RepVGGBlock.shortcut` (the `1x1 conv-bn`). Then we pad each kernel of the fused `1x1` by `1` in each dimension creating a `3x3`.

The identity `bn` is trickier. We need to create a `3x3` `conv` that will act as an identity function and then use `get_fused_bn_to_conv_state_dict` to fuse it with the identity `bn`. This can be done by having `1` in the center of the corresponding kernel for that corresponding channel. 

Recall that a conv's weight is a tensor of `in_channels, out_channels, kernel_h, kernel_w`. If we want to create an identity conv, such that `conv(x) = x`, we need to have one single `1` for that channel.

For example:


https://gist.github.com/c499d53431d243e9fc811f394f95aa05

    torch.Size([2, 2, 3, 3])
    Parameter containing:
    tensor([[[[0., 0., 0.],
              [0., 1., 0.],
              [0., 0., 0.]],
    
             [[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]]],
    
    
            [[[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]],
    
             [[0., 0., 0.],
              [0., 1., 0.],
              [0., 0., 0.]]]], requires_grad=True)


See, we created a `Conv` that acts like an identity function. 

Now, putting everything together, this step if formally called **reparametrization**


https://gist.github.com/b55edcbd0ef1c2afc9f17f9bedd847d1

Finally, let's define a `RepVGGFastBlock`. It's only composed by a `conv + relu` 🤩


https://gist.github.com/4dcecf0b35442d370df8917f09cf37c1

and add a `to_fast` method to `RepVGGBlock` to quickly create the correct `RepVGGFastBlock`


https://gist.github.com/74d09d878fdc07edb10e1dc88e26b3b6

### RepVGG

Let's define `RepVGGStage` (collection of blocks) and `RepVGG` with an handy `switch_to_fast` method that will swith to the fast block in-place:


https://gist.github.com/0dd6293691dfc8fd07398a4276278b5c

### Let's test it out! 

I've created a benchmark inside `benchmark.py`, running the model on my **gtx 1080ti** with different batch sizes and this is the result:


The model has two layers per stage, four stages and widths of `64, 128, 256, 512`. 

In their paper, they scale these values by some amount (called `a` and `b`) and they used grouped convs. Since we are more interested in the reparametrization part, I skip them.

![img](images/time.png)


Yeah, so basically the reparametrization model is on a different scaled time compared to the vanilla one. Wow!

Let me copy and pasted the dataframe I used to store the benchmark

https://gist.github.com/1b872f62575ad030b04801a85dacba2e

You can see that the default model (multi branch) tooks `1.45`s for a `batch_size=128` while the reparamitizated one (fast) only took `0.0134`s.That is **108x** 🚀🚀🚀.

## Conclusions

Conclusions
In this article we have seen, step by step, how to create RepVGG; a blaziling fast model using a smart reparameterization technique. 

This technique can be ported to other architecture as well.

Thank you for reading it!

Francesco


