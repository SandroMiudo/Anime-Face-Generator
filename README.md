# Generative Anime Face Model
This project focusses on training a generative model, to generate realistic anime faces.

![Training Data](media/images.png)

## Prerequisites

**Make sure to execute the init script, before starting.**

```
./scripts/init.sh

source ~/.alias

conda create -f conda_env.yml [-n custom]

conda activate (gan_anime_faces | custom) 
```

## Run

There are three distinct routines that can be executed, which are specified in the sub-sections below.

To train the model with your custom parameters, just follow the [train](#train) sub-section.

To load the model, including its state, parameters and more just refer to this [sub-section](#load).

If you are just interested in generating images, refer to the [inference](#inference) sub-section.

**Available targets:**
- **Seq-1:**
    - [64, 128, 256, 512]
- **Seq-2:**
    - [48, 96]
- **Seq-3:**
    - [56, 112]

**The default target is 64! Meaning that if you want to have a different target, make sure to specifiy the desired target.**

### Train

If the **train routine** is the desired routine, it is required to specify `--train` when invoking the script.

In this training section we'll just cover the basic parameters that can be used.

```shell
ENTRY --train [--epochs EPOCHS] [--batch-size B_SIZE] [--noise-vector N_VECTOR] [--generate-per-epoch N] [--learning-rate A[, B]] [--no-augment] [--target X]
```

A different approach is just executing the train script. The easiest way of executing this script, would just be to pass in `S, M or L` with its corresponding level `{1..3}`. This will generate the final command on the fly, abstracting away the inner arguments.

**Note** : S = Small , M = Medium , L = Large is used as acroynm to describe the amount of time for training. I.e S is the routine which takes the least time.

```shell
TRAIN [[S,{1..3} | M,{1..3} | L,{1..3}] | [[[D,{E,C,P},{1..3}] | [LR,{1..3}]], [E,{1..3}], [N,{1..3}], [B,{1..3}]] | [-h]]
```
If you to want to customize your learning rate over epochs, consider using [learning rate schedulers](#using-learning-rate-schedulers).

Gathering more information about the prior script invocation ? No problem just use the `-h` option.

### Load

Specify `--load` to indicate that the **loading routine** is the desired routine.

```shell
ENTRY --load [--epochs N] [--generate-per-epoch K] [--no-augment] [--target X]
```

If you don't want to write out the options every time, just consider using the load script, which takes in 3 arguments, or optionally 4 arguments.

```shell
LOAD <generate-images> <epochs> <no-augment> [target]
```

### Inference

Using `--inference` signals the **last** of the three routines.

```shell
ENTRY --inference [--inference-count N] [--target X]
```

Optionally there is also an inference script, which can be executed by just passing in the amount of images to generate (or by passing in the target as well).

```shell
INF <inference_count> [target]
```

### Using learning rate schedulers

If you want to customize the learning rate over epochs, consider using the available learning rate schedulers listed below.

**Note** : It is always required to specify either the basic learning rate or the learning rate scheduler. Using both in conjuction will lead to undefined state.

Available learning rate schedulers : 

+ Exponential Decay : 
    + `--exponential-decay`
+ Polynomial Decay :
    + `--polynomial-decay`
+ Constant Decay : 
    + `--constant-decay`
        + `--constant-decay-boundaries`
        + `--constant-decay-values`

To use a common learning rate scheduler for both, i.e generator and discriminator just pass in the required arguments for the specific decay. However if the contrary is the case, and you want to have distinct decays, i.e discriminator and generator should have its own set of parameters you stack them on top of each other, whereas the first set is used for the generator and the latter for the discriminator correspondigly.  

**Exponential Decay**

```shell
--exponential-decay <initial_learning_rate=float> <decay_steps=integer> <base=float> [<opt1> <opt2> <opt3>]
```

---

**Polynomial Decay**

```shell
--polynomial-decay <initial_learning_rate=float> <decay_steps=integer> <end_learning_rate=float> <power=float> [<opt1> <opt2> <opt3> <opt4>]
```

---

**Constant Decay**

The only build, that differs from the others above, is that this stacking is not allowed in here.
In this build it is required to specify in the `--constant-decay` option, if the arguments in `--constant-decay-boundaries` and `--constant-decay-values` are used for both in common or seperately. Also it is required that the **len(list of arguments)**, specified by `--constant-decay-values` is always `N+1`.

```shell
--constant-decay {1..2} --constant-decay-boundaries [1..N] --constant-decay-values [1..N+1]
```

## Showcase

Currently there are 3 (which can be extended) total models at disposal (Targets : 56, 64 & 96).

**Important note : These models were not trained with the best available options, but instead use a configuration for the lowest possible execution time! Meaning for those, who desire better performance on a specific target, might want to train it with a custom configuration using this [train-doc](#train) as reference point.**

### Target 56

<table style="margin-left: auto; margin-right: auto">
    <tr>
        <td><img src="media/generation/inf-1-target-56.jpg" width="450px"></td>
        <td><img src="media/generation/inf-2-target-56.jpg" width="450px"></td>
    </tr>
    <tr>
        <td><img src="media/generation/inf-3-target-56.jpg" width="450px"></td>
        <td><img src="media/generation/inf-4-target-56.jpg" width="450px"></td>
    </tr>
</table>

### Target 64

<table style="margin-left: auto; margin-right: auto">
    <tr>
        <td><img src="media/generation/inf-1-target-64.jpg" width="450px"></td>
        <td><img src="media/generation/inf-2-target-64.jpg" width="450px"></td>
    </tr>
    <tr>
        <td><img src="media/generation/inf-3-target-64.jpg" width="450px"></td>
        <td><img src="media/generation/inf-4-target-64.jpg" width="450px"></td>
    </tr>
</table>

### Target 96

<table style="margin-left: auto; margin-right: auto">
    <tr>
        <td><img src="media/generation/inf-1-target-96.jpg" width="450px"></td>
        <td><img src="media/generation/inf-2-target-96.jpg" width="450px"></td>
    </tr>
    <tr>
        <td><img src="media/generation/inf-3-target-96.jpg" width="450px"></td>
        <td><img src="media/generation/inf-4-target-96.jpg" width="450px"></td>
    </tr>
</table>

### Target 128

<table style="margin-left: auto; margin-right: auto">
    <tr>
        <td><img src="media/generation/inf-1-target-128.jpg" width="450px"></td>
        <td><img src="media/generation/inf-2-target-128.jpg" width="450px"></td>
    </tr>
    <tr>
        <td><img src="media/generation/inf-3-target-128.jpg" width="450px"></td>
        <td><img src="media/generation/inf-4-target-128.jpg" width="450px"></td>
    </tr>
</table>