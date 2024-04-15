# Translation_Machine# Project Name <--- FIXME

Authors:
- Github: [ktoan911](https://github.com/ktoan911) 
- Email: khanhtoan.forwork@gmail.com 

Advisors:
- Github: [bangoc123](https://github.com/bangoc123) 
- Email: protonxai@gmail.com



Implementation of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) . This
library is part of our project: Building an Machine Translation Model library with ProtonX.

<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

This project is a Vietnamese - English translation machine using a Transformer model. It aims to provide accurate translations, bridging language barriers and meeting the demand for efficient translation tools.

Slide about your project (if it's available) <--- **FIXME**

## Architecture Image 

![image](assets/model_architect.png)


## I.  Set up environment
- Step 1: create a Conda environment named your_env_name with Python version 3.11.5

```python
conda create -n ${your_env_name} python=3.11.5
```

- Step 2: Activate the newly created environment using the following command
```
conda activate ${your_env_name}
```

- Step 3: Install Packages from requirements.txt

```
pip install -r requirements.txt
``` 

## II.  Set up your dataset

-  
<--- **FIXME**
- References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer)

## III. Training Process

There are some important arguments for the script you should consider when running it:

- `vocab-size`
- `max-length-input`
- `embedding-dim`
- `num-heads-attention`: [3.2.2](https://arxiv.org/pdf/1706.03762.pdf)
- `dff`: [3.3](https://arxiv.org/pdf/1706.03762.pdf)
- `num-encoder_layers` : [3.1](https://arxiv.org/pdf/1706.03762.pdf)
- `d-model`: [3.2.2](https://arxiv.org/pdf/1706.03762.pdf)
- `batch-size`
- `epochs` 
- `dropout-rate`

Training script:


```python

python train.py --vocab-size ${vocab-size} --max-length-input ${max-length-input} --embedding-dim ${embedding-dim} --num-heads-attention ${num-heads-attention} --dff ${dff} --num-encoder-layers ${num-encoder-layers} --d-model ${d-model} --batch-size ${batch-size} --epochs ${epochs} --learning-rate ${learning-rate} --dropout-rate ${dropout-rate}

```

Example:

```python

!python train.py --vocab-size 10000 --max-length-input 200 --embedding-dim 32 --num-heads-attention 3 --dff 512 --num-encoder-layers 6 --d-model 128 --batch-size 32 --epochs 10 --learning-rate 0.01 --dropout-rate 0.1

``` 
**FIXME**

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- ...

## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

**FIXME**

Your implementation
```
Epoch 7/10
782/782 [==============================] - 261s 334ms/step - loss: 0.8315 - acc: 0.8565 - val_loss: 0.8357 - val_acc: 0.7978
Epoch 8/10
782/782 [==============================] - 261s 334ms/step - loss: 0.3182 - acc: 0.8930 - val_loss: 0.6161 - val_acc: 0.8047
Epoch 9/10
782/782 [==============================] - 261s 333ms/step - loss: 1.1965 - acc: 0.8946 - val_loss: 3.9842 - val_acc: 0.7855
Epoch 10/10
782/782 [==============================] - 261s 333ms/step - loss: 0.4717 - acc: 0.8878 - val_loss: 0.4894 - val_acc: 0.8262

```

**FIXME**

Other architecture

```
Epoch 6/10
391/391 [==============================] - 115s 292ms/step - loss: 0.1999 - acc: 0.9277 - val_loss: 0.4719 - val_acc: 0.8130
Epoch 7/10
391/391 [==============================] - 114s 291ms/step - loss: 0.1526 - acc: 0.9494 - val_loss: 0.5224 - val_acc: 0.8318
Epoch 8/10
391/391 [==============================] - 115s 293ms/step - loss: 0.1441 - acc: 0.9513 - val_loss: 0.5811 - val_acc: 0.7875
```

Your comments about these results <--- **FIXME**


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


