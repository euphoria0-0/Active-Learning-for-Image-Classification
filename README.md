# Active-Learning-for-Image-Classification

This repository is the unofficial implementation of Active Learning baseline algorithms for image classification.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python main.py -AL random -D CIFAR10 --data_dir <path_to_data> --batch_size 128 --lr 0.1 --lrscheduler multistep --milestone 160 --num_epoch 200 --resize 32 --init kaiming
```


- ```-AL```: implemented AL method 
  - ```fixed```: AL with fixed indices
  - ```random```: random selection
  - ```coreset```: core set selection [[paper](https://arxiv.org/abs/1708.00489)]
  - ```vaal```: VAAL [[paper](https://arxiv.org/abs/1904.00370)]
    - ```num_epoch_vaal``` (default 100) 
  - ```learningloss```: learning loss [[paper](https://arxiv.org/abs/1905.03677)]
    - ```subset_size``` (default 10000)
    - ```epoch_loss``` (default 120)
    - ```margin``` (default 1.0)
    - ```weight``` (default 1.0)
  - ```ws```: weight decay scheduling [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710426.pdf)]
    - ```ws_sampling_type```
  - ```badge```: BADGE selection [[paper](https://arxiv.org/abs/1906.03671)]
  - ```seqgcn```: sequential GCN [[paper](https://arxiv.org/abs/2006.10219)]
    - ```subset_size``` (default 10000)
    - ```lambda_loss``` (default 1.2)
    - ```s_margin``` (default 0.1)
    - ```hidden_units``` (default 128)
    - ```dropout_rate``` (default 0.3)
    - ```lr_gcn``` (default 0.001)
  - ```tavaal```: TA-VAAL[[paper](https://arxiv.org/abs/2002.04709)]
    - ```num_epoch_vaal``` (default 100)
    - ```weight``` (default 1.0)
    - ```subset_size``` (default 10000)
  - ```bait```: BAIT [[paper](https://arxiv.org/abs/2106.09675)]
  - ```alfamix```: ALFA-Mix [[paper](https://arxiv.org/abs/2203.07034)]
  - ```gradnorm```: AL GradNorm [[paper](https://arxiv.org/abs/2112.05683)]
    - ```subset_size``` (default 10000)


## Results

Our model achieves the following performance on:

### Image Classification on CIFAR10

| Model name \ Accuracy over labeled set | 600 | 800 | 1000 |
| ------------------ |---------------- | -------------- | -------------- |
| Ours   |  .  |  .  |  .  |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing
MIT License
>📋  Pick a licence and describe how to contribute to your code repository. 


## References:
1. https://github.com/ej0cl6/deep-active-learning
2. https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
3. https://github.com/JordanAsh/badge
4. https://github.com/sinhasam/vaal
5. https://github.com/cubeyoung/TA-VAAL
6. https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
7. https://github.com/AminParvaneh/alpha_mix_active_learning
8. https://github.com/xulabs/aitom/tree/master/aitom/ml/active_learning/al_gradnorm
