# Notes

links
http://stacom.cardiacatlas.org/
https://acdc.creatis.insa-lyon.fr/#
http://www.cardiacatlas.org/challenges/

(ED-ES info)[https://www.medicalnewstoday.com/articles/325498.php]


## Todo

1 - Add callbacks for training, start with end of epoch (so we can run eval)
2 - Data augmentation


## Data processig

Initial steps:

1 - convert nib to numpy array (Done)

Build dataloader:

2 - need to deal with different images sizes (done)
3 - build data loader with image normalising (done)

Later:
4 - build in data augmentation 


## Model 

1 - build model and archetecture and or find one on the interweb


## Build Training (Done)

2 - save checkpoints [docs](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

## Build Eval 

## Tensorboard  
[pytorch tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)

1 - Add metrics to tensorboard (Done - training)
2 - Add predicions (images) to tensorboard

