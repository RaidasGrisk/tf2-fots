# tf2-fots

I no longer work on this project.  
To sum things up, in order for this to work better:
1. Use bigger model, MobileNet does not seem to suffice.
2. Better training data, with different text angle, size and length.
3. Clever loss functions and combinations of losses of different model branches.

Okay, here's a gif showing the current state of the model.   
It ain't much but it's honest work ¯\\\_(ツ)\_/¯

![Gif](misc/gif.gif?raw=true)

In progress. Current state:

![Loss](misc/loss.bmp?raw=true)

The loss of detection part started to flatten,  
so I experimented with the relative weights of  
losses from both branches when computing grads.  

This results in shifting the trade-off between  
what will the model learn more: detect or recognize.  
That is why the sudden drop in detection loss.  

# TODO

- [x] Backbone model (resnet/mobilenet) feature extraction + fusion
- [x] Text detection part
- [x] RoI rotate part
- [x] Text recognition part
- [x] Data generation
- [ ] Fix detection box bug during inference

The problem is the code is still very messy and poorly structured.  
Given that the main project pipeline is working, next steps are:

- [ ] Clean and restructure the code
- [ ] See if RoiRotatePart can be improved using affine transformation

Once above is finished:

- [ ] Explore data augmentation
- [ ] Training/testing pipelines
- [ ] Documentation
