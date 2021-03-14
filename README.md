# image_super_resolution
A project to ramp up deep learning principle/framework/tools on Image Super Resolution. 

## Reference
This project copy lots codes and idea from following projects:
* https://github.com/krasserm/super-resolution
* https://github.com/idealo/image-super-resolution
* https://github.com/xinntao/BasicSR

## Network

This model base on RRDN and GAN.


## Loss and Hyperparameter

generate loss, L1 loss between HR with SR

GAN loss, discriminator loss on SR

feature loss: HR and SR pass through VGG block2_conv2/block5_conv4 L2 loss, loss value from block2_conv2 and block5_conv4 output are different levels

discriminator loss(real/fake) is for discriminator network

Firstly, the different loss isn't uniform. Secondly, model need weight parts loss to obtain expectation result. 
Hyperparameters are use to balance gen loss, gan loss and feat loss while training generator.

## Test

Tained without L1 or L2 loss, and compared visual quality on 55' TV (crop 960x540 area from orignal, X2, X3, X4 and merged to 1920x1080 picture, then show on 4K resolution with full screen). Yes, the same as lots of super resluiton papaer said, and got a very suppirsing visual quality. At the same time, I found some obviously artificial, especifiaclly in high rate scale. Some texture detail lost in low resolution cannot be properly recovered, too.  
