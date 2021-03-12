# image_super_resolution
A project to ramp up deep learning principle/framework/tools on Image Super Resolution. 

## Reference
This project copy lots codes and idea from following projects:
* https://github.com/krasserm/super-resolution
* https://github.com/idealo/image-super-resolution
* https://github.com/xinntao/BasicSR

## Network

This model based on RRDN and GAN.


## Loss and Hyperparameter

generate loss, L1 loss between HR with SR

GAN loss, discriminator loss on SR

feature loss: HR and SR pass through VGG block2_conv2/block5_conv4 L2 loss, loss value from block2_conv2 and block5_conv4 output are different levels

discriminator loss(real/fake) is for discriminator network

Firstly, the different loss isn't uniform. Secondly, model need weight parts loss to obtain expectation result. 
Hyperparameters are use to balance gen loss, gan loss and feat loss while training generator.
