from time import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean

from utils.datahandler import DataHandler
from utils.logger import get_logger


class BaseTrainer:
    """
    common interface for trainer
    """
    def __init__(
        self,
        generator,
        checkpoint_dir='./ckpt/pre'
    ):
        self.generator = generator
        self.checkpoint = tf.train.Checkpoint(
            psnr=tf.Variable(-1.0),
            epoch=tf.Variable(0),
            model=generator.model
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3
        )
        
    
    def checkpoint_save(self, psnr, epoch):
        if psnr > self.checkpoint.psnr:
            self.checkpoint.psnr = psnr
            self.checkpoint.epoch = epoch
            self.checkpoint_manager.save()
    
    
    def checkpoint_restore(self):
        print('before restore:')
        print(f'psnr {self.checkpoint.psnr}, epoch {self.checkpoint.epoch}')
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        print('after restore:')
        print(f'psnr {self.checkpoint.psnr}, epoch {self.checkpoint.epoch}')
    
    
    def save_best_weights(self, name):
        #restore latest saved weights with best PSNR
        self.checkpoint_restore()
        #save weights
        self.checkpoint.model.save_weights(name + self.generator.arch_name + '.h5')
    
    
    def evaluate(self, valid_ds):
        valid_psnr = []
        for idx in range(len(valid_ds['lr'])):            
            sr_img = self.generator.predict(valid_ds['lr'][idx])
            psnr = tf.image.psnr(valid_ds['hr'][idx], sr_img, max_val=1.0)
            valid_psnr.append(psnr)
        
        return tf.reduce_mean(valid_psnr)
    
   
    


class PreTrainer(BaseTrainer):
    """
    pre train generator with MeanSquaredError loss
    """
    
    def __init__(
        self,
        generator,
        discriminator,
        feature_extractor,
        lr_train_dir,
        hr_train_dir,
        lr_valid_dir,
        hr_valid_dir,
        n_validation=None,
        flatness=0.01,
        learning_rate=0.001
    ):
        super().__init__(generator, './ckpt/pre_rrdn')
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.scale = generator.scale
        self.lr_patch_size = generator.patch_size
        self.learning_rate = learning_rate
        self.flatness = flatness
        self.n_validation = n_validation
        
        self.logger = get_logger(__name__)
        
        self.train_dh = DataHandler(
            lr_dir=lr_train_dir,
            hr_dir=hr_train_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=None,
        )
        
        self.valid_dh = DataHandler(
            lr_dir=lr_valid_dir,
            hr_dir=hr_valid_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=n_validation,
        )
        
        self.loss = MeanSquaredError()
        self.optimizer = Adam(learning_rate=learning_rate)
        
        
    def train(self, epochs, steps_per_epoch, batch_size):      
        validation_set = self.valid_dh.get_validation_set(batch_size)
        loss_mean = Mean()
        for epoch in range(epochs):
            self.logger.info('Epoch {e}/{tot_eps}'.format(e=epoch, tot_eps=epochs))
            start = time()
            for step in range(steps_per_epoch):
                batch = self.train_dh.get_batch(batch_size, flatness=self.flatness)
                
                loss = self.train_step(batch['lr'], batch['hr'])
                loss_mean(loss)
                
            loss_value = loss_mean.result()
            loss_mean.reset_states()
            self.logger.info('step {} loss {}'.format(step, loss_value))
            
            psnr = self.evaluate(validation_set)
            self.logger.info(f'validate psnr {psnr}')
            
            self.checkpoint_save(psnr, epoch)
            
            elapsed_time = time() - start
            self.logger.info('took {:10.3f}s'.format( elapsed_time))

    
    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            sr = self.generator.model(lr)
            loss_value = self.loss(sr, hr)
        gradients_of_generator = tape.gradient(loss_value, self.generator.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))

        return loss_value 



class GANTrainer(BaseTrainer):
    """
    Train as GAN
    """
    
    def __init__(
        self,
        generator,
        discriminator,
        feature_extractor,
        lr_train_dir,
        hr_train_dir,
        lr_valid_dir,
        hr_valid_dir,        
        n_validation=None,
        flatness=0.01,
        learning_rate=0.0001,
        loss_weights={'generator': 1.0, 'discriminator': 0.003, 'feature_extractor': 1 / 12}
    ):
        super().__init__(generator, './ckpt/gan_rrdn')
        self.generator = generator
        self.discriminator = discriminator
        self.feature_extractor = feature_extractor
        self.scale = generator.scale
        self.lr_patch_size = generator.patch_size
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.flatness = flatness
        self.n_validation = n_validation
       
        self.logger = get_logger(__name__)
        
        self.train_dh = DataHandler(
            lr_dir=lr_train_dir,
            hr_dir=hr_train_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=None,
        )
        
        self.valid_dh = DataHandler(
            lr_dir=lr_valid_dir,
            hr_dir=hr_valid_dir,
            patch_size=self.lr_patch_size,
            scale=self.scale,
            n_validation_samples=n_validation,
        )
        
        self.mean_absolute_error = MeanAbsoluteError()
        self.mean_squared_error = MeanSquaredError()
        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        
        self.generator_optimizer = Adam(learning_rate=self.learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=self.learning_rate)
        
   
    def train(self, epochs, steps_per_epoch, batch_size):
        validation_set = self.valid_dh.get_validation_set(batch_size)
        gan_loss_mean = Mean()
        gen_loss_mean = Mean()
        feat_loss_mean = Mean()
        disc_loss_mean = Mean()
        perc_loss_mean = Mean()
        for epoch in range(epochs):
            self.logger.info('Epoch {e}/{tot_eps}'.format(e=epoch, tot_eps=epochs))
            start = time()
            for step in range(steps_per_epoch):                
                batch = self.train_dh.get_batch(batch_size, flatness=self.flatness)                
                gan_loss, gen_loss, feat_loss, disc_loss, perc_loss = self.train_step(batch['lr'], batch['hr'])
                gan_loss_mean(gan_loss)
                gen_loss_mean(gen_loss)
                feat_loss_mean(feat_loss)
                disc_loss_mean(disc_loss)
                perc_loss_mean(perc_loss)
                
                
            self.logger.info('gan_loss {},\tgen_loss {},\tfeat_loss {},\tdisc_loss {}\tperc_loss {}'.format(
                    gan_loss_mean.result(),
                    gen_loss_mean.result(),
                    feat_loss_mean.result(),
                    disc_loss_mean.result(),
                    perc_loss_mean.result()
                ) )
            gan_loss_mean.reset_states()
            gen_loss_mean.reset_states()
            feat_loss_mean.reset_states()
            disc_loss_mean.reset_states()
            perc_loss_mean.reset_states()
            
            psnr = self.evaluate(validation_set)
            self.logger.info(f'validate psnr {psnr}')
            
            self.checkpoint_save(psnr, epoch)
            
            elapsed_time = time() - start
            self.logger.info('took {:10.3f}s'.format( elapsed_time))

        
        
    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                sr = self.generator.model(lr)
                hr_output = self.discriminator.model(hr, training=True)
                sr_output = self.discriminator.model(sr, training=True)
                
                
                gan_loss, gen_loss = self._generator_loss(hr, sr, sr_output)
                feat_loss = self._feature_loss(hr, sr)
                disc_loss = self._discriminator_loss(hr_output, sr_output)
                
                perc_loss = self.loss_weights['generator'] * gen_loss \
                            + self.loss_weights['discriminator'] * gan_loss \
                            + self.loss_weights['feature_extractor'] * feat_loss
                
            
        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.model.trainable_variables)    
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.model.trainable_variables))

        return gan_loss, gen_loss, feat_loss, disc_loss, perc_loss
                
        

    @tf.function
    def _feature_loss(self, hr, sr):
        #todo: need check what doing in tensorflow.keras.applications.vgg19 import preprocess_input
        #sr = preprocess_input(sr)
        #hr = preprocess_input(hr)
        sr_features = self.feature_extractor.model(sr)
        hr_features = self.feature_extractor.model(hr)
        loss = 0.0
        for idx in range(len(sr_features)):
            loss += self.mean_squared_error(hr_features[idx], sr_features[idx])
        return loss
    
    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss
    
    def _generator_loss(self, hr, sr, sr_out):
        #gan discriminator loss
        gan_loss = self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)
        #gen L1 loss
        gen_loss = self.mean_absolute_error(hr, sr)
        return gan_loss, gen_loss
    



            
        