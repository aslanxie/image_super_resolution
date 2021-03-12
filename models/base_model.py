import numpy as np


class BaseModel:
    """
    Base model for super resolution
    """
    
    def predict(self, input_image_array, batch_size=10):
        """
        Processes the image array into a suitable format
        and transforms the network output in a suitable image format.
        Args:
            input_image_array: input image array.
            batch_size: for large image inferce. Number of patches processed at a time.
                Keep low and increase by_patch_of_size instead.
        Returns:
            sr_img: image output, float [0.0, 1.0].
        """    
       
        lr_img = np.expand_dims(input_image_array, axis=0)
        sr_img = self.model.predict(lr_img)[0] 
        sr_img = np.float32(sr_img.clip(0, 1))

        return sr_img