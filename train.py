"""
    Author: Negar Golestani
    Created: August 2023
"""

import os 

from options.train_options import TrainOptions
from utils.result_logger import Logger
from models import get_models
from data import get_dataset

  
'''
Example:
    python train.py --name nn-reg-test   --dataset sharpcut-reuse  --model regDualModel  --network regCorrDualNet-vgg16-layer2-0  --loss mse+diceLoss+0.01*mmd  --val_type 0.2-split 

'''

 
#------------------------------------------------------------------------------------------
if __name__ == "__main__":
#------------------------------------------------------------------------------------------
    opt = TrainOptions().parse(save=True)   # get training options
        
    print('Creating Dataset ...')
    dataset = get_dataset(opt)   
    train_dataset_list, val_dataset_list = dataset.partition(opt.val_type)    

    for k, (train_dataset, val_dataset) in enumerate(zip(train_dataset_list, val_dataset_list)):
        print('-'*20, f'Version {k}', '-'*20)        

        print('Creating Model ...')
        model = get_models(opt, version=k)  

        print('Start training ...')
        save_dir = os.path.join(opt.save_dir, str(k))           # Save directory  
        trainRes_logger = Logger(save_dir, 'training_evalRes')
        valRes_logger = Logger(save_dir, 'validation_evalRes')

        while model.epoch < opt.num_epochs:
            # Train
            trainRes = model.train(train_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, verbose=not opt.silent)   
            trainRes_logger.log(trainRes, index=model.epoch)                                        
            model.save_states()  # save                                               

            # Validation
            if opt.validate: 
                valRes = model.validate(val_dataset, num_workers=opt.num_workers, batch_size=opt.batch_size, verbose=not opt.silent)  
                valRes_logger.log(valRes, index=model.epoch)                          
            
    print('Done!')