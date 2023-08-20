"""
    Author: Negar Golestani
    Created: August 2023
"""


from torch.utils.data import DataLoader

from .registration_model import regModel
from .networks import*
import pandas as pd



####################################################################################################################################
class regDualModel(regModel):             
    # ----------------------------------------------------------------------------------------------------
    def run_singleEpoch(self, dataset, isTraining=False, num_workers=4, batch_size=1, shuffle=False, criterions=[]):
        criterions_ = [*criterions, *self.LOSS.metrics]                             # make sure all criterions for loss are considered to be calculated

        evalRes_epoch = pd.DataFrame()                                              # epoch eval

        target_dataset = dataset.filter(label='faxi')
        source_dataset = dataset.filter(label='hist')
        Ns, Nt = len(source_dataset), len(target_dataset)
        if Ns < Nt: source_dataset.filenames = [*source_dataset.filenames, *source_dataset.filenames[:(Nt-Ns)]]
        else: target_dataset.filenames = [*target_dataset.filenames, *target_dataset.filenames[:(Nt-Ns)]]

        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        for target_batch, source_batch in zip(target_loader, source_loader):

            if isTraining:
                self.target_batch = target_batch
                self.source_batch = source_batch

            for (batch, input_type) in zip([target_batch, source_batch], ['target','source']):
                self.set_inputs(batch)
                self.forward(isTraining=isTraining, input_type=input_type)               
                evalRes_batch = self.eval(criterions=criterions_)                       # calculate eval results 
                if isTraining: self.backward()                                          
                evalRes_epoch = evalRes_epoch.append(evalRes_batch, ignore_index=True)  

        evalRes_epoch = evalRes_epoch.mean().to_dict()         

        return evalRes_epoch                            
####################################################################################################################################

