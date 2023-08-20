"""
    Author: Negar Golestani
    Created: August 2023
"""



import os
import pandas as pd

####################################################################################################################################
class Logger(object):
    def __init__(self, save_dir, filename, autosave=True):
        self.autosave = autosave

        self.save_path = os.path.join(save_dir, f"{filename}.csv")
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        self.reset()
    # ----------------------------------------------------------------------------------------------------
    def log(self, log_dict, index=None):        
        # Get index
        idxs = self.logs_df.index
        if index is None: 
            index = idxs.max()+1 if len(idxs)>0 else 0
        elif index in idxs: 
            self.remove(index)

        self.logs_df = self.logs_df.append( pd.Series(log_dict, name=index) )

        if self.autosave: self.save()     
    # ----------------------------------------------------------------------------------------------------
    def save(self):
        self.logs_df.to_csv(self.save_path)  
    # ----------------------------------------------------------------------------------------------------
    def remove(self, index=None):
        if index is None:
            index = self.logs_df.index[-1]

        self.logs_df = self.logs_df.drop(index)
    # ----------------------------------------------------------------------------------------------------
    def reset(self):
        self.logs_df = pd.DataFrame()
        self.logs_df.index.names = ['index']
####################################################################################################################################

