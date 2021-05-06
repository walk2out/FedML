import logging
import numpy as np
from fedml_api.standalone.fedavg.q_utils import haq_quantize_model
from fedml_api.standalone.fedavg.q_utils import Read_list

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def quant(self, b_w=8, mp_list_path=''):
        if mp_list_path is not None:
            mp_lists = Read_list(mp_list_path)
            mp_list = np.empty([len(mp_lists)], dtype = int)
            for ii in range(len(mp_lists)):
                mp_list[ii] = int(mp_lists[ii][2])
        else:
            mp_list = None

        model_quant = haq_quantize_model(self.model_trainer, b_w=b_w, mp_list=mp_list)
        self.model_trainer.set_model_params(model_quant.get_model_params())

        weights = self.model_trainer.get_model_params()
        return weights
