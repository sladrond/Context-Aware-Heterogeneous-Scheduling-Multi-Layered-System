import torch
from torchvision import transforms
import os
import zmq, pickle, zlib
import time
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, DistilBertConfig
from transformers import RobertaTokenizer, RobertaForQuestionAnswering, RobertaConfig
from transformers import SegformerFeatureExtractor, SegformerForImageClassification,SegformerForSemanticSegmentation
from efficientnet_pytorch import EfficientNet
import gc
import csv
from ddqn_scheduler import actions

class Dispatcher:
    def __init__(self):
        self.no_context=True
        self.models_saved = {}
        self.models_loaded = {}
        self.loading_delays = {}
        self.inference_delays = []
        self.preprocessing_delays = []
        self.batch_number = 0
        self.feature_extractor = None
        self.path_to_models=''

    def offload_model_task(self, processor, type_task):
        to_remove=[]
        for key in self.models_loaded:
            if processor+'-'+str(type_task) in key:
                to_remove.append(key)

        for element in to_remove:
            self.remove_model(element)
            print("removed "+key)

    def clean_cuda(self):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def print_params(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2

    def is_loaded(self, model_label, processor,type_task):
        if processor == 'gpu' or processor == 'cpu':
            model_name = model_label + '-' + processor + '-' + str(type_task)
        try:
            self.models_loaded[model_name]
            return True

        except KeyError:
            return False

    def try_again(self, model, key):
        try:
            self.models_loaded[key] = model
            self.models_loaded[key].cuda()
            self.models_loaded[key].eval()
            return True
        except:
            return False

    def load_model_in_processor(self, model_label, processor, type_task):
        if type_task == 0:
            if processor == 'cpu':
                self.models_loaded[model_label + '-cpu-'+str(type_task)] = EfficientNet.from_pretrained(model_label)
                self.models_loaded[model_label + '-cpu-'+str(type_task)].eval()
                return True
            else:
                try:
                    self.models_loaded[model_label + '-gpu-'+str(type_task)] = EfficientNet.from_pretrained(model_label)
                    self.models_loaded[model_label + '-gpu-'+str(type_task)].cuda()
                    self.models_loaded[model_label + '-gpu-'+str(type_task)].eval()
                    return True
                except:
                    keys_to_remove = [key for key in self.models_loaded if 'gpu' in key]
                    for key in keys_to_remove: self.remove_model(key)
                    key = model_label + '-gpu-' + str(type_task)
                    return self.try_again(EfficientNet.from_pretrained(model_label), key)

        elif type_task == 1:
            if processor == 'cpu':
                if model_label == 'roberta-base':
                    config = RobertaConfig.from_pretrained(model_label)
                    state_dict = torch.load(self.path_to_models+model_label+'.pt')
                    self.models_loaded[model_label + '-cpu-'+str(type_task)] = RobertaForQuestionAnswering(config)
                    self.models_loaded[model_label + '-cpu-' + str(type_task)].load_state_dict(state_dict)
                else:
                    config = DistilBertConfig.from_pretrained(model_label)
                    state_dict = torch.load(self.path_to_models + model_label + '.pt')
                    self.models_loaded[model_label + '-cpu-'+str(type_task)] = DistilBertForQuestionAnswering(config)
                    self.models_loaded[model_label + '-cpu-' + str(type_task)].load_state_dict(state_dict)
                self.models_loaded[model_label + '-cpu-'+str(type_task)].eval()
            else:
                try:
                    if model_label == 'roberta-base':
                        config = RobertaConfig.from_pretrained(model_label)
                        state_dict = torch.load(self.path_to_models + model_label + '.pt')
                        self.models_loaded[model_label + '-gpu-' + str(type_task)] = RobertaForQuestionAnswering(config)
                        self.models_loaded[model_label + '-gpu-' + str(type_task)].load_state_dict(state_dict)
                        self.models_loaded[model_label + '-gpu-'+str(type_task)].cuda()
                    else:
                        config = DistilBertConfig.from_pretrained(model_label)
                        state_dict = torch.load(self.path_to_models + model_label + '.pt')
                        self.models_loaded[model_label + '-gpu-' + str(type_task)] = DistilBertForQuestionAnswering(
                            config)
                        self.models_loaded[model_label + '-gpu-' + str(type_task)].load_state_dict(state_dict)
                    self.models_loaded[model_label + '-gpu-'+str(type_task)].eval()
                except:
                    keys_to_remove = [key for key in self.models_loaded if 'gpu' in key]
                    for key in keys_to_remove: self.remove_model(key)
                    return self.try_again(self.models_loaded[model_label + '-gpu-' + str(type_task)].
                                          load_state_dict(state_dict), model_label + '-gpu-'+str(type_task))
            del config
            del state_dict

        elif type_task == 3:
            self.feature_extractor = SegformerFeatureExtractor.\
                from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            if processor == 'cpu':
                self.models_loaded[model_label + '-cpu-'+str(type_task)] = SegformerForSemanticSegmentation.\
                    from_pretrained(model_label)
                self.models_loaded[model_label + '-cpu-'+str(type_task)].eval()
                return True
            else:
                try:
                    self.models_loaded[model_label + '-gpu-'+str(type_task)] = SegformerForSemanticSegmentation.\
                        from_pretrained(model_label)
                    self.models_loaded[model_label + '-gpu-'+str(type_task)].cuda()
                    self.models_loaded[model_label + '-gpu-'+str(type_task)].eval()
                    return True
                except:
                    keys_to_remove = [key for key in self.models_loaded if 'gpu' in key]
                    for key in keys_to_remove: self.remove_model(key)
                    key = model_label + '-gpu-' + str(type_task)
                    return self.try_again(SegformerForSemanticSegmentation.from_pretrained(model_label), key)

    def send_task(self, tasks_list, ipadd, port):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://"+ipadd+":"+port)
        p = pickle.dumps(tasks_list)
        z = zlib.compress(p)
        socket.send(z)
        return (len(socket.recv())>0)

    def get_model_fts(self):
        tot_mod = len(self.models_loaded)
        cpu_cnt = 0;
        gpu_cnt = 0

        if tot_mod == 0:
            return np.zeros(4, )
        else:
            size_mod = np.zeros(tot_mod)
            for idx, element in enumerate(self.models_loaded):
                if element[len(element) - 3:] == 'cpu':
                    cpu_cnt += 1
                else:
                    gpu_cnt += 1
                model = self.models_loaded[element]
                param_size = 0
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                buffer_size = 0
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()

                size_all_mb = (param_size + buffer_size) / 1024 ** 2
                size_mod[idx] = size_all_mb

            prc_cpu = [cpu_cnt if cpu_cnt == 0 else cpu_cnt / tot_mod][0]
            prc_gpu = [gpu_cnt if gpu_cnt == 0 else gpu_cnt / tot_mod][0]
            size_mod = np.sum(size_mod) / tot_mod
            return np.array([tot_mod, prc_cpu, prc_gpu, size_mod])

    def get_model_size(self, model_label):
        for key in actions.MODEL.keys():
            if key == model_label:
                try:
                    temp = actions.MODEL[key](key)
                    param_size = 0
                    for param in temp.parameters():
                        param_size += param.nelement() * param.element_size()
                    buffer_size = 0
                    for buffer in temp.buffers():
                        buffer_size += buffer.nelement() * buffer.element_size()

                    size_all_mb = (param_size + buffer_size) / 1024 ** 2
                    return size_all_mb
                except:
                    return None

    def save_model(self, path, model_label, processor):
        if processor == 'cpu' or processor == 'gpu':
            print("local")

        elif processor == 'edge':
            os.system("scp " + path + " sladrond@cognet-testbed1.ics.uci.edu:/home/sladrond/models")
            self.models_saved[model_label] = path
        else:
            print("no")

    def offload_model(self, model_label):
        self.models_loaded.pop(model_label)
        # self.model =

    def tfms(self, image):
        res = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        return res(image)

    def predict_image(self, list_images, model_label, processor):
        if processor == 'gpu':
            for image in list_images:
                try:
                    self.models_loaded[model_label + '-gpu-0'].eval()
                    with torch.no_grad():
                        outputs = self.models_loaded[model_label + '-gpu-0'](image.cuda())
                        idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()
                        prob = torch.softmax(outputs, dim=1)[0, idx].item()
                        del outputs
                except Exception as e:
                    print("Cuda exception was catched predict image")
                    print(e)
                    self.clean_cuda()
        else:
            for image in list_images:
                with torch.no_grad():
                    outputs = self.models_loaded[model_label + '-cpu-0'](image)
                    idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()
                    prob = torch.softmax(outputs, dim=1)[0, idx].item()
                    del outputs

        del list_images

    def predict_class(self, list_images, model_label, processor):
        if processor == 'gpu':
            for image in list_images:
                try:
                    self.models_loaded[model_label + '-gpu-3'].eval()
                    inputs = self.feature_extractor(np.array(image), return_tensors="pt")
                    inputs.to('cuda:0')
                    with torch.no_grad():
                        outputs = self.models_loaded[model_label + '-gpu-3'](**inputs).logits
                        del outputs
                except Exception as e:
                    print("Cuda exception was catched predict image")
                    print(e)
                    self.clean_cuda()
        else:
            for image in list_images:
                inputs = self.feature_extractor(image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.models_loaded[model_label + '-cpu-3'](**inputs).logits
                    del outputs

        del list_images

    def execute_task_laptop(self, task_list, model_label, processor, batch_number, stats):
        self.batch_number = batch_number
        batch_size = 0
        list_images = [];
        list_type_3=[]
        start_img_time = 0;
        start_type_3_time=0;
        for task in task_list:
            if task.get_type() == 0:
                start_img_time=time.time()
                list_images.append(task.get_data())
            elif task.get_type() == 3:
                start_type_3_time = time.time()
                list_type_3.append(task.get_data())

        if len(list_images):
            type_task = 0
            self.predict(list_images, model_label, type_task, processor)
            start = start_img_time
            end = time.time()
            #print("ADDED STATS: ", [len(list_images), type_task, (end - start) / len(list_images)])
            stats.put([len(list_images), type_task, (end - start) / len(list_images)])

        if len(list_type_3):
            type_task = 3
            self.predict(list_type_3, model_label, type_task, processor)
            start = start_type_3_time
            end = time.time()
            #print("ADDED STATS: ", [len(list_type_3), type_task, (end - start) / len(list_type_3)])
            stats.put([len(list_type_3), type_task, (end - start) / len(list_type_3)])

        del list_images
        del list_type_3

    def predict(self, list_objects, model_label, type_task, processor):
        if type_task == 0:
            return self.predict_image(list_objects, model_label, processor)
        elif type_task == 3:
            return self.predict_class(list_objects, model_label, processor)
        else:
            print("not valid type, I cannot predict")

    def get_model(self, model_label):
        model = self.models_loaded.get(model_label, None)
        if model is not None:
            return model
        else:
            print("This model is not loaded")
            return model

    def remove_model(self, model):
        del self.models_loaded[model]
        gc.collect()
        print("model ", model)
        if 'gpu' in model:
            self.clean_cuda()

    def remove_models(self):
        for element in self.models_loaded:
            del element
        gc.collect()
        self.clean_cuda()

    def get_loading_delays(self):
        return self.loading_delays

    def get_inference_delays(self):
        return self.inference_delays

    def get_processing_delays(self):
        return self.preprocessing_delays
