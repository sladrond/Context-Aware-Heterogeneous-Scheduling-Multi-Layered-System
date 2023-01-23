from queue import Queue
import torch
import torchvision.models as models
import json
from PIL import Image
import PIL
from torchvision import transforms
import os
import zmq
import time
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from efficientnet_pytorch import EfficientNet
import gc
import csv
from ddqn_scheduler import actions
import torch.multiprocessing as mp
from numba import cuda
labels_map = json.load(open('../data/images/labels_map.txt'))

class Dispatcher:
    def __init__(self):

        self.models_saved = {}
        self.models_loaded = {}
        self.loading_delays = {}
        self.inference_delays = []
        self.preprocessing_delays = []
        self.batch_number = 0
        # self.stats = mp.Queue()

        pass

    def clean_cuda(self):
        torch.cuda.empty_cache()
        # cuda.select_device(0)
        # cuda.close()
        # cuda.select_device(0)

    def print_params(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('model size: {:.3f}MB'.format(size_all_mb))

    def is_loaded(self, model_label, processor, type_task):
        if processor == 'gpu' or processor == 'cpu':
            model_name = model_label + '-' + processor + type_task

        try:
            self.models_loaded[model_name]
            return True

        except KeyError:
            print(model_name)
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
        # print("loading model ", model_label, " to processor ", processor, " type of task ", type_task)
        if type_task == 0:
            if processor == 'cpu':
                # start = time.time()
                self.models_loaded[model_label + '-cpu-' + type_task] = EfficientNet.from_pretrained(model_label)
                # end = time.time()
                self.models_loaded[model_label + '-cpu-' + type_task].eval()
                # self.loading_delays = {'processor':'cpu','start_loading':start,'end_loading':end,'type_task':0}
                # print(model_label+'-cpu has been stored')
                # self.print_params(self.models_loaded[model_label+'-cpu'])
                return True
            else:
                try:
                    # start = time.time()
                    self.models_loaded[model_label + '-gpu-' + type_task] = EfficientNet.from_pretrained(model_label)
                    # end = time.time()
                    self.models_loaded[model_label + '-gpu-' + type_task].cuda()
                    self.models_loaded[model_label + '-gpu-' + type_task].eval()
                    # print(model_label+'-gpu has been stored')
                    # self.print_params(self.models_loaded[model_label+'-gpu'])
                    # self.loading_delays = {'processor':'gpu','start_loading':start,'end_loading':end,'type_task':0}
                    return True
                except:
                    keys_to_remove = [key for key in self.models_loaded if 'gpu' in key]
                    for key in keys_to_remove: self.remove_model(key)
                    key = model_label + '-gpu-' + type_task
                    return self.try_again(EfficientNet.from_pretrained(model_label), key)


        elif type_task == 1:
            if processor == 'cpu':
                # start = time.time()
                if model_label == "roberta-base":
                    self.models_loaded[model_label + '-cpu-' + type_task] = RobertaForQuestionAnswering.from_pretrained(
                        "roberta-base")
                else:
                    self.models_loaded[
                        model_label + '-cpu-' + type_task] = DistilBertForQuestionAnswering.from_pretrained(
                        model_label)
                # end = time.time()
                self.models_loaded[model_label + '-cpu-' + type_task].eval()
                # self.loading_delays = {'processor':'cpu','start_loading':start,'end_loading':end, 'type_task':1}
                # print(model_label+'-cpu has been stored')
                # self.print_params(self.models_loaded[model_label+'-cpu'])
            else:
                if model_label == "roberta-base":
                    model = RobertaForQuestionAnswering.from_pretrained("roberta-base")
                else:
                    model = DistilBertForQuestionAnswering.from_pretrained(model_label)
                try:
                    # start = time.time()
                    self.models_loaded[model_label + '-gpu-' + type_task] = model.cuda()
                    # end = time.time()
                    # self.models_loaded[model_label+'-gpu'].cuda()
                    self.models_loaded[model_label + '-gpu-' + type_task].eval()
                    # print(model_label+'-gpu has been stored')
                    # self.print_params(self.models_loaded[model_label+'-gpu'])
                    # self.loading_delays = {'processor':'gpu','start_loading':start,'end_loading':end, 'type_task':1}
                except:
                    keys_to_remove = [key for key in self.models_loaded if 'gpu' in key]
                    for key in keys_to_remove: self.remove_model(key)
                    return self.try_again(model, model_label + '-gpu-' + type_task)


    def load_model(self, model_label):
        start = time.time()
        self.models_loaded[model_label] = EfficientNet.from_pretrained(model_label)
        end = time.time()
        self.models_loaded[model_label].eval()
        self.loading_delays = {'processor': 'cpu', 'start_loading': start, 'end_loading': end}

        if self.processor == 'edge':
            model_loaded = self.models_loaded.get(model_label,None)
            model_saved = self.models_saved.get(model_label,None)

            if model_saved is not None and model_loaded is None:
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.connect("tcp://128.195.54.86:5555")
                socket.send(bytes(self.edge_models_saved[model_label],'utf-8'))
                message = socket.recv()
                if message == 'loaded':
                    self.models_loaded[model_label] = 'true'
                else:
                    print("Something is wrong with the server connection")
        else:
            print("Either the model is not saved or it has been loaded")



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

    def preprocess_text(self, list_texts, processor, model_label):
        print("Start preprocessing text ", len(list_texts), " tasks ")
        new_list = []
        if model_label == "distilbert-base-uncased":
            tokenizer = DistilBertTokenizer.from_pretrained(model_label)
        else:
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        if processor == 'gpu':
            for text in list_texts:
                question = text[0]
                final_text = text[1]
                inputs = tokenizer(question, final_text, return_tensors="pt")
                new_list.append(inputs)

        else:
            for text in list_texts:
                question = text[0]
                final_text = text[1]
                inputs = tokenizer(question, final_text, return_tensors="pt")
                new_list.append(inputs)
        print("Finish preprocessing text ", len(list_texts), " tasks ")
        del list_texts
        return new_list


    def preprocess_image(self, list_images, processor):
        new_list = []
        print("Start preprocessing images ", len(list_images), " tasks ")
        if processor == 'gpu':
            for image in list_images:
                # new_image = Image.fromarray(image)
                # new_list.append(self.tfms(new_image).unsqueeze(0).cuda())
                try:
                    new_list.append(image.cuda(non_blocking=True))
                except:
                    print(".............didn't send to cuda.........................")

        else:
            for image in list_images:
                # new_image = Image.fromarray(image)
                # new_list.append(self.tfms(new_image).unsqueeze(0))
                new_list.append(image)

        print("Finish preprocessing images ", len(list_images), " tasks ")
        del list_images
        return new_list


    def predict_text(self, list_texts, model_label, processor):
        print("Start prediction of text, model: ", model_label, " processor: ", processor)
        if processor == 'gpu':
            for inputs in list_texts:
                print("The inference in GPU will start")
                start_positions = torch.tensor([1]).cuda()
                end_positions = torch.tensor([3]).cuda()
                inputs.to('cuda:0')
                model = self.models_loaded[model_label + '-gpu-1']
                print("model is in: ", next(model.parameters()).device)
                print("inputs is in: ", inputs.is_cuda)
                print("inputs is in: ", inputs.is_cuda)
                print("start_positions is in: ", start_positions.is_cuda)
                print("end_positions is in: ", end_positions.is_cuda)
                try:
                    with torch.no_grad():
                        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
                        # print("Measured delay of " +model_label)
                        del outputs
                    del inputs
                    # print("Inference delay: ", (end-start))
                    self.clean_cuda()
                except Exception as e:
                    print("Cuda exception was catched predict text")
                    print(e)
                    self.clean_cuda()

        else:
            start_positions = torch.tensor([1])
            end_positions = torch.tensor([3])
            for inputs in list_texts:
                with torch.no_grad():
                    outputs = self.models_loaded[model_label + '-cpu'](**inputs, start_positions=start_positions,
                                                                       end_positions=end_positions)
                    # print("Measured delay of " +model_label)
                    del outputs
                # list_texts.remove(inputs)
                # print("Inference delay: ", (end-start))


    def predict_text_with_files(self, list_texts, model_label, processor):
        print("Start prediction of text, model: ", model_label, " processor: ", processor)
        if processor == 'gpu':
            with open('delays_' + processor + '_' + str(self.batch_number) + '.csv', 'w') as csvfile:
                fieldnames = ['processor', 'start_inference', 'end_inference', 'size', 'start_loading', 'end_loading',
                              'type_task']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # print(self.loading_delays)
                writer.writerow(self.loading_delays)
                for inputs in list_texts:
                    # print("The inference in GPU will start")
                    with torch.no_grad():
                        start_positions = torch.tensor([1]).cuda(non_blocking=True)
                        end_positions = torch.tensor([3]).cuda(non_blocking=True)
                        # inputs.to('cuda:0')
                        start = time.time()
                        model = self.models_loaded[model_label + '-gpu']
                        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
                        end = time.time()
                        # print("Measured delay of " +model_label)
                    del outputs
                    writer.writerow({'processor': processor, 'start_inference': start, 'end_inference': end, 'size': 0,
                                     'type_task': 1})
                    del inputs
                    # print("Inference delay: ", (end-start))
                    self.clean_cuda()

        else:
            start_positions = torch.tensor([1])
            end_positions = torch.tensor([3])
            with open('delays_' + processor + '_' + str(self.batch_number) + '.csv', 'w') as csvfile:
                fieldnames = ['processor', 'start_inference', 'end_inference', 'size', 'start_loading', 'end_loading',
                              'type_task']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # print(self.loading_delays)
                writer.writerow(self.loading_delays)
                for inputs in list_texts:
                    with torch.no_grad():
                        start = time.time()
                        outputs = self.models_loaded[model_label + '-cpu'](**inputs, start_positions=start_positions,
                                                                           end_positions=end_positions)
                        end = time.time()
                        # print("Measured delay of " +model_label)
                    del outputs
                    writer.writerow({'processor': processor, 'start_inference': start, 'end_inference': end, 'size': 0,
                                     'type_task': 1})
                    del inputs
                    # print("Inference delay: ", (end-start))

    def predict_image(self, list_images, model_label, processor):
        print("Start prediction of images, model: ", model_label, " processor: ", processor)
        if processor == 'gpu':
            for image in list_images:
                print("The inference in GPU will start")
                try:
                    self.models_loaded[model_label + '-gpu-0'].eval()
                    with torch.no_grad():
                        outputs = self.models_loaded[model_label + '-gpu-0'] \
                            (image.cuda())
                        idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()
                        prob = torch.softmax(outputs, dim=1)[0, idx].item()
                        print("Predicted: ", prob)
                        del outputs
                    # print("Inference delay: ", (end-start))
                except Exception as e:
                    print("Cuda exception was catched predict image")
                    print(e)
                    self.clean_cuda()

                # list_images.remove(image)



        else:
            for image in list_images:
                print("The inference in CPU will start")
                with torch.no_grad():
                    outputs = self.models_loaded[model_label + '-cpu-0'](image)
                    # print("Inference delay: ", (end-start))
                    idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()
                    prob = torch.softmax(outputs, dim=1)[0, idx].item()
                    print("Predicted: ", prob)
                    del outputs
                # list_images.remove(image)

        del list_images


        # Print predictions
        print('-----')
        for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))



    def predict_image_with_files(self, list_images, model_label, processor):
        print("Start prediction of images, model: ", model_label, " processor: ", processor)
        if processor == 'gpu':
            with open('delays_' + processor + '_' + str(self.batch_number) + '.csv', 'w') as csvfile:
                fieldnames = ['processor', 'start_inference', 'end_inference', 'size', 'start_loading', 'end_loading',
                              'type_task']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # print(self.loading_delays)
                writer.writerow(self.loading_delays)
                for image in list_images:
                    # print("The inference in GPU will start")
                    with torch.no_grad():
                        start = time.time()
                        outputs = self.models_loaded[model_label + '-gpu'].cuda()(image.cuda(non_blocking=True))
                        end = time.time()
                        # print("Measured delay of " +model_label)
                    del outputs
                    writer.writerow(
                        {'processor': processor, 'start_inference': start, 'end_inference': end, 'size': image.size(),
                         'type_task': 0})
                    del image
                    # print("Inference delay: ", (end-start))
                    self.clean_cuda()

        else:
            with open('delays_' + processor + '_' + str(self.batch_number) + '.csv', 'w') as csvfile:
                fieldnames = ['processor', 'start_inference', 'end_inference', 'size', 'start_loading', 'end_loading',
                              'type_task']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # print(self.loading_delays)
                writer.writerow(self.loading_delays)
                for image in list_images:
                    with torch.no_grad():
                        start = time.time()
                        outputs = self.models_loaded[model_label + '-cpu'](image)
                        end = time.time()
                        # print("Measured delay of " +model_label)
                    del outputs
                    writer.writerow(
                        {'processor': processor, 'start_inference': start, 'end_inference': end, 'size': image.size(),
                         'type_task': 0})
                    del image
                    # print("Inference delay: ", (end-start))
                    # idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()
                    # prob = torch.softmax(outputs, dim=1)[0, idx].item()

                    # print("Predicted: ", prob)


        #Print predictions
        print('-----')
        for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))


    def predict_audio(self, list_audios, model_label, processor):
        # TODO: Make it work for audio data
        print("Start prediction")
        if processor == 'gpu':
            with open('delays_' + processor + '_' + str(self.batch_number) + '.csv', 'w') as csvfile:
                fieldnames = ['processor', 'start_inference', 'end_inference', 'size', 'start_loading', 'end_loading',
                              'type_task']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # print(self.loading_delays)
                writer.writerow(self.loading_delays)
                for audio in list_audios:
                    # print("The inference in GPU will start")
                    with torch.no_grad():
                        start = time.time()
                        outputs = self.models_loaded[model_label + '-gpu'](audio.cuda())
                        end = time.time()
                        # print("Measured delay of " +model_label)
                    del outputs
                    writer.writerow(
                        {'processor': processor, 'start_inference': start, 'end_inference': end, 'size': audio.size(),
                         'type_task': 2})
                    del audio
                    # print("Inference delay: ", (end-start))
                    self.clean_conda()

        else:
            with open('delays_' + processor + '_' + str(self.batch_number) + '.csv', 'w') as csvfile:
                fieldnames = ['processor', 'start_inference', 'end_inference', 'size', 'start_loading', 'end_loading',
                              'type_task']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # print(self.loading_delays)
                writer.writerow(self.loading_delays)
                for audio in list_audios:
                    with torch.no_grad():
                        start = time.time()
                        outputs = self.models_loaded[model_label + '-cpu'](audio)
                        end = time.time()
                        # print("Measured delay of " +model_label)
                    del outputs
                    writer.writerow(
                        {'processor': processor, 'start_inference': start, 'end_inference': end, 'size': audio.size(),
                         'type_task': 2})
                    del audio
                    # print("Inference delay: ", (end-start))
                    # idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()
                    # prob = torch.softmax(outputs, dim=1)[0, idx].item()

                    # print("Predicted: ", prob)



    def predict(self, list_objects, model_label, type_task, processor):
        # print("The task is type ", type_task)
        if type_task == 0:
            # list_images = self.preprocess_image(data, processor)
            return self.predict_image(list_objects, model_label, processor)
        elif type_task == 1:  # Text
            # list_text = self.preprocess_text(data, processor)
            return self.predict_text(list_objects, model_label, processor)
        elif type_task == 2:  # Audio
            # list_audio = self.preprocess_audio(data, processor)
            return self.predict_audio(list_objects, model_label, processor)
        else:
            print("not valid type, I cannot predict")

    def execute_task(self, task_list, model_label, processor, batch_number):
        print("Batch number " + str(batch_number))
        print("Processor " + processor)

        self.batch_number = batch_number
        list_images = []
        for task in task_list: list_images.append(task.get_data())

        list_img = self.preprocess_image(list_images, processor)
        print("size of the list of images: ", len(list_img))


        if processor == 'gpu':
            start=time.time()
            self.models_loaded[model_label+'-'+processor].cuda()
            end = time.time()
            self.loading_delays = {'processor':'gpu','start_loading':start,'end_loading':end}
            self.predict(list_images,model_label, 0,processor)
            torch.cuda.empty_cache()
        else:
            self.predict(list_images,model_label, 0, processor)

        self.predict(list_images, model_label, 0, processor)
        self.clean_cuda()


    def execute_task_laptop(self, task_list, model_label, processor, batch_number, stats):
        print("Batch number " + str(batch_number))
        print("Processor " + processor)

        self.batch_number = batch_number
        batch_size = 0
        list_images = [];
        list_texts = [];
        list_audio = []
        start_img_time = 0;
        start_text_time = 0;
        start_audio_time = 0;
        for task in task_list:
            if task.get_type() == 0:
                start_img_time += task.get_timestamp()
                list_images.append(task.get_data())
            elif task.get_type() == 1:
                start_text_time += task.get_timestamp()
                list_texts.append(task.get_data())
            elif task.get_type() == 2:
                start_audio_time += task.get_timestamp()
                list_audio.append(task.get_data())

        if len(list_images):
            type_task = 0
            # list_images = self.preprocess_image(list_images, processor)
            self.predict(list_images, model_label, type_task, processor)
            start = start_img_time / len(list_images)  # Average starting time
            end = time.time()
            print("ADDED STATS: ", [len(list_images), type_task, (end - start) / len(list_images)])
            # TODO: Do this in the rest of the tasks
            stats.put([len(list_images), type_task, (end - start) / len(list_images)])

        if len(list_texts):
            type_task = 1
            list_texts = self.preprocess_text(list_texts, processor, model_label)
            self.predict(list_texts, model_label, type_task, processor)
            start = start_text_time / len(list_texts)  # Average starting time
            end = time.time()
            print("ADDED STATS: ", [len(list_texts), type_task, (end - start) / len(list_texts)])
            stats.put([len(list_texts), type_task, (end - start) / len(list_texts)])

        # TODO: Make it work for audio tasks
        if len(list_audio):
            type_task = 2
            list_audio = self.preprocess_text(list_audio, processor, model_label)
            self.predict(list_audio, model_label, 2, processor)
            start = start_audio_time / len(list_audio)
            end = time.time()
            print("ADDED STATS: ", [len(list_audio), type_task, (end - start) / len(list_audio)])
            stats.put([len(list_audio), type_task, (end - start) / len(list_audio)])


        if processor == 'cpu':
            print("cpu")
        else:
            labels_map = json.load(open('../data/images/labels_map.txt'))
            labels_map = [labels_map[str(i)] for i in range(1000)]

            with open('delays_'+processor+'_'+str(self.batch_number)+'.csv', 'w') as csvfile:
                fieldnames = ['processor','start_inference','end_inference','size','start_loading','end_loading']
                writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
                writer.writeheader()
                print(self.loading_delays)
                writer.writerow(self.loading_delays)
                for image in list_images:
                    image = Image.fromarray(image)
                    image = self.tfms(image).unsqueeze(0)
                    #print("The inference in GPU will start")
                    with torch.no_grad():
                        start= time.time()
                        outputs = self.models_loaded[model_label](image.cuda())
                        end = time.time()
                        print("Measured delay of " +model_label)
                    del outputs
                    writer.writerow({'processor':processor,'start_inference':start,'end_inference':end,'size':image.size()})
                    del image
                    print("Inference delay: ", (end-start))
                    torch.cuda.empty_cache()

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
        self.clean_cuda()

    def remove_models(self):
        for element in self.models_loaded:
            del element
        gc.collect()

    def get_loading_delays(self):
        return self.loading_delays

    def get_inference_delays(self):
        return self.inference_delays

    def get_processing_delays(self):
        return self.preprocessing_delays
