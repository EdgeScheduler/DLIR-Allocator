import os
import numpy as np

def get_network(name, batch_size, dtype = "float32", sequence = 128, hidden_size = 768, num_hidden_layers = 12, num_attention_heads = 12, intermediate_size = 3072, max_position_embeddings = 512):
    '''
    name: value in ['bert', 'gpt2', 'roberta', 'nasnetalarge']
    '''
    input_shape = ()
    inputs = {}
    if name == 'bert':
        import torch
        import transformers  # pip3 install transformers==3.0
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        input_shape = [batch_size, sequence]

        # if os.path.exists("bert-mod.relay"):
        #     print("Load relay model from file...")
        #     with open("bert-mod.relay", "r") as fi:
        #         mod = tvm.ir.load_json(fi.read())
        #     with open("bert-params.relay", "rb") as fi:
        #         params = relay.load_param_dict(fi.read())
        # else:
        model_class = transformers.BertModel
        tokenizer_class = transformers.BertTokenizer

            # You can also download them manualy
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
            #   https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
            # Then rename to pytorch_model.bin, vocab.txt & config.json
            # weight = 'path to downloaded model dir'
            # weight = 'bert-base-uncased'
            # model = model_class.from_pretrained(weight,return_dict=False)
        configuration = transformers.BertConfig(return_dict=False, hidden_size = hidden_size, num_hidden_layers = num_hidden_layers, num_attention_heads = num_attention_heads, intermediate_size = intermediate_size, max_position_embeddings = max_position_embeddings)
        model = transformers.BertModel(configuration)
        model.eval()

            # tokenizer = tokenizer_class.from_pretrained(weight)
            # A = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])
            # There is 30522 words in bert-base-uncased's vocabulary list
            # input_dtype = 'int64'
        # input_name = 'input_ids'
        A = torch.randint(30000, input_shape)
        # scripted_model = torch.jit.trace(model, [A], strict=False)
        # shape_list = [(input_name, input_shape)]
        torch.onnx._export(model, A , name+"-"+str(sequence)+".onnx", export_params=True,input_names=["input"], output_names=["output"])
    elif name == 'gpt2':
        import torch
        from transformers import GPT2Model, GPT2Config
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        input_shape = [batch_size, sequence]
        
        configuration = GPT2Config(return_dict=False)
        model = GPT2Model(configuration)
        input_name = 'input_ids'
        A = torch.randint(50000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False).eval()
        # shape_list = [(input_name, input_shape)]
        # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        torch.onnx._export(model, A , name+"-"+str(sequence)+".onnx", export_params=True,input_names=["input"], output_names=["output"])
    elif name == 'roberta':
        import torch
        from transformers import RobertaConfig, RobertaModel
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        configuration = RobertaConfig(return_dict=False)
        model = RobertaModel(configuration).eval()
        input_shape = [batch_size, sequence]
        input_name = 'input_ids'
        A = torch.randint(30000, input_shape)
        scripted_model = torch.jit.trace(model, [A], strict=False)
        # shape_list = [(input_name, input_shape)]
        # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        torch.onnx._export(model, A , name+"-"+str(sequence)+".onnx", export_params=True,input_names=["input"], output_names=["output"])
    elif name == 'nasnetalarge':
        import torch
        import pretrainedmodels
        from torch.autograd import Variable
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').eval()
        input_shape = [batch_size, 3, 331, 331]
        A = torch.randn(batch_size, 3, 331, 331)
        input_name = 'input0'
        scripted_model = torch.jit.trace(model, [A], strict=False)
        # shape_list = [(input_name, input_shape)]
        # mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        torch.onnx._export(model, A , name+"-"+str(sequence)+".onnx", export_params=True,input_names=["input"], output_names=["output"])


# generate transformers-models by onnx
# get_network("bert",batch_size=1,sequence=128)
# get_network('gpt2',batch_size=1,sequence=128)
# get_network('roberta',batch_size=1,sequence=128)
# # get_network('nasnetalarge',batch_size=1,sequence=128)