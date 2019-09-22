from argparse import ArgumentParser
import torch

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

import torch.nn as nn
import torch.nn.functional as F


# Load Predict Model
class_dict = {1:'Company',
              2:'EducationalInstitution',
              3:'Artist',
              4:'Athlete',
              5:'OfficeHolder',
              6:'MeanOfTransportation',
              7:'Building',
              8:'NaturalPlace',
              9:'Village',
              10:'Animal',
              11:'Plant',
              12:'Album',
              13:'Film',
              14:'WrittenWork'}

# Network has to be the same as the one trained
class Net(nn.Module):
    def __init__(self, EMBED_DIM, CLASS_DIM):
        super(Net, self).__init__()
        self.number_neurons = 500
        self.fc1 = nn.Linear(EMBED_DIM, self.number_neurons)

        self.fc2 = nn.Sequential(nn.BatchNorm1d(self.number_neurons),
                                 nn.Linear(self.number_neurons, self.number_neurons),
                                 nn.Dropout(0.3, inplace=True),

                                 nn.BatchNorm1d(self.number_neurons),
                                 nn.Linear(self.number_neurons, self.number_neurons),
                                 nn.Dropout(0.3, inplace=True)
                                 )

        self.fc3 = nn.Sequential(
            nn.BatchNorm1d(self.number_neurons),
            nn.Linear(self.number_neurons, CLASS_DIM)
        )

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return (x)


net = Net(512, 15)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

net_path = 'model_Name_Predict/trained_net.pt'
net = torch.load(net_path, map_location=torch.device('cpu'))
net.eval()


# Load Embedder
tf.logging.set_verbosity(tf.logging.ERROR)
module_url = 'module_Multi_Large/'

g = tf.Graph()
with g.as_default():
    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    embed = hub.Module(module_url)
    encoding_tensor = embed(similarity_input_placeholder)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

sess = tf.Session(graph=g)
sess.run(init_op)


def run(args):
    input_name = ' '.join(args.name)
    print('HELLO! The class of your word (%s) is:' % input_name)

    message_embedding = sess.run(encoding_tensor, feed_dict={similarity_input_placeholder: [input_name]})

    tensor_input = torch.Tensor(message_embedding).to(device)

    predicted_class = class_dict[int(torch.argmax(net(tensor_input)))]

    print(predicted_class)

    return predicted_class

if __name__ == '__main__':
    parser = ArgumentParser()
    subparser = parser.add_subparsers()

    run_parser = subparser.add_parser("run", help="Run the model in an interactive terminal.")
    run_parser.set_defaults(mode=run)

    run_parser.add_argument('--name', nargs='+', default=None, help="Name.")

    args = parser.parse_args()
    args.mode(args)
