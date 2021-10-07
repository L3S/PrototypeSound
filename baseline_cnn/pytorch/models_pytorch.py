import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchlibrosa as tl
import numpy as np
import matplotlib.pyplot as plt
import config

def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    elif 'bool' in str(x.dtype):
        x = torch.BoolTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='max', activation='relu'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if activation == 'relu':
            x = F.relu_(self.bn2(self.conv2(x)))
        elif activation == 'sigmoid':
            x = torch.sigmoid(self.bn2(self.conv2(x)))

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num):
        super(DecisionLevelMaxPooling, self).__init__()
        sample_rate=config.sample_rate
        window_size = config.win_length
        hop_size = config.hop_length
        mel_bins = config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = tl.Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        self.logmel_extractor = tl.LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=20, fmax=2000, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        self.cnn_encoder = CNN_encoder()

        self.fc_final = nn.Linear(512, classes_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, date_length)
        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        batch_size, channel_num, _, mel_bins = x.shape
        x_diff1 = torch.diff(x, n=1, dim=2, append=x[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x_diff2 = torch.diff(x_diff1, n=1, dim=2, append=x_diff1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x = torch.cat((x, x_diff1, x_diff2), dim=1)

        if False:
            x_array = x.data.cpu().numpy()[0]
            x_array = np.squeeze(x_array)
            plt.matshow(x_array.T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig('test.png')

        x = self.cnn_encoder(x)

        # (samples_num, 512, hidden_units)
        output = F.max_pool2d(x, kernel_size=x.shape[2:])
        output = output.view(output.shape[0:2])

        output = F.log_softmax(self.fc_final(output), dim=-1)

        return output




class ProtoNet(nn.Module):
    def __init__(self, classes_num, n_proto, proto_form):  # proto_type: 'vector1d', 'vector2d', 'vector2d_att'
        super(ProtoNet, self).__init__()
        sample_rate=config.sample_rate
        window_size = config.win_length
        hop_size = config.hop_length
        mel_bins = config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.classes_num = classes_num
        self.spectrogram_extractor = tl.Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        self.logmel_extractor = tl.LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=20, fmax=2000, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        self.cnn_encoder = CNN_encoder()
        self.prototypelayer = PrototypeLayer(n_proto=n_proto, distance='cosine', proto_form=proto_form)

        self.fc_final = nn.Linear(n_proto, classes_num, bias=False)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, date_length)
        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        batch_size, channel_num, _, mel_bins = x.shape
        x_diff1 = torch.diff(x, n=1, dim=2, append=x[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x_diff2 = torch.diff(x_diff1, n=1, dim=2, append=x_diff1[:, :, -1, :].view((batch_size, channel_num, 1, mel_bins)))
        x = torch.cat((x, x_diff1, x_diff2), dim=1)

        if False:
            x_array = x.data.cpu().numpy()[0]
            x_array = np.squeeze(x_array)
            plt.matshow(x_array.T, origin='lower', aspect='auto', cmap='jet')
            plt.savefig('test.png')

        x_emb = self.cnn_encoder(x)
        x_emb = F.dropout(x_emb, p=0.5, training=self.training)
        similarity, prototype = self.prototypelayer(x_emb)  # for cosine
        distance = torch.exp(-similarity)

        # (samples_num, 512, hidden_units)
        output = F.log_softmax(self.fc_final(similarity), dim=-1)

        if self.training:
            loss_diverse = self.diverse_loss(prototype)
            return output, x, distance, similarity, loss_diverse
        else:
            return output

    def diverse_loss(self, a):
        # Compute the "diversity" loss to penalize prototypes close to each other
        if len(a.shape) == 2:
            pt_flat = a
        elif len(a.shape) > 2:
            pt_flat = torch.flatten(a, start_dim=1, end_dim=-1)
        '''
        loss_sim = 0
        for i in range(0, a.shape[0]):
            for j in range(0, a.shape[0]):
                if i != j:
                    loss_sim = loss_sim + F.cosine_similarity(torch.unsqueeze(pt_flat[i], dim=0), torch.unsqueeze(pt_flat[j], dim=0))
        loss_sim = loss_sim/(pt_flat.shape[0]*pt_flat.shape[0]-pt_flat.shape[0])  # 12
        '''
        '''
        proto_per_class = int(a.shape[0] / self.classes_num)
        loss_sim_internal = 0
        n_sim_internal = 0
        loss_sim_external = 0
        n_sim_external = 0
        for i in range(0, a.shape[0]):
            for j in range(0, a.shape[0]):
                i_class = int(i/proto_per_class)
                if i != j:
                    if j >= i_class * proto_per_class and j < (i_class+1) * proto_per_class:
                        loss_sim_internal = loss_sim_internal + F.cosine_similarity(torch.unsqueeze(pt_flat[i], dim=0), torch.unsqueeze(pt_flat[j], dim=0))
                        n_sim_internal = n_sim_internal + 1
                    else:
                        loss_sim_external = loss_sim_external + F.cosine_similarity(torch.unsqueeze(pt_flat[i], dim=0), torch.unsqueeze(pt_flat[j], dim=0))
                        n_sim_external = n_sim_external + 1
        loss_sim_internal = loss_sim_internal/n_sim_internal
        loss_sim_external = loss_sim_external/n_sim_external
        if loss_sim_internal == 0:
            loss_sim = loss_sim_external
        else:
            loss_sim = loss_sim_external/loss_sim_internal
        '''

        proto_per_class = int(pt_flat.shape[0] / self.classes_num)
        loss_sim_internal = 0
        a1 = torch.unsqueeze(pt_flat, dim=1)
        a2 = torch.unsqueeze(pt_flat, dim=0)
        sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((a1.shape[0], a1.shape[0]), dtype=torch.float), True)
        similarity = torch.sum(a1 * a2, dim=2) / torch.maximum(torch.norm(a1, p=2, dim=2)*torch.norm(a2, p=2, dim=2), sim_lowbound)
        for i in range(0, self.classes_num):
            loss_sim_internal = loss_sim_internal + torch.sum(similarity[i * proto_per_class:(i+1) * proto_per_class, i * proto_per_class:(i+1) * proto_per_class], dim=(0, 1))
        loss_sim_external = torch.sum(similarity, dim=(0, 1)) - loss_sim_internal
        diag_sum = torch.sum(similarity * move_data_to_gpu(torch.eye(pt_flat.shape[0]), True), dim=(0, 1))
        loss_sim_internal = loss_sim_internal - diag_sum
        # mean
        loss_sim_internal = loss_sim_internal / (self.classes_num*(proto_per_class*proto_per_class-proto_per_class))
        loss_sim_external = loss_sim_external / (pt_flat.shape[0]*pt_flat.shape[0] - self.classes_num*proto_per_class*proto_per_class)

        if proto_per_class == 1:
            loss_sim = loss_sim_external
        else:
            loss_sim = loss_sim_external/loss_sim_internal

        return loss_sim


class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()

        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input):
        # (batch_size, 3, time_steps, mel_bins)
        x = input.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # (samples_num, channel, time_steps, freq_bins)
        x = self.conv1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)

        return x

class PrototypeLayer(nn.Module):
    def __init__(self, n_proto, distance='euclidean', proto_form=None):
        super(PrototypeLayer, self).__init__()
        self.n_proto = n_proto
        self.distance = distance
        self.proto_form = proto_form

        # (n_proto, channel, time_steps, freq_bins)
        if self.proto_form == 'vector1d':
            self.prototype = nn.Parameter(torch.empty((self.n_proto, 512), dtype=torch.float), requires_grad=True)
        elif 'att' in self.proto_form:
            self.prototype = nn.Parameter(torch.empty((self.n_proto, 512, 56), dtype=torch.float), requires_grad=True)
        else:
            self.prototype = nn.Parameter(torch.empty((self.n_proto, 512, 7, 8), dtype=torch.float), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype.shape), requires_grad=False)

        if 'att' in self.proto_form:
            self.fc = nn.Linear(512, 1)

        self.bn = nn.BatchNorm2d(self.n_proto)
        self.bn1d = nn.BatchNorm1d(self.n_proto)
        self.ln = nn.LayerNorm(self.n_proto)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.prototype)
        init_bn(self.bn)
        init_bn(self.bn1d)
        if 'att' in self.proto_form:
            init_layer(self.fc)

    def forward(self, input):
        # (samples_num, channel, time_steps, freq_bins)
        if self.distance == 'euclidean':
            x2 = input ** 2
            x2_sum = F.conv2d(x2, weight=self.ones)

            p2 = self.prototype ** 2
            p2 = torch.sum(p2, dim=(1, 2, 3)).view(1,-1, 1, 1)

            xp = F.conv2d(input, weight=self.prototype)
            distance = F.relu_(x2_sum + p2-2 * xp).view(xp.shape[0:2])
            #distance = torch.norm(self.prototype-x, p=2, dim=(2, 3, 4))
            #distance = torch.sum(torch.pow((x - self.prototype), 2), dim=(2, 3, 4))   # (samples_num, n_proto)
            #distance = torch.unsqueeze(distance, dim=2)
            #distance = torch.unsqueeze(distance, dim=3)
            #distance = F.relu_(distance)
            #distance = distance.view(distance.shape[0:2])
        elif self.distance == 'cosine':
            if self.proto_form == 'vector1d':
                x = F.max_pool2d(input, kernel_size=input.shape[2:])
                similarity = np.zeros((x.shape[0], self.n_proto), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones(1, dtype=torch.float), True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        a = torch.flatten(x[i], start_dim=0, end_dim=-1)
                        b = torch.flatten(self.prototype[j], start_dim=0, end_dim=-1)
                        #similarity[i][j] = F.cosine_similarity(a, b, dim=0)
                        similarity[i][j] = torch.sum(a * b, dim=0) / torch.maximum(torch.norm(a, p=2, dim=0)*torch.norm(b, p=2, dim=0), sim_lowbound)
                similarity = self.ln(similarity)

            elif self.proto_form == 'vector2d':
                x = input
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[3]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[3]), dtype=torch.float), True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        #similarity[i][j] = F.cosine_similarity(x[i], self.prototype[j], dim=0)
                        similarity[i][j] = torch.sum(x[i] * self.prototype[j], dim=0) / torch.maximum(torch.norm(x[i], p=2, dim=0) * torch.norm(self.prototype[j], p=2, dim=0), sim_lowbound)
                similarity = self.bn(similarity)
                similarity = F.avg_pool2d(similarity, kernel_size=similarity.shape[2:])
                similarity = similarity.view(similarity.shape[0:2])
                #similarity = self.ln(similarity)

            elif self.proto_form == 'vector2d_att':
                x = torch.flatten(input, start_dim=2, end_dim=-1) # 16, 512, 56
                similarity_att = np.zeros((x.shape[0], self.n_proto, x.shape[2]), dtype=np.float)
                similarity_att = move_data_to_gpu(similarity_att, True)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2]), dtype=torch.float), True)
                for i in range(0, similarity_att.shape[0]):
                    for j in range(0, similarity_att.shape[1]):
                        #similarity_att[i][j] = F.cosine_similarity(x[i], self.prototype[j], dim=0)
                        similarity_att[i][j] = torch.sum(x[i] * self.prototype[j], dim=0) / torch.maximum(torch.norm(x[i], p=2, dim=0) * torch.norm(self.prototype[j], p=2, dim=0), sim_lowbound)

                similarity_att = F.softmax(similarity_att, dim=-1)  # (16, 4, 56)
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[1]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        similarity[i][j] = torch.sum(torch.unsqueeze(similarity_att[i][j], dim=0) * x[i] * self.prototype[j], dim=1)
                #similarity_att = self.bn1d(similarity_att)
                #similarity = F.avg_pool1d(similarity, kernel_size=similarity.shape[2])
                similarity = F.relu_(self.fc(similarity))
                similarity = similarity.view(similarity.shape[0:2])

            elif self.proto_form == 'vector2d_avgp':
                x = input
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=np.float)
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(torch.unsqueeze(x, dim=4), dim=5)
                b = torch.unsqueeze(torch.unsqueeze(self.prototype, dim=2), dim=3)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_all = torch.mean(similarity_all, 5)
                similarity = torch.mean(similarity_all, 4)
                similarity = self.bn(similarity)
                similarity = F.avg_pool2d(similarity, kernel_size=similarity.shape[2:])
                similarity = similarity.view(similarity.shape[0:2])

            elif self.proto_form == 'vector2d_avgp_att':
                x = torch.flatten(input, start_dim=2, end_dim=-1) # 16, 512, 56
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[2]), dtype=np.float) # 16, 4, 56, 56
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(x, dim=3)  # 16, 512, 56, 1
                b = torch.unsqueeze(self.prototype, dim=2) # 4, 512, 1, 56
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[2]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_att = torch.mean(similarity_all, 3) # 16, 4, 56

                similarity_att = F.softmax(similarity_att, dim=-1)  # (16, 4, 56)
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[1]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        similarity[i][j] = torch.sum(torch.unsqueeze(similarity_att[i][j], dim=0) * x[i] * self.prototype[j], dim=-1) #?
                similarity = F.relu_(self.fc(similarity))
                similarity = similarity.view(similarity.shape[0:2])

            elif self.proto_form == 'vector2d_maxp':
                x = input
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=np.float)
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(torch.unsqueeze(x, dim=4), dim=5)
                b = torch.unsqueeze(torch.unsqueeze(self.prototype, dim=2), dim=3)
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[3], x.shape[2], x.shape[3]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_all, _ = torch.max(similarity_all, 5)
                similarity, _ = torch.max(similarity_all, 4)
                '''similarity = self.bn(similarity)'''
                similarity = F.avg_pool2d(similarity, kernel_size=similarity.shape[2:])
                similarity = similarity.view(similarity.shape[0:2])
                #similarity = self.ln(similarity)

            elif self.proto_form == 'vector2d_maxp_att':
                x = torch.flatten(input, start_dim=2, end_dim=-1) # 16, 512, 56
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[2]), dtype=np.float) # 16, 4, 56, 56
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(x, dim=3)  # 16, 512, 56, 1
                b = torch.unsqueeze(self.prototype, dim=2) # 4, 512, 1, 56
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[2]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_att, ind = torch.max(similarity_all, 3) # 16, 4, 56

                #prototype_ind = np.zeros((x.shape[0], self.n_proto, self.prototype.shape[1], self.prototype.shape[2]), dtype=np.float)  # (16, 4, 512, 56)
                #prototype_ind = move_data_to_gpu(prototype_ind, True)
                #for i in range(0, ind.shape[0]):
                #    for j in range(0, ind.shape[1]):
                #            prototype_ind[i, j, :, :] = self.prototype[j, :, ind[i, j, :]]
                similarity_att = F.softmax(similarity_att, dim=-1)  # (16, 4, 56)
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[1]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        similarity[i][j] = torch.sum(torch.unsqueeze(similarity_att[i][j], dim=0) * x[i] * self.prototype[j, :, ind[i, j, :]], dim=-1)
                similarity = F.relu_(self.fc(similarity))
                similarity = similarity.view(similarity.shape[0:2])

            elif self.proto_form == 'vector2d_maxpx_att':
                x = torch.flatten(input, start_dim=2, end_dim=-1) # 16, 512, 56
                similarity_all = np.zeros((x.shape[0], self.n_proto, x.shape[2], x.shape[2]), dtype=np.float) # 16, 4, 56, 56
                similarity_all = move_data_to_gpu(similarity_all, True)
                a = torch.unsqueeze(x, dim=3)  # 16, 512, 56, 1
                b = torch.unsqueeze(self.prototype, dim=2) # 4, 512, 1, 56
                sim_lowbound = move_data_to_gpu((1E-8) * torch.ones((x.shape[2], x.shape[2]), dtype=torch.float), True)
                for i in range(0, similarity_all.shape[0]):
                    for j in range(0, similarity_all.shape[1]):
                        similarity_all[i][j] = torch.sum(a[i] * b[j], dim=0) / torch.maximum(torch.norm(a[i], p=2, dim=0)*torch.norm(b[j], p=2, dim=0), sim_lowbound)
                similarity_att, _ = torch.max(similarity_all, 3) # 16, 4, 56
                similarity_att_p, _ = torch.max(similarity_all, 2) # 16, 4, 56

                similarity_att = F.softmax(similarity_att, dim=-1)  # (16, 4, 56)
                similarity_att_p = F.softmax(similarity_att_p, dim=-1)  # (16, 4, 56)
                similarity = np.zeros((x.shape[0], self.n_proto, x.shape[1]), dtype=np.float)
                similarity = move_data_to_gpu(similarity, True)
                for i in range(0, similarity.shape[0]):
                    for j in range(0, similarity.shape[1]):
                        similarity[i][j] = torch.sum(torch.unsqueeze(similarity_att[i][j], dim=0) * x[i] * torch.unsqueeze(similarity_att_p[i][j], dim=0) * self.prototype[j], dim=-1)
                similarity = F.relu_(self.fc(similarity))
                similarity = similarity.view(similarity.shape[0:2])

        else:
            print('Wrong distance type!')

        if self.distance == 'euclidean':
            return distance, self.prototype
        elif self.distance == 'cosine':
            return similarity, self.prototype
        else:
            print('Wrong distance type!')


