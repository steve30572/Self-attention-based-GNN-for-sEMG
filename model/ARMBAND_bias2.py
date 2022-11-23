import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("error")
initializer = nn.init.xavier_uniform_
################
# input data shape : (batch_size, 12, 8, 7)
################
adj = torch.ones((8,8))
class GNN(nn.Module):
    def __init__(self, input_feat, output_feat, indicator):
        super(GNN, self).__init__()
        self.W_gnn = nn.Parameter(initializer(torch.randn(input_feat*7, output_feat*7)))    # gnn 이랑 attention인 경우에 사용
        self.W_gnn2 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7)))  # gnn 이랑 attention인 경우에 사용
        self.B_gnn = nn.Parameter((torch.randn(output_feat*7)))
        self.W_cat = nn.Parameter(initializer(torch.randn(input_feat *7*2, output_feat*7)))  # concat만 할 경우 사용
        self.B_cat = nn.Parameter((torch.randn(output_feat * 7)))
        self.W_att = nn.Parameter(initializer(torch.randn(output_feat* 2*7, 1)))   #W_att 에 쓰이는 파라미터
        self.W_att2 = nn.Parameter(initializer(torch.randn(output_feat * 2 * 7, 1)))
        self.W_att3 = nn.Parameter(initializer(torch.randn(output_feat * 2 * 7, 1)))
        self.MHA = torch.nn.MultiheadAttention(embed_dim=output_feat * 7, num_heads=4, batch_first=True)


        ### for GAT --0705
        self.W_head1 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat*7 // 4)))
        self.W_head2 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7 // 4)))
        self.W_head3 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7 // 4)))
        self.W_head4 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7 // 4)))
        self.W_alpha1 = nn.Parameter(initializer(torch.randn(output_feat*7//2, 1)))
        self.W_alpha2 = nn.Parameter(initializer(torch.randn(output_feat * 7 // 2, 1)))
        self.W_alpha3 = nn.Parameter(initializer(torch.randn(output_feat * 7 // 2, 1)))
        self.W_alpha4 = nn.Parameter(initializer(torch.randn(output_feat * 7 // 2, 1)))


        self.indicator = indicator
        self.output_feat = output_feat


    def forward(self, x):
        B , T, C, F = x.shape   # B: batch size, T: time, C : channel(8), F : features
        x = torch.transpose(x, 1, 2)
        x = x.reshape(B, C, -1)
        a, b = self.MHA(x, x, x)

        if self.indicator == 0:   #GCN
            adj2 = torch.ones((B, 8, 8))/8
            print(adj2.shape, self.B_gnn.shape)
            x = torch.bmm(adj2+self.B_gnn, x)
            x = torch.matmul(x, self.W_gnn)
            # return x
            # x += self.B_gnn

        elif self.indicator == 1: #concat
            ##temporary GAT
            W_head_list = [self.W_head1, self.W_head2, self.W_head3, self.W_head4]
            W_alpha_list = [self.W_alpha1, self.W_alpha2, self.W_alpha3, self.W_alpha4]
            result = None
            for head in range(4):
                adj = torch.ones((B, C, C))
                temp_feature = torch.matmul(x, W_head_list[head])
                for i in range(C):
                    for j in range(C):
                        adj[:, i, j] = torch.matmul(torch.cat((temp_feature[:, i], temp_feature[:, j]), dim=1), W_alpha_list[head]).reshape(-1)
                adj = torch.nn.functional.softmax(adj, dim=2)
                if result == None:
                    result = torch.bmm(adj, temp_feature)
                else:
                    temp_result = torch.bmm(adj, temp_feature)
                    result = torch.cat((result, temp_result), dim=2)
            x = result
            x += self.B_gnn




        else:   #attention
            x = torch.bmm(b, x)
            x = torch.matmul(x, self.W_gnn)
            x += self.B_gnn

        x = x.reshape(B, C, T, -1)
        x = torch.transpose(x, 1, 2)
        return x#torch.nn.functional.relu(x)
class Temporal_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Temporal_layer, self).__init__()
        self.WT_input = torch.nn.Parameter(initializer(torch.randn(7, 7, 1, in_dim-out_dim+1)))  #out_dim, in_dim, 1, 1)))
        self.WT_glu = torch.nn.Parameter(initializer(torch.randn(14, 7, 1, in_dim-out_dim+1)))  #out_dim*2, in_dim, 1, 1)))
        self.B_input = nn.Parameter((torch.FloatTensor(7)))
        self.B_glu = nn.Parameter((torch.FloatTensor(14)))
        self.out_dim = out_dim
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x_input = F.conv2d(x, self.WT_input)#, bias = self.B_input)
        x_glu = F.conv2d(x, self.WT_glu)#, bias=self.B_glu)
        # print(x_input.shape, x_glu.shape, "input's shape")
        return (x_glu[:,0:7,:,:]+x_input)*self.sigmoid(x_glu[:,-7:,:,:])

class Spatial_layer(nn.Module):
    def __init__(self, in_dim, out_dim, indicator):
        super(Spatial_layer, self).__init__()
        self.WS_input = torch.nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.out_dim = out_dim
        self.batch1 = nn.BatchNorm2d(in_dim)
        self.gnn = GNN(in_dim, out_dim, indicator)
        self.gnn2 = GNN(in_dim, out_dim, indicator)
        self.batch2 = nn.BatchNorm2d(out_dim)
    def forward(self, x):
        x2 = self.gnn(x)
        self.batch1(x2)
        x2 = F.relu(x2)
        x2 = self.gnn2(x2)
        x2 = self.batch2(x2)
        x2 = F.relu(x2)
        return x+x2
class ARMBANDGNN(nn.Module):
    def __init__(self, channels, indicator, num_classes):
        super(ARMBANDGNN, self).__init__()
        first, second, third, fourth = channels
        self.Temp1 = Temporal_layer(first, second)
        self.batch1 = nn.BatchNorm2d(7)#(second)
        self.Spat1 = Spatial_layer(16, 16, indicator)#(second, second, indicator)
        self.Spat2 = Spatial_layer(8, 8, indicator)
        self.Temp2 = Temporal_layer(second, third)
        self.batch2 = nn.BatchNorm2d(7)#(third)
        self.Temp3 = Temporal_layer(third, fourth)
        self.MLP1 = nn.Linear(fourth*56, 500)
        self.MLP2 = nn.Linear(500, 2000)
        self.MLP3 = nn.Linear(2000, num_classes)
        self.drop1 = nn.Dropout2d(p=0.4)
        self.drop2 = nn.Dropout2d(p=0.4)
        self.drop3 = nn.Dropout2d(p=0.4)
        self.MLP = nn.Linear(16*8*7, 16*8*7)
        self.CNN = nn.Conv2d(16, 16, 3, padding=1)
    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        x = self.Temp1(x)
        x = self.batch1(x)
        x = torch.transpose(x, 1, 3)
        x = self.Spat1(x)
        x = torch.transpose(x, 1, 3)
        x = self.Temp2(x)
        x = self.batch2(x)
        x = self.Temp3(x)
        bs, _, _, _ = x.shape
        x = x.reshape(bs, -1)
        x = self.MLP1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.MLP2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.MLP3(x)
        return F.log_softmax(x, dim=1)