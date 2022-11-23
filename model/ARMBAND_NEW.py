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
# adj = torch.zeros(8,8)
# for i in range(8):
#     devider= i+8
#     adj[i][devider % 8] = 0.25
#     adj[i][(devider + 1) % 8] = 0.25
#     adj[i][(devider - 1) % 8] = 0.25
#     adj[i][(devider + 4) % 8] = 0.25
class GNN(nn.Module):
    def __init__(self, input_feat, output_feat, indicator):
        super(GNN, self).__init__()
        self.W_gnn = nn.Parameter(initializer(torch.randn(input_feat*7, output_feat*7)))    # gnn 이랑 attention인 경우에 사용
        self.B_gnn = nn.Parameter((torch.randn(output_feat*7)))
        # print(torch.isnan(self.B_gnn).any())
        self.W_gnn2 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7)))  # gnn 이랑 attention인 경우에 사용
        self.B_gnn2 = nn.Parameter((torch.randn(output_feat * 7)))
        self.W_gnn3 = nn.Parameter(initializer(torch.randn(input_feat * 7, output_feat * 7)))  # gnn 이랑 attention인 경우에 사용
        self.B_gnn3 = nn.Parameter((torch.randn(output_feat * 7)))
        self.W_cat = nn.Parameter(initializer(torch.randn(input_feat *7*2, output_feat*7)))  # concat만 할 경우 사용
        self.B_cat = nn.Parameter((torch.randn(output_feat * 7)))
        self.W_att = nn.Parameter(initializer(torch.randn(output_feat* 2*7, 1)))   #W_att 에 쓰이는 파라미터
        self.W_att2 = nn.Parameter(initializer(torch.randn(output_feat * 2 * 7, 1)))
        self.W_att3 = nn.Parameter(initializer(torch.randn(output_feat * 2 * 7, 1)))
        self.MHA = torch.nn.MultiheadAttention(embed_dim=output_feat * 7, num_heads=4, batch_first=True)


        self.concat_att = nn.Linear(output_feat * 7 * 3, output_feat * 7)
        self.indicator = indicator
        self.output_feat = output_feat

    # def get_alpha(self, x):
    #     e = torch.matmul(x, self.W_gnn)
    #     B, _, _ = e.shape
    #     a = torch.zeros((B, 8, 8, 2*self.output_feat*7))
    #     soft = torch.nn.Softmax(dim=1)
    #
    #     left_emb = e.repeat_interleave(8, dim=2)
    #     right_emb = e.repeat(1, 1, 8)
    #     a = torch.cat((left_emb, right_emb), dim=2)
    #     a = a.view(-1, 8, 8, 2 * self.output_feat*7)
    #     a1 = torch.matmul(a, self.W_att)
    #     a1 = soft(a1)
    #
    #
    #     e = torch.matmul(x, self.W_gnn2)
    #     B, _, _ = e.shape
    #     a = torch.zeros((B, 8, 8, 2 * self.output_feat * 7))
    #     soft = torch.nn.Softmax(dim=1)
    #
    #     left_emb = e.repeat_interleave(8, dim=2)
    #     right_emb = e.repeat(1, 1, 8)
    #     a = torch.cat((left_emb, right_emb), dim=2)
    #     a = a.view(-1, 8, 8, 2 * self.output_feat * 7)
    #
    #     a2 = torch.matmul(a, self.W_att2)
    #     a2 = soft(a2)
    #
    #     e = torch.matmul(x, self.W_gnn3)
    #     B, _, _ = e.shape
    #     a = torch.zeros((B, 8, 8, 2 * self.output_feat * 7))
    #     soft = torch.nn.Softmax(dim=1)
    #
    #     left_emb = e.repeat_interleave(8, dim=2)
    #     right_emb = e.repeat(1, 1, 8)
    #     a = torch.cat((left_emb, right_emb), dim=2)
    #     a = a.view(-1, 8, 8, 2 * self.output_feat * 7)
    #
    #     a3 = torch.matmul(a, self.W_att3)
    #     a3 = soft(a3)
    #     return a1, a2, a3

    def forward(self, x):
        B , T, C, F = x.shape   # B: batch size, T: time, C : channel(8), F : features
        x = torch.transpose(x, 1, 2)
        x = x.reshape(B, C, -1)
        # alpha1, alpha2, alpha3 = self.get_alpha(x)           #GAT에서의 알파
        a, b = self.MHA(x, x, x)



        if self.indicator == 0:   #GCN
            adj2 = torch.ones((B, 8, 8))/8
            x = torch.bmm(adj2, x)
            x = torch.matmul(x, self.W_gnn)
            x += self.B_gnn

        elif self.indicator == 1: #concat
            #get the neighbor and concat
            neighbor_adj = adj - torch.tensor(np.identity(8), dtype=torch.float)   # A-I
            neighbor_x = torch.matmul(neighbor_adj, x)          # (A-I) * x (8, feat)
            x = torch.cat((x, neighbor_x), dim=2)               # (8, feat) (8, feat) --> (8, feat*2)
            x = torch.matmul(x, self.W_cat)                     # (8, output_feat)
            x += self.B_cat
        else:   #attention
            x = torch.bmm(b, x)


        x = x.reshape(C, T, B, -1)
        x = torch.transpose(x, 0, 2)
        return x#torch.nn.functional.relu(x)
class Temporal_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Temporal_layer, self).__init__()
        self.WT_input = torch.nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.WT_glu = torch.nn.Parameter(initializer(torch.randn(out_dim*2, in_dim, 1, 1)))
        self.B_input = nn.Parameter((torch.FloatTensor(out_dim)))
        self.B_glu = nn.Parameter((torch.FloatTensor(out_dim*2)))
        self.out_dim = out_dim
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x_input = F.conv2d(x, self.WT_input)#, bias = self.B_input)
        print(x_input.shape)
        x_glu = F.conv2d(x, self.WT_glu)#, bias=self.B_glu)
        print(x_glu.shape)
        print("8"*100)
        return (x_glu[:,0:self.out_dim,:,:]+x_input)*self.sigmoid(x_glu[:,-self.out_dim:,:,:])

class Spatial_layer(nn.Module):
    def __init__(self, in_dim, out_dim, indicator):
        super(Spatial_layer, self).__init__()
        self.WS_input = torch.nn.Parameter(initializer(torch.randn(out_dim, in_dim, 1, 1)))
        self.B_input = torch.nn.Parameter((torch.FloatTensor(out_dim)))
        self.out_dim = out_dim
        self.gnn = GNN(in_dim, out_dim, indicator)
        self.WQ1 = torch.nn.Parameter(initializer(torch.randn(112, 256)))
        self.WK1 = torch.nn.Parameter(initializer(torch.randn(112, 256)))
        self.WQ2 = torch.nn.Parameter(initializer(torch.randn(112, 256)))
        self.WK2 = torch.nn.Parameter(initializer(torch.randn(112, 256)))
        self.WQ3 = torch.nn.Parameter(initializer(torch.randn(112, 256)))
        self.WK3 = torch.nn.Parameter(initializer(torch.randn(112, 256)))
    def forward(self, x):
        # self.self_attention_adj(x)
        x_input = F.conv2d(x, self.WS_input)#, bias=self.B_input)
        x = self.gnn(x)
        return F.relu(x_input + x)
    def self_attention_adj(self, x):
        input_emb = torch.transpose(x, 1, 2)
        b, channel, f, a = input_emb.shape
        input_emb = input_emb.reshape(b, channel, -1)
        WQ_list = [self.WQ1, self.WQ2, self.WQ3]
        WK_list = [self.WK1, self.WK2, self.WK3]
        A = torch.zeros(b, 8, 8)
        for three in range(3):
            IDX = torch.zeros(b, 8, 8)
            Q = torch.matmul(input_emb, WQ_list[three])
            K = torch.matmul(input_emb, WK_list[three])
            K = torch.transpose(K, 1, 2)
            R = torch.bmm(Q, K)
            R = R / (f*a)
            temp_R = R.reshape(b, -1)
            topk, topk_index = torch.topk(temp_R, 51, dim=1)
            for i in range(b):
                IDX[i, (torch.div(topk_index[i] , 8).to(torch.long)), (torch.remainder(topk_index[i] , 8)).to(torch.long)] = 1 #R[i, topk_index[i] // 8, topk_index[i] % 8]
            # for k in range(b):
            #
            #     v, i = torch.topk(R[k].flatten(), 51)
            #     index_list = np.array(np.unravel_index(i.numpy(), R.shape)).T
            #     for temp_A in range(51):
            #         IDX[k, index_list[temp_A][0], index_list[temp_A][1]] = R[k, index_list[temp_A][0], index_list[temp_A][1]]
            R = R * IDX + -1*1e4*(torch.ones((b, 8, 8)) - IDX)
            R = R.reshape(b, -1)
            AA = torch.nn.functional.softmax(R, dim=1)  # dim 문제
            AA = AA.reshape(b, 8, 8)
            # print(three, AA)
            A = A + AA
        global adj
        adj = A/3
class ARMBANDGNN(nn.Module):
    def __init__(self, channels, indicator, num_classes):
        super(ARMBANDGNN, self).__init__()
        first, second, third, fourth = channels
        self.Temp1 = Temporal_layer(first, second)
        self.batch1 = nn.BatchNorm2d(second)
        self.Spat1 = Spatial_layer(second, second, indicator)
        self.Temp2 = Temporal_layer(second, third)
        self.batch2 = nn.BatchNorm2d(third)
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
        x = self.Temp1(x)
        print("first_timep")
        print(x.shape)
        x = self.batch1(x)
        #print(x.shape)

        x = self.Spat1(x)
        # exit(0)
        # x = self.CNN(x)
        ########
        # x = x.reshape(-1, 16*8*7)
        # x = self.MLP(x)
        # x = x.reshape(-1, 16, 8, 7)

        #print(x.shape)
        x = self.Temp2(x)
        print("second_timep")
        print(x.shape)
        x = self.batch2(x)
        x = self.Temp3(x)
        print("third_timep")
        print(x.shape)
        bs, _, _, _ = x.shape
        x = x.view(bs, -1)
        x = self.MLP1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.MLP2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.MLP3(x)
        # y = F.log_softmax(x, dim=1)

        # print("y"*100)
        # print(y)
        return F.log_softmax(x, dim=1)
















if __name__=='__main__':
    a=torch.randn(12,24,8,7)
    model=ARMBANDGNN([24, 16, 8, 4], 2, 7)
    first=list(model.parameters())[5].clone()
    last=torch.nn.Linear(8,4)
    last.requires_grad=True
    for i in range(10):
        model.train()
        b=model(a)

        answer=torch.randn(12,7)

        #answer=answer.to(torch.float32)
        #answer.requires_grad=True
        #answer=answer.reshape(-1,1,1,1)
        #print(answer.shape)
        #b=b.reshape(1,12)
        #answer=answer.reshape(1,12)
        #print(b.shape,answer.shape)
        optim=torch.optim.Adam(model.parameters(),lr=2)
        loss=torch.nn.MSELoss()
        losses=loss(b,answer)

        losses.backward()
        optim.step()

    count=0
    for name,param in model.named_parameters():
        count+=1
        print(name)
        if param.grad is not None:
            print("not None bro",count)
    #optim.step()
    b=list(model.parameters())[5].clone()
    print(torch.equal(first.data,b.data))
    print(adj)























