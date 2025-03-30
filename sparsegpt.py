import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0


    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        # Online Update via EMA
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # Root-Decaying Sample-Dependent Scaling
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01, sparsity_way="origin"
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        #del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        
        
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        def get_block_sensitivity(W_block, Hinv_block):
            """计算子块敏感度得分"""
            return torch.sum(W_block**2 / torch.diag(Hinv_block).reshape(1,-1)**2)

        # ===== 动态N:M分配 =====
        def assign_nm_pattern(salience_list, M=8):
            """根据敏感度排名分配N值"""
            sorted_indices = torch.argsort(torch.tensor(salience_list), descending=True)
            n_values = []
            if sparsity<=0.4:
                four_eight = 0.25
            elif sparsity<=0.6:
                four_eight = 0.55
            else:
                four_eight = 0.4
            three_eight = 2.5 - 0.5*four_eight - 4*sparsity
            five_eight = 1 - three_eight - four_eight
            for idx in sorted_indices:
                if idx < len(salience_list)*three_eight:
                    n_values.append(3)
                elif idx < len(salience_list)*(three_eight + four_eight):
                    n_values.append(4)
                else:
                    n_values.append(5)
            return n_values
        
        # ===== 修改剪枝循环 =====
        all_saliences = []
        block_Hinvs = []
        
        # 首次遍历：计算所有块的敏感度
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            Hinv1 = Hinv[i1:i2, i1:i2]
            W1 = W[:, i1:i2].clone()
            
            # 将大块进一步划分为8x8子块
            for sub_i in range(0, W1.shape[1], 8):
                sub_block = W1[:, sub_i:sub_i+8]
                Hinv_sub = Hinv1[sub_i:sub_i+8, sub_i:sub_i+8]
                all_saliences.append(get_block_sensitivity(sub_block, Hinv_sub))
                block_Hinvs.append(Hinv_sub)
        
        # 分配N值（3:8/4:8/5:8）            
        n_values = assign_nm_pattern(all_saliences)
        #print(f"n_values: {n_values}")
        block_idx = 0

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            
            
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            
            
            Hinv1 = Hinv[i1:i2, i1:i2]
            

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    ### OBS pruning  
                    if sparsity_way == "origin":
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2 #saliences
                    ### Our Saliency
                    else:
                        tmp = W1 ** 2 * (torch.diag(self.H[i1:i2,i1:i2]).reshape((1, -1)) + 1.0 / torch.diag(Hinv1).reshape((1, -1))) 
                    #tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) 
                    # import pdb
                    # pdb.set_trace()
                    #tmp = W1 ** 2 * (torch.diag(self.H[i1:i2,i1:i2]).reshape((1, -1))) 
                    
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh #得到哪些位置应该被mask
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:    #n:m的半结构裁剪
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, n_values[block_idx], dim=1, largest=False)[1], True)
                    block_idx+=1

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                #
                Losses1[:, i] = (w - q) ** 2 / d ** 2  #delta L = W^2/d^2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) # 
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            #？？？
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])  #除了blocksize里面的需要更新，后面不在当前block里面的也需要更新

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()

    def average_trace(self):
        return torch.diag(self.H).mean()
