# 库函数的调用
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import os
import wfdb
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from scipy import signal
from tqdm import tqdm
import math
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torch.autograd as autograd
import sklearn.metrics
import csv
import torch.optim as optim


# 训练流程和学习率调度
def get_macro_f1(tar,pred):
    # 多分类多标签或单标签，输入为概率或one hot
    pred_idx = pred.argmax(axis=1)
    pred_=(pred_idx[:,None] == np.arange(pred.shape[1])).astype(int)
    macro_F1_score=sklearn.metrics.f1_score(tar, pred_, average='macro')
    return macro_F1_score

def get_class_f1(tar,pred):
    # 多分类多标签或单标签，输入为概率或one hot
    pred_idx = pred.argmax(axis=1)
    pred_=(pred_idx[:,None] == np.arange(pred.shape[1])).astype(int)
    F1_normal_score,F1_AF_score,F1_other_score=list(sklearn.metrics.f1_score(tar, pred_, average=None))
    return F1_normal_score,F1_AF_score,F1_other_score

def get_macro_precision(tar,pred):
    # 多分类单标签，输入为one hot
    pred_idx = pred.argmax(axis=1)
    pred_=(pred_idx[:,None] == np.arange(pred.shape[1])).astype(int)
    return sklearn.metrics.precision_score(tar, pred_, average='macro')

def get_macro_recall(tar,pred):
    # 多分类单标签，输入one hot
    pred_idx = pred.argmax(axis=1)
    pred_=(pred_idx[:,None] == np.arange(pred.shape[1])).astype(int)
    return sklearn.metrics.recall_score(tar, pred_, average='macro')

def get_acc(tar,pred):
    acc= ((np.argmax(tar, axis=1)==np.argmax(pred, axis=1)).sum())/tar.shape[0]
    return acc

def adjust_learning_rate(optimizer, current_iter,warm_iter,total_iter,max_lr,min_lr,mode='warm+constant'):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if current_iter < warm_iter:
        lr = max_lr * current_iter / warm_iter 
        
    else:
        if mode=='cycle':
            lr = min_lr + (max_lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (current_iter - warm_iter) / (total_iter - warm_iter)))
        elif mode=='warm+constant':
            lr=max_lr
        else:
            print('lr set error')
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def MMD_my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1),
                      x1,
                      x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res.clamp_min_(1e-30)

def MMD_gaussian_kernel(x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                       1000]):
    D = MMD_my_cdist(x, y)
    K = torch.zeros_like(D)

    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))

    return K

def MMD_mmd(x, y):
    Kxx = MMD_gaussian_kernel(x, x).mean()
    Kyy = MMD_gaussian_kernel(y, y).mean()
    Kxy = MMD_gaussian_kernel(x, y).mean()
    return Kxx + Kyy - 2 * Kxy


def CORAL_mmd(x, y):
    mean_x = x.mean(0, keepdim=True)
    mean_y = y.mean(0, keepdim=True)
    cent_x = x - mean_x
    cent_y = y - mean_y
    cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
    cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

    mean_diff = (mean_x - mean_y).pow(2).mean()
    cova_diff = (cova_x - cova_y).pow(2).mean()

    return mean_diff + cova_diff


def _irm_penalty(logits, y,device):
#     device = "cuda" if logits[0][0].is_cuda else "cpu"
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
    loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
    grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
    result = torch.sum(grad_1 * grad_2)
    return result


def irm_penalty(logits, y, device):
    scale = torch.tensor(1.0, device=device, requires_grad=True)
    loss_1 = F.cross_entropy(logits*scale, y)
    # Compute gradients with respect to the scale factor
    # Compute the IRM penalty as the dot product of gradients
    penalty = autograd.grad(loss_1, scale, create_graph=True)[0].pow(2).mean()    
    
    return penalty


# class global_count(int):
#     def __init__(self,):
#         self.count=0
#     def add1(self,):
#         self.count+=1

class FishMatrix:
    def __init__(self):
        self.grads = []

    def add_grad(self, grad):
        self.grads.append(grad)

    def get_fish_matrix(self):
        return torch.mean(torch.stack(self.grads), dim=0)

    
def repass_backward(model, model_copy_converged, train_loader, theta, device, loss_fn):
    model_copy_converged.zero_grad()
    grad_theta_sum = [torch.zeros_like(t) for t in theta]  # Initialize the sum with zeros

    for batch_idx, (in_data_, targets_, indices) in enumerate(train_loader):
        preds_ = model_copy_converged(in_data_.float().to(device)).float()
        targets_for_loss = torch.max(targets_[:, :3], 1).indices.long().to(device)
        loss = loss_fn(preds_, targets_for_loss)

        grad_weights_on_full_train = torch.autograd.grad(loss, model_copy_converged.parameters(), create_graph=True, allow_unused=True)

        grads = []
        for p, g in zip(model.parameters(), grad_weights_on_full_train):
            if g is not None:
                p.grad = g.data
                grads.append(torch.sum(p.view(-1) * g.view(-1), dim=0))
            else:
                p.grad = torch.zeros_like(p)

        grad_theta_batch = torch.autograd.grad(outputs=torch.stack(grads).sum(), inputs=[theta[idx] for idx in indices], allow_unused=True)
        for i, grad in enumerate(grad_theta_batch):
            if grad is not None:
                grad_theta_sum[i] += grad

    for i in range(len(grad_theta_sum)):
        grad_theta_sum[i] /= len(train_loader)

    return grad_theta_sum

class CosineSimilarityLoss(torch.nn.Module):  
    def __init__(self):  
        super(CosineSimilarityLoss, self).__init__()  
  
    def forward(self, input1, input2):  
        cos_sim = torch.mean(F.cosine_similarity(input1, input2))
        loss = 1 - cos_sim  
        return loss

loss_fn_tslb = CosineSimilarityLoss()

def at_spatial(x):
    if len(list(x.shape))==3:
        return F.normalize(x.pow(2),dim=1 )
    else:
        return x.pow(2)

def at_spatial_loss(x, y):
    return (at_spatial(x) - at_spatial(y)).pow(2).mean()
    
def train_one_epoch(epoch,warm_epoch, total_epoch, model,max_lr,min_lr,loss_fn,optimizer, base_optimizer, train_loader, device,DG_method=None,MMD_iter=4,MMD_gamma=1., irm_lambda=1.0, irm_penalty_anneal_iters=100, ali_weight=0.5, SAM_operate=False, recons_model=None,add_recons=None,DGGR_smooth_eps=0.,weights=0,theta=0,GRL_loss_beta=0.05,GRL_condition_beta=0.05,class_num=3,distill=False, domain_class=0,GAN_aug=None,GAN_model=None):
    # irm_lambda=1e2, irm_penalty_anneal_iters=10000000,效果很好  # irm_lambda=1, 后者=100，效果差一些。唯一的区别就是达到iters时重置了优化器。
    
    if (class_num==3)and(DG_method =='teacher_self_distill_leadI_GRL'):
        model.train()
        loss_beta=GRL_loss_beta#0.05
        running_loss=None
        for step, (in_data_, targets_) in (enumerate(train_loader)):
            current_iter=step+epoch*len(train_loader)
            warm_iter=warm_epoch*len(train_loader)
            total_iter=total_epoch*len(train_loader)
            adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
            optimizer.zero_grad()
            diag_preds_,domain_preds_,x_center_F1,x_center_F2,diag_preds_lead1,domain_preds_lead1,lead1_T1,lead1_T2=model(in_data_.float().to(device))
            diag_preds_=diag_preds_.float()
            domain_preds_=domain_preds_.float()
            diag_preds_lead1=diag_preds_lead1.float()
            domain_preds_lead1=domain_preds_lead1.float()
            targets_for_diag_loss=torch.max(targets_[:,:class_num],1).indices.long().to(device)  # 这里的class_num是指类别数量
            targets_for_domain_loss=torch.max(targets_[:,int(-1*domain_class):],1).indices.long().to(device)   # 这里的是指后面的领域情况。
            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
            loss_diag_lead1=loss_fn(diag_preds_lead1,targets_for_diag_loss)
            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
            loss_domain_lead1=loss_fn(domain_preds_lead1,targets_for_domain_loss)
            loss_sd_logits=loss_fn_tslb(diag_preds_, diag_preds_lead1)
            loss_sd_F1=at_spatial_loss(x_center_F1,lead1_T1)
            loss_sd_F2=at_spatial_loss(x_center_F2,lead1_T2)
            loss=loss_diag+loss_diag_lead1+loss_beta*loss_domain+loss_beta*loss_domain_lead1+0.3*loss_sd_logits+0.3*loss_sd_F1+0.3*loss_sd_F2

            loss.backward()
            optimizer.step()
            if running_loss is None:
                running_loss = loss.item() # 只输出重构损失
            else:
                running_loss = running_loss * .9 + loss.item() * .1   
        print('epoch=',epoch,)
        print("train_running_loss=",running_loss)
    else:
        if class_num==2:
            assert (DG_method in [None,'DG_GR','DG_GR_method_B','DG_GR_method_C','DG_GR_method_A','MMD','CORAL','IRM','IRM_new','teacher_self_distill_leadI_GRL']), 'Some DG not allowed now when 2 class!'

            if (DG_method in [None,'DG_GR_method_A']):
                model.train()
                running_loss = None 
        #         pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
                for step, (in_data_, targets_) in (enumerate(train_loader)):
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)
                    adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
                    optimizer.zero_grad()
                    if (distill==False):
                        preds_=model(in_data_.float().to(device))
                    else:
                        preds_,_=model(in_data_.float().to(device))

                    preds_=preds_.float()

                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)
                    loss.backward()
                    optimizer.step()
                    if running_loss is None:
                        running_loss = loss.item() # 只输出重构损失
                    else:
                        running_loss = running_loss * .9 + loss.item() * .1   
                print('epoch=',epoch,)
                print("train_running_loss=",running_loss)
        #         print("________________________________")
        #             description = f'epoch {epoch} loss: {running_loss:.4f}'
        #             pbar.set_description(description)
            elif DG_method in ['DG_GR','DG_GR_method_B','DG_GR_method_C']:
                model.train()
                loss_beta=GRL_loss_beta#0.05
                running_loss = None 
                for step, (in_data_, targets_) in (enumerate(train_loader)):
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)
                    adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
                    optimizer.zero_grad()
                    if add_recons==None:
                        if (distill==False):
                            diag_preds_,domain_preds_=model(in_data_.float().to(device))
                        else:
                            diag_preds_,domain_preds_,_=model(in_data_.float().to(device))
                    else:
                        print('error')
                    diag_preds_=diag_preds_.float()
                    domain_preds_=domain_preds_.float()
                    targets_for_diag_loss=torch.max(targets_[:,:2],1).indices.long().to(device)  # 这里的2是指两类。
                    loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                    targets_for_domain_loss=torch.max(targets_[:,int(-1*domain_class):],1).indices.long().to(device)   # 这里的2是指后面的领域情况。
                    loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                    loss=loss_diag+loss_beta*loss_domain
                    loss.backward()
                    optimizer.step()
                    if running_loss is None:
                        running_loss = loss.item() # 只输出重构损失
                    else:
                        running_loss = running_loss * .9 + loss.item() * .1   
                print('epoch=',epoch,)
                print("train_running_loss=",running_loss)

            elif (DG_method in ['teacher_self_distill_leadI_GRL']):
                model.train()
                loss_beta=GRL_loss_beta#0.05
                running_loss=None
                for step, (in_data_, targets_) in (enumerate(train_loader)):
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)
                    adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
                    optimizer.zero_grad()
                    diag_preds_,domain_preds_,x_center_F1,x_center_F2,diag_preds_lead1,domain_preds_lead1,lead1_T1,lead1_T2=model(in_data_.float().to(device))
                    diag_preds_=diag_preds_.float()
                    domain_preds_=domain_preds_.float()
                    diag_preds_lead1=diag_preds_lead1.float()
                    domain_preds_lead1=domain_preds_lead1.float()
                    targets_for_diag_loss=torch.max(targets_[:,:class_num],1).indices.long().to(device)  # 这里的class_num是指类别数量
                    targets_for_domain_loss=torch.max(targets_[:,int(-1*domain_class):],1).indices.long().to(device)   # 这里的是指后面的领域情况。
                    loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                    loss_diag_lead1=loss_fn(diag_preds_lead1,targets_for_diag_loss)
                    loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                    loss_domain_lead1=loss_fn(domain_preds_lead1,targets_for_domain_loss)
                    loss_sd_logits=loss_fn_tslb(diag_preds_, diag_preds_lead1)
                    loss_sd_F1=at_spatial_loss(x_center_F1,lead1_T1)
                    loss_sd_F2=at_spatial_loss(x_center_F2,lead1_T2)
                    loss=loss_diag+loss_diag_lead1+loss_beta*loss_domain+loss_beta*loss_domain_lead1+0.3*loss_sd_logits+0.3*loss_sd_F1+0.3*loss_sd_F2

                    loss.backward()
                    optimizer.step()
                    if running_loss is None:
                        running_loss = loss.item() # 只输出重构损失
                    else:
                        running_loss = running_loss * .9 + loss.item() * .1   
                print('epoch=',epoch,)
                print("train_running_loss=",running_loss)

            elif (DG_method in ['MMD']):  # kernel_type= "gaussian"
                model.train()
                running_loss = None 
        #         pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
                MMD_feature_list=[]
                for step, (in_data_, targets_) in (enumerate(train_loader)):
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)
                    adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
                    objective=0
                    penalty=0
                    features,classifs=model(in_data_.float().to(device))
                    targets_for_diag_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    objective+=loss_fn(classifs, targets_for_diag_loss)
                    MMD_feature_list.append(features)
                    if len(MMD_feature_list)==MMD_iter:
                        for MMD_i in range(MMD_iter):
                            for MMD_j in range(MMD_i+1,MMD_iter):
                                penalty += MMD_mmd(MMD_feature_list[MMD_i], MMD_feature_list[MMD_j])
                        objective /= MMD_iter
                        if len(MMD_feature_list) > 1:
                            penalty /= (len(MMD_feature_list) * (len(MMD_feature_list) - 1) / 2)
                        loss=objective + (MMD_gamma*penalty)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        MMD_feature_list=[]

            elif (DG_method in ['CORAL']):  # kernel_type= "mean_cov"  
                model.train()
                running_loss = None 
        #         pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
                MMD_feature_list=[]
                for step, (in_data_, targets_) in (enumerate(train_loader)):
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)
                    adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
                    objective=0
                    penalty=0
                    features,classifs=model(in_data_.float().to(device))
                    targets_for_diag_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    objective+=loss_fn(classifs, targets_for_diag_loss)
                    MMD_feature_list.append(features)
                    if len(MMD_feature_list)==MMD_iter:
                        for MMD_i in range(MMD_iter):
                            for MMD_j in range(MMD_i+1,MMD_iter):
                                penalty += CORAL_mmd(MMD_feature_list[MMD_i], MMD_feature_list[MMD_j])  # 惩罚计算方式的不同
                        objective /= MMD_iter
                        if len(MMD_feature_list) > 1:
                            penalty /= (len(MMD_feature_list) * (len(MMD_feature_list) - 1) / 2)
                        loss=objective + (MMD_gamma*penalty)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        MMD_feature_list=[]
            elif (DG_method in ['IRM']):
                model.train()
                running_loss = None 
        #         pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
                for step, (in_data_, targets_) in (enumerate(train_loader)):
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)
                    adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)

                    penalty_weight = (irm_lambda if current_iter 
                                                  >= irm_penalty_anneal_iters else
                                                  1.0)
                    logits=model(in_data_.float().to(device))
                    targets_for_diag_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    nll=loss_fn(logits,targets_for_diag_loss)
                    penalty=_irm_penalty(logits, targets_for_diag_loss,device)
                    loss = nll + (penalty_weight * penalty)
    #                 print(penalty_weight)
    #                 print(penalty)
    #                 print(nll)
    #                 print(loss)
    #                 print('____')
                    if current_iter==irm_penalty_anneal_iters:
                        optimizer=generate_optimizer(base_optimizer=base_optimizer, \
                                           model_parameters=model.parameters(),SAM_flag=False,weight_decay=1e-4)  #reset
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            elif (DG_method in ['IRM_new']):
                model.train()
                running_loss = None 
        #         pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
                for step, (in_data_, targets_) in (enumerate(train_loader)):
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)
                    adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)

                    logits=model(in_data_.float().to(device))
                    targets_for_diag_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    nll=loss_fn(logits,targets_for_diag_loss)
                    penalty=irm_penalty(logits, targets_for_diag_loss,device)
                    loss = irm_lambda*nll + (1-irm_lambda) * penalty
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


        else:
            if DG_method == 'DG_GR_ensemble':
                pass
            else:
                model.train() 
            loss_beta=GRL_loss_beta#0.05
            loss_beta_class=GRL_condition_beta#0.05
            running_loss = None 


            pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
            MMD_feature_list=[]
            fish_matrix = FishMatrix()
        #     IRM_update_count=0


            if DG_method in ['maple']:
                for step, (in_data_, targets_, indices ) in pbar:
                    preds_=model(in_data_.float().to(device)).float()
                    targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)
                    optimizer.zero_grad()
                    (loss * theta[indices]).sum().backward()
                    optimizer.step()


        #         # inner loop

        #         step_size=1e-2
        #         projection_epsilon=1e-8

        #         for step, (in_data_, targets_, sample_idx ) in pbar:
        #             current_iter=step+epoch*len(train_loader)
        #             warm_iter=warm_epoch*len(train_loader)
        #             total_iter=total_epoch*len(train_loader)
        #             adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
        #             assert SAM_operate==False
        #             preds_=model(in_data_.float().to(device)).float()
        #             loss = F.binary_cross_entropy_with_logits(preds_, targets_[:,:3].to(device), reduction='none')
        # #             print(weights[sample_idx].shape)
        #             weighted_loss = torch.mean(loss * weights[sample_idx].unsqueeze(1))
        #             optimizer.zero_grad()
        #             weighted_loss.backward()
        #             optimizer.step()

        #         # outer loop
        #         for step, (in_data_, targets_, sample_idx ) in pbar:
        #             preds_=model(in_data_.float().to(device)).float()
        #             loss = F.binary_cross_entropy_with_logits(preds_, targets_[:,:3].to(device), reduction='none')
        #             grad_weights = loss.detach().cpu().numpy()
        #             weights[sample_idx] -= step_size * grad_weights
        #             weights = torch.maximum(weights, projection_epsilon)  # Projection step


        #         try:
        #             if running_loss is None:
        #                 running_loss = loss_num # 只输出重构损失
        #             else:
        #                 running_loss = running_loss * .9 + loss_num * .1           
        #             description = f'epoch {epoch} loss: {running_loss:.4f}'
        #             pbar.set_description(description)

        #         except:
        #             try:
        #                 if running_loss is None:
        #                     running_loss = loss.item() # 只输出重构损失
        #                 else:
        #                     running_loss = running_loss * .9 + loss.item() * .1           
        #                 description = f'epoch {epoch} loss: {running_loss:.4f}'
        #                 pbar.set_description(description)
        #             except:
        #                 pass

            else:

                for step, (in_data_, targets_) in pbar:
                    current_iter=step+epoch*len(train_loader)
                    warm_iter=warm_epoch*len(train_loader)
                    total_iter=total_epoch*len(train_loader)

            #         print(DG_method)

                    if DG_method=='DG_GR_ensemble':
                        for model_type in model.keys():
                            adjust_learning_rate(optimizer[model_type],current_iter,warm_iter,total_iter,max_lr,min_lr)
                    else:
                        adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)

                    if SAM_operate==False:

                        if DG_method == 'class_condition_GRL':

                            if (torch.sum(targets_[:,0])<2)or(torch.sum(targets_[:,1])<2)or(torch.sum(targets_[:,2])<2):
                                continue



                            optimizer.zero_grad()
                            if add_recons==None:
                                diag_preds_,domain_preds_,domain_preds_n,domain_preds_a,domain_preds_o=model(in_data_.float().to(device))
                            else:
                                print('do not support')
                                return -1
                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()
                            domain_preds_n=domain_preds_n.float()
                            domain_preds_a=domain_preds_a.float()
                            domain_preds_o=domain_preds_o.float()

                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)

                            targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)

                            # Find the indices where targets_[index, 0] == 1
                            selected_indices_n = (targets_[:, 0] == 1).nonzero().squeeze()
        #                     print(selected_indices_n.shape)
        #                     print(type(selected_indices_n))
        #                     print(selected_indices_n)
                            targets_for_domain_loss_n = torch.max(targets_[selected_indices_n, 3:], 1).indices.long().to(device)
                            domain_preds_n_selected = domain_preds_n[selected_indices_n]
                            loss_domain_n = loss_fn(domain_preds_n_selected, targets_for_domain_loss_n)

                            # Find the indices where targets_[index, 1] == 1
                            selected_indices_a = (targets_[:, 1] == 1).nonzero().squeeze()
                            targets_for_domain_loss_a = torch.max(targets_[selected_indices_a, 3:], 1).indices.long().to(device)
                            domain_preds_a_selected = domain_preds_a[selected_indices_a]
                            loss_domain_a = loss_fn(domain_preds_a_selected, targets_for_domain_loss_a)

                            # Find the indices where targets_[index, 2] == 1
                            selected_indices_o = (targets_[:, 2] == 1).nonzero().squeeze()
                            targets_for_domain_loss_o = torch.max(targets_[selected_indices_o, 3:], 1).indices.long().to(device)
                            domain_preds_o_selected = domain_preds_o[selected_indices_o]
                            loss_domain_o = loss_fn(domain_preds_o_selected, targets_for_domain_loss_o)

                            loss=loss_diag+loss_beta*loss_domain+loss_beta_class*(loss_domain_n+loss_domain_a+loss_domain_o)
                            loss.backward()
                            optimizer.step()

                        elif DG_method in ['DG_GR','DG_GR_method_B','DG_GR_method_C']:
                            optimizer.zero_grad()
                            
                            
                            if add_recons==None:
                                
                                if (distill==False):
                                    diag_preds_,domain_preds_=model(in_data_.float().to(device))
                                else:
                                    diag_preds_,domain_preds_,_=model(in_data_.float().to(device))
                                    
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                diag_preds_,domain_preds_=model(in_data_2.float().to(device))
                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                            targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                            loss=loss_diag+loss_beta*loss_domain
                            loss.backward()
                            optimizer.step()

                        elif DG_method == 'DGGR_smooth':
                            optimizer.zero_grad()
                            if add_recons==None:
                                diag_preds_,domain_preds_=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                diag_preds_,domain_preds_=model(in_data_2.float().to(device))
                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
            #                 print(targets_for_diag_loss.shape)
                            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                            targets_for_domain_loss=targets_[:,3:].to(device)
                            if targets_for_domain_loss.shape[-1]==1:
                                pass
                            else:
                                targets_for_domain_loss=targets_for_domain_loss*(1-DGGR_smooth_eps)+DGGR_smooth_eps/(targets_for_domain_loss.shape[-1]-1)*(1-targets_for_domain_loss)

                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                            loss=loss_diag+loss_beta*loss_domain
                            loss.backward()
                            optimizer.step()

                        elif DG_method == 'DG_GR_ensemble': # 无attention，只是分别训练，输出简单聚合
                            model['origin'].train()
                            loss_num_list=[]
                            # origin
                            diag_preds_,domain_preds_=model['origin'](in_data_.float().to(device))
                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                            targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                            loss=loss_diag+loss_beta*loss_domain
                            loss_num_list.append(loss.item())
                            optimizer['origin'].zero_grad()
                            loss.backward()
                            optimizer['origin'].step()
                            # recons
                            for model_type in model.keys():
                                if model_type=='origin':
                                    pass
                                else:
                                    model[model_type].train()
                                    with torch.no_grad():
                                        signal_recons=recons_model[model_type](in_data_.float().to(device)).detach()
                                    diag_preds_,domain_preds_=model[model_type](signal_recons.float().to(device))
                                    diag_preds_=diag_preds_.float()
                                    domain_preds_=domain_preds_.float()
                                    targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                                    loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                                    targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                                    loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                                    loss=loss_diag+loss_beta*loss_domain
                                    loss_num_list.append(loss.item())
                                    optimizer[model_type].zero_grad()
                                    loss.backward()
                                    optimizer[model_type].step()

            #             elif DG_method == 'Fish':

            #                 # Fish 域泛化方法
            #                 optimizer.zero_grad()
            #                 diag_preds_ = model(in_data_.float().to(device))
            #                 targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
            #                 loss=loss_fn(diag_preds_,targets_for_diag_loss)
            #                 loss.backward(create_graph=True)  # 创建计算图以获取二阶导数

            #                 # 计算当前梯度并存储到 Fish 矩阵
            #                 grads = []
            #                 for param in model.parameters():
            #                     if param.requires_grad:
            #                         print(1)
            #                         print(param)
            #                         grads.append(param.grad.view(-1))
            #                 fish_matrix.add_grad(torch.cat(grads))

            #                 # 使用 Fish 矩阵更新参数
            #                 fish_mat = fish_matrix.get_fish_matrix()
            #                 with torch.no_grad():
            #                     for param in model.parameters():
            #                         if param.requires_grad:
            #                             param.add_(fish_mat * param.grad, alpha=-optimizer.param_groups[0]["lr"])

            #                 # 在此处添加您原有的其他代码，例如记录损失、准确率等。


                        elif DG_method == 'MMD': # kernel_type= "gaussian"
                            objective=0
                            penalty=0

                            if add_recons==None:
                                features,classifs=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                features,classifs=model(in_data_2.float().to(device))

            #                 features,classifs=model(in_data_.float().to(device))
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            objective+=loss_fn(classifs, targets_for_diag_loss)
                            MMD_feature_list.append(features)
                            if len(MMD_feature_list)==MMD_iter:
                                for MMD_i in range(MMD_iter):
                                    for MMD_j in range(MMD_i+1,MMD_iter):
                                        penalty += MMD_mmd(MMD_feature_list[MMD_i], MMD_feature_list[MMD_j])
                                objective /= MMD_iter
                                if len(MMD_feature_list) > 1:
                                    penalty /= (len(MMD_feature_list) * (len(MMD_feature_list) - 1) / 2)
                                loss=objective + (MMD_gamma*penalty)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                MMD_feature_list=[]

                        elif DG_method == 'CausIRL_MMD': # kernel_type= "gaussian"
                            objective=0
                            penalty=0

                            if add_recons==None:
                                features,classifs=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                features,classifs=model(in_data_2.float().to(device))
            #                 features,classifs=model(in_data_.float().to(device))
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            objective+=loss_fn(classifs+ 1e-16, targets_for_diag_loss)
                            MMD_feature_list.append(features)
                            if len(MMD_feature_list)==MMD_iter:
                                MMD_first = None
                                MMD_second = None
                                for MMD_i in range(MMD_iter):

                                    MMD_slice = np.random.randint(0, len(MMD_feature_list[MMD_i]))

                                    if MMD_first is None:
                                        MMD_first = MMD_feature_list[MMD_i][:MMD_slice]
                                        MMD_second = MMD_feature_list[MMD_i][MMD_slice:]
                                    else:
                                        MMD_first = torch.cat((MMD_first, MMD_feature_list[MMD_i][:MMD_slice]), 0)
                                        MMD_second = torch.cat((MMD_second, MMD_feature_list[MMD_i][MMD_slice:]), 0)

                                if len(MMD_first) > 1 and len(MMD_second) > 1:
                                    penalty = torch.nan_to_num(MMD_mmd(MMD_first, MMD_second))
                                else:
                                    penalty = torch.tensor(0)
                                objective /= MMD_iter

                                loss=objective + (MMD_gamma*penalty)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                MMD_feature_list=[]

                        elif DG_method == 'CORAL': # kernel_type= "mean_cov"  # 除了惩罚计算方式外，别的同MMD
                            objective=0
                            penalty=0

                            if add_recons==None:
                                features,classifs=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                features,classifs=model(in_data_2.float().to(device))
            #                 features,classifs=model(in_data_.float().to(device))
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            objective+=loss_fn(classifs, targets_for_diag_loss)
                            MMD_feature_list.append(features)
                            if len(MMD_feature_list)==MMD_iter:
                                for MMD_i in range(MMD_iter):
                                    for MMD_j in range(MMD_i+1,MMD_iter):
                                        penalty += CORAL_mmd(MMD_feature_list[MMD_i], MMD_feature_list[MMD_j])
                                objective /= MMD_iter
                                if len(MMD_feature_list) > 1:
                                    penalty /= (len(MMD_feature_list) * (len(MMD_feature_list) - 1) / 2)
                                loss=objective + (MMD_gamma*penalty)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                MMD_feature_list=[]


                        elif DG_method == 'CausIRL_CORAL': # kernel_type= "mean_cov"  # 除了惩罚计算方式外，别的同MMD
                            objective=0
                            penalty=0

                            if add_recons==None:
                                features,classifs=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                features,classifs=model(in_data_2.float().to(device))
            #                 features,classifs=model(in_data_.float().to(device))
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            objective+=loss_fn(classifs, targets_for_diag_loss)
                            MMD_feature_list.append(features)
                            if len(MMD_feature_list)==MMD_iter:
                                MMD_first = None
                                MMD_second = None
                                for MMD_i in range(MMD_iter):
                                    MMD_slice = np.random.randint(0, len(MMD_feature_list[MMD_i]))

                                    if MMD_first is None:
                                        MMD_first = MMD_feature_list[MMD_i][:MMD_slice]
                                        MMD_second = MMD_feature_list[MMD_i][MMD_slice:]
                                    else:
                                        MMD_first = torch.cat((MMD_first, MMD_feature_list[MMD_i][:MMD_slice]), 0)
                                        MMD_second = torch.cat((MMD_second, MMD_feature_list[MMD_i][MMD_slice:]), 0)

                                if len(MMD_first) > 1 and len(MMD_second) > 1:
                                    penalty = torch.nan_to_num(CORAL_mmd(MMD_first, MMD_second))
                                else:
                                    penalty = torch.tensor(0)
                                objective /= MMD_iter

                                loss=objective + (MMD_gamma*penalty)
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                MMD_feature_list=[]


                        elif DG_method == 'IRM':
                            penalty_weight = (irm_lambda if current_iter 
                                                          >= irm_penalty_anneal_iters else
                                                          1.0)
                            if add_recons==None:
                                logits=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                logits=model(in_data_2.float().to(device))

            #                 logits=model(in_data_.float().to(device))
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            nll=loss_fn(logits,targets_for_diag_loss)
                            penalty=_irm_penalty(logits, targets_for_diag_loss,device)
                            loss = nll + (penalty_weight * penalty)
                            if current_iter==irm_penalty_anneal_iters:
                                optimizer=generate_optimizer(base_optimizer=base_optimizer, \
                                                   model_parameters=model.parameters(),SAM_flag=False,weight_decay=1e-4)  #reset
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        elif DG_method == 'DG_GR+IRM':
                            penalty_weight = (irm_lambda if current_iter 
                                                          >= irm_penalty_anneal_iters else
                                                          1.0)

                            if add_recons==None:
                                diag_preds_,domain_preds_=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                diag_preds_,domain_preds_=model(in_data_2.float().to(device))

                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()

                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            nll=loss_fn(diag_preds_,targets_for_diag_loss)

                            targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)

                            penalty=_irm_penalty(diag_preds_, targets_for_diag_loss,device)
                            loss = nll + (penalty_weight * penalty) + loss_beta*loss_domain

                            if current_iter==irm_penalty_anneal_iters:
                                optimizer=generate_optimizer(base_optimizer=base_optimizer, \
                                                   model_parameters=model.parameters(),SAM_flag=False,weight_decay=1e-4)  #reset

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()


                        elif DG_method=='cls_awr_ali':
                            weight = torch.tensor(ali_weight, device=device)
                            if add_recons==None:
                                output=model(in_data_.float().to(device))
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                output=model(in_data_2.float().to(device))
            #                 output=model(in_data_.float().to(device))
                            soft_label=F.softmax(output, dim=1)
                            target=targets_[:,:3].float().to(device)
                            cross_entropy_loss = loss_fn(output, target)
                            # Calculate the Euclidean distance between the center and each soft label
                            pdist=torch.nn.PairwiseDistance(p=2) 
                            c=soft_label[target==0]
                            ns_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
                            c=soft_label[target==1]
                            s_align_loss=torch.pow(pdist(c,c.mean(dim=0)),2).sum()/len(c)
                            align_loss=ns_align_loss+s_align_loss
                            loss = cross_entropy_loss + ali_weight*align_loss
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()              

                        elif DG_method=='origin_add_fft_amp':
            #                 print(DG_method)
                            optimizer.zero_grad()
                            if add_recons==None:

                                # Sampling frequency
                                sampling_frequency = 100

                                # Compute the FFT
                                fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                                # Number of data points
                                N = 1000

                                # Normalize the FFT amplitudes by dividing by N
                                fft_signal_normalized_tensor = fft_signal_tensor / N

                                # Compute the amplitude spectrum
                                amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)

                                # Detach the tensor to remove gradient tracking
                                amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

            #                     print(amplitude_spectrum_tensor_no_grad.shape)

                                in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                                preds_=model(in_data_fft_amp.float().to(device)).float()

                            else:
                                print("don't support!")
                            targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss=loss_fn(preds_,targets_for_loss)
                            loss.backward()
                            optimizer.step()


                        elif DG_method=='origin_add_fft_amp_DGGR':
            #                 print(DG_method)
                            optimizer.zero_grad()
                            if add_recons==None:

                                # Sampling frequency
                                sampling_frequency = 100

                                # Compute the FFT
                                fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                                # Number of data points
                                N = 1000

                                # Normalize the FFT amplitudes by dividing by N
                                fft_signal_normalized_tensor = fft_signal_tensor / N

                                # Compute the amplitude spectrum
                                amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)

                                # Detach the tensor to remove gradient tracking
                                amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

            #                     print(amplitude_spectrum_tensor_no_grad.shape)

                                in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                                diag_preds_,domain_preds_=model(in_data_fft_amp.float().to(device))



                            else:
                                print("don't support!")

                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                            targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                            loss=loss_diag+loss_beta*loss_domain
                            loss.backward()
                            optimizer.step()

                        elif DG_method=='origin_add_fft_amp_DGGR_smooth':
            #                 print(DG_method)
                            optimizer.zero_grad()
                            if add_recons==None:

                                # Sampling frequency
                                sampling_frequency = 100

                                # Compute the FFT
                                fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                                # Number of data points
                                N = 1000

                                # Normalize the FFT amplitudes by dividing by N
                                fft_signal_normalized_tensor = fft_signal_tensor / N

                                # Compute the amplitude spectrum
                                amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)

                                # Detach the tensor to remove gradient tracking
                                amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

            #                     print(amplitude_spectrum_tensor_no_grad.shape)

                                in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                                diag_preds_,domain_preds_=model(in_data_fft_amp.float().to(device))



                            else:
                                print("don't support!")

                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
            #                 targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                            targets_for_domain_loss=targets_[:,3:].to(device)
                            if targets_for_domain_loss.shape[-1]==1:
                                pass
                            else:
                                targets_for_domain_loss=targets_for_domain_loss*(1-DGGR_smooth_eps)+DGGR_smooth_eps/(targets_for_domain_loss.shape[-1]-1)*(1-targets_for_domain_loss)
                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                            loss=loss_diag+loss_beta*loss_domain
                            loss.backward()
                            optimizer.step()


                        elif DG_method=='origin_add_fft_phase':     # 虽然说是amplitude但是实际用了angle为相位
            #                 print(DG_method)
                            optimizer.zero_grad()
                            if add_recons==None:

                                # Sampling frequency
                                sampling_frequency = 100

                                # Compute the FFT
                                fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                                # Number of data points
                                N = 1000

                                # Normalize the FFT amplitudes by dividing by N
                                fft_signal_normalized_tensor = fft_signal_tensor / N

                                # Compute the amplitude spectrum
                                amplitude_spectrum_tensor = torch.angle(fft_signal_normalized_tensor)

                                # Detach the tensor to remove gradient tracking
                                amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

            #                     print(amplitude_spectrum_tensor_no_grad.shape)

                                in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                                preds_=model(in_data_fft_amp.float().to(device)).float()

                            else:
                                print("don't support!")
                            targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss=loss_fn(preds_,targets_for_loss)
                            loss.backward()
                            optimizer.step()

                        elif DG_method=='origin_add_fft_amp_phase':     # 虽然说是amplitude但是实际用了angle为相位
            #                 print(DG_method)
                            optimizer.zero_grad()
                            if add_recons==None:

                                # Sampling frequency
                                sampling_frequency = 100

                                # Compute the FFT
                                fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                                # Number of data points
                                N = 1000

                                # Normalize the FFT amplitudes by dividing by N
                                fft_signal_normalized_tensor = fft_signal_tensor / N

                                # Compute the amplitude spectrum
                                amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)
                                phase_spectrum_tensor = torch.angle(fft_signal_normalized_tensor)

                                # Detach the tensor to remove gradient tracking
                                amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()
                                phase_spectrum_tensor_no_grad = phase_spectrum_tensor.detach()

            #                     print(amplitude_spectrum_tensor_no_grad.shape)

                                in_data_fft_amp_phase=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad,phase_spectrum_tensor_no_grad),dim=-1)

                                preds_=model(in_data_fft_amp_phase.float().to(device)).float()

                            else:
                                print("don't support!")
                            targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss=loss_fn(preds_,targets_for_loss)
                            loss.backward()
                            optimizer.step()


                        elif DG_method in [None,'DG_GR_method_A']:
        #                     print(DG_method)
                            optimizer.zero_grad()
                            if add_recons==None:
                    
                                if (GAN_aug=='0211'):
#                                     GAN_model
                                    with torch.no_grad():
                                        in_data_1 = in_data_.float().to(device).clone()
                                        in_data_2 = GAN_model.generate_images(in_data_1)  # 32,
                                        in_data_3 = GAN_model.generate_images(in_data_1)  # 32,
                                        in_data_4 = GAN_model.generate_images(in_data_1)  # 32,
                                    
                                        in_data_=torch.cat((in_data_1,in_data_2,in_data_3,in_data_4),dim=0)
                                    
                
                                if (distill==False):
                                    preds_=model(in_data_.float().to(device))
                                else:
                                    preds_,_=model(in_data_.float().to(device))
                            
                            
#                                 preds_=model(in_data_.float().to(device)).float()
                            else:
                                with torch.no_grad():
                                    signal_recons=recons_model(in_data_.float().to(device)).detach()
                                    in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                                preds_=model(in_data_2.float().to(device)).float()
            #                 preds_=model(in_data_.float().to(device)).float()
                            targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            if (GAN_aug=='0211'):
                                targets_for_loss=torch.cat((targets_for_loss,targets_for_loss,targets_for_loss,targets_for_loss),dim=0)
                                
                
                            loss=loss_fn(preds_,targets_for_loss)
                            loss.backward()
                            optimizer.step()

                    elif SAM_operate==True:
                        if add_recons!=None:
                            print('Do not support SAM with recons Now')
                            return 0
                        optimizer.zero_grad()

                        if DG_method == 'DG_GR':
                            diag_preds_,domain_preds_=model(in_data_.float().to(device))
                            diag_preds_=diag_preds_.float()
                            domain_preds_=domain_preds_.float()
                            targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
                            targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
                            loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
                            loss=loss_diag+loss_beta*loss_domain
                        else:
                            preds_=model(in_data_.float().to(device)).float()
                            targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                            loss=loss_fn(preds_,targets_for_loss)
                        loss.backward()#(retain_graph=True)
                        optimizer.first_step(zero_grad=True)

            #             if DG_method == 'DG_GR':
            #                 diag_preds_,domain_preds_=model(in_data_.float().to(device))
            #                 diag_preds_=diag_preds_.float()
            #                 domain_preds_=domain_preds_.float()
            #                 targets_for_diag_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
            #                 loss_diag=loss_fn(diag_preds_,targets_for_diag_loss)
            #                 targets_for_domain_loss=torch.max(targets_[:,3:],1).indices.long().to(device)
            #                 loss_domain=loss_fn(domain_preds_,targets_for_domain_loss)
            #                 loss=loss_diag+loss_beta*loss_domain
            #             else:
            #                 preds_=model(in_data_.float().to(device)).float()
            #                 targets_for_loss=torch.max(targets_,1).indices.long().to(device)
            #                 loss=loss_fn(preds_,targets_for_loss)
            #             loss.backward()#(retain_graph=True)
            #             optimizer.second_step(zero_grad=True)

                    else:
                        print('SAM set error!')

                    try:
                        if running_loss is None:
                            running_loss = loss_num # 只输出重构损失
                        else:
                            running_loss = running_loss * .9 + loss_num * .1           
                        description = f'epoch {epoch} loss: {running_loss:.4f}'
                        pbar.set_description(description)

                    except:
                        try:
                            if running_loss is None:
                                running_loss = loss.item() # 只输出重构损失
                            else:
                                running_loss = running_loss * .9 + loss.item() * .1           
                            description = f'epoch {epoch} loss: {running_loss:.4f}'
                            pbar.set_description(description)
                        except:
                            pass
    return running_loss
    
    
    






def val_one_epoch(epoch,warm_epoch, total_epoch, model,loss_fn, test_loader, device,DG_method=None,class_F1=False, recons_model=None,add_recons=None,class_num=3,distill=False):
    
    if (class_num==3)and(DG_method =='teacher_self_distill_leadI_GRL'):
        model.eval() 
        running_loss = None 
        test_pred=0
        running_loss = []
        test_tar=0
        for step, (in_data_, targets_) in (enumerate(test_loader)):
            preds_,_,_,_,_,_,_,_=model(in_data_.float().to(device))
            preds_=preds_.float()
            targets_for_loss=torch.max(targets_[:,:class_num],1).indices.long().to(device)
            loss=loss_fn(preds_,targets_for_loss)    
            running_loss.append(loss.item())
    #             loss.item() # 只输出重构损失
            mean_loss=np.mean(running_loss)
    #             running_loss = running_loss * .95 + loss.item() * .05           
#             description = f'epoch {epoch} loss: {mean_loss:.4f}'
#             pbar.set_description(description)


            # 使用AUC作为评价指标
            if type(test_pred)==int:
                test_pred=preds_.clone().detach().cpu().numpy()
            else:
                test_pred=np.concatenate((test_pred,
                                    preds_.clone().detach().cpu().numpy(),
                                    ),axis=0)
            if type(test_tar)==int:
                test_tar=targets_.clone().detach().cpu().numpy()
            else:
                test_tar=np.concatenate((test_tar,
                                    targets_.clone().detach().cpu().numpy(),
                                    ),axis=0)

        print('epoch=',epoch,)
        print("val_loss=",mean_loss)
        print("________________________________")
        
    else:    
        
        if class_num==2:
            assert (DG_method in [None,'DG_GR','DG_GR_method_B','DG_GR_method_C','DG_GR_method_A','MMD','CORAL','IRM','IRM_new','teacher_self_distill_leadI_GRL']), 'Some DG not allowed now when 2 class!'

            if (DG_method in [None,'DG_GR_method_A','IRM','IRM_new']):

                model.eval()
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
        #         pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
                for step, (in_data_, targets_) in (enumerate(test_loader)):

                    if (distill==False):
                        preds_=model(in_data_.float().to(device))
                    else:
                        preds_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)    
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05           
        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)


                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("val_loss=",mean_loss)
                print("________________________________")

            elif (DG_method in ['DG_GR','DG_GR_method_B','DG_GR_method_C']):
                model.eval() 
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
                for step, (in_data_, targets_) in (enumerate(test_loader)):
                    if (distill==False):
                        preds_,_=model(in_data_.float().to(device))
                    else:
                        preds_,_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)    
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05           
        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)


                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("val_loss=",mean_loss)
                print("________________________________")
            elif (DG_method in ['teacher_self_distill_leadI_GRL']):
                model.eval() 
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
                for step, (in_data_, targets_) in (enumerate(test_loader)):
                    preds_,_,_,_,_,_,_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)    
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05           
        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)


                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("val_loss=",mean_loss)
                print("________________________________")






            elif (DG_method in ['MMD','CORAL']):
                model.eval()
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
                for step, (in_data_, targets_) in (enumerate(test_loader)):
                    _,preds_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)    
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05           
        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)


                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("val_loss=",mean_loss)
                print("________________________________")


    #         到这里没写完，GRRL还没加distill，val和test也没加GRRL，也没加GRRL的distill ,DGGR 要把3改成2

        else:

            if DG_method == 'DG_GR_ensemble':
                pass
            else:
                model.eval() 
        #     model.eval() 
            running_loss = None 
            test_pred=0
            running_loss = []
            test_tar=0
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
            for step, (in_data_, targets_) in pbar:
                if (DG_method in ['DG_GR','DG_GR+IRM','DGGR_smooth','DG_GR_method_B','DG_GR_method_C']):
                    if add_recons==None:
                        
                        if (distill==False):
                            preds_,_=model(in_data_.float().to(device))
                        else:
                            preds_,_,_=model(in_data_.float().to(device))
                                    
                        
                    else:
                        with torch.no_grad():
                            signal_recons=recons_model(in_data_.float().to(device)).detach()
                            in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                        preds_,_=model(in_data_2.float().to(device))
        #             preds_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()

                elif (DG_method in ['class_condition_GRL']):
                    if add_recons==None:
                        preds_,_,_,_,_=model(in_data_.float().to(device))
                    else:
                        print('do not support')
                        return -1
                    preds_=preds_.float()

                elif (DG_method in ['DG_GR_ensemble']):
                    model['origin'].eval()
                    loss_num_list=[]
                    preds_sum=0
                    preds_num=0
                    preds_,_=model['origin'](in_data_.float().to(device))
                    preds_=preds_.clone().detach()
                    preds_sum=preds_sum+preds_.clone().detach()
                    preds_num+=1

                    for model_type in model.keys():
                        if model_type=='origin':
                            pass
                        else:
                            model[model_type].eval()
                            with torch.no_grad():
                                signal_recons=recons_model[model_type](in_data_.float().to(device)).detach()
                            preds_,_=model[model_type](in_data_.float().to(device))
                            preds_=preds_.clone().detach()
                            preds_sum=preds_sum+preds_.clone().detach()

                    loss_num=-1 # 懒得算了
                    preds_=preds_sum/preds_num
                    preds_=preds_.float()


                elif DG_method in ['MMD','CORAL','CausIRL_MMD','CausIRL_CORAL']:
                    if add_recons==None:
                        _,preds_=model(in_data_.float().to(device))
                    else:
                        with torch.no_grad():
                            signal_recons=recons_model(in_data_.float().to(device)).detach()
                            in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                        _,preds_=model(in_data_2.float().to(device))
        #             _,preds_=model(in_data_.float().to(device))
                    preds_=preds_.float()

                elif DG_method=='origin_add_fft_amp':
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                        preds_=model(in_data_fft_amp.float().to(device))

                    else:
                        print("don't support!")

                elif DG_method in ['origin_add_fft_amp_DGGR','origin_add_fft_amp_DGGR_smooth']:
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                        preds_,_=model(in_data_fft_amp.float().to(device))

                    else:
                        print("don't support!")

                elif DG_method=='origin_add_fft_phase':
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.angle(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                        preds_=model(in_data_fft_amp.float().to(device))

                    else:
                        print("don't support!")

                elif DG_method=='origin_add_fft_amp_phase':
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)
                        phase_spectrum_tensor = torch.angle(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()
                        phase_spectrum_tensor_no_grad = phase_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp_phase=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad,phase_spectrum_tensor_no_grad),dim=-1)

                        preds_=model(in_data_fft_amp_phase.float().to(device))

                    else:
                        print("don't support!")

                else:
                    if add_recons==None:
#                         preds_=model(in_data_.float().to(device)).float()
                        
                
                        if (distill==False):
                            preds_=model(in_data_.float().to(device))
                        else:
                            preds_,_=model(in_data_.float().to(device))
                    else:
                        with torch.no_grad():
                            signal_recons=recons_model(in_data_.float().to(device)).detach()
                            in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                        preds_=model(in_data_2.float().to(device)).float()
        #             preds_=model(in_data_.float().to(device)).float()

                targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                loss=loss_fn(preds_,targets_for_loss)    
                running_loss.append(loss.item())
        #             loss.item() # 只输出重构损失
                mean_loss=np.mean(running_loss)
        #             running_loss = running_loss * .95 + loss.item() * .05           
                description = f'epoch {epoch} loss: {mean_loss:.4f}'
                pbar.set_description(description)
                # 使用AUC作为评价指标
                if type(test_pred)==int:
                    test_pred=preds_.clone().detach().cpu().numpy()
                else:
                    test_pred=np.concatenate((test_pred,
                                        preds_.clone().detach().cpu().numpy(),
                                        ),axis=0)
                if type(test_tar)==int:
                    test_tar=targets_.clone().detach().cpu().numpy()
                else:
                    test_tar=np.concatenate((test_tar,
                                        targets_.clone().detach().cpu().numpy(),
                                        ),axis=0)



                
                
                
#     print('val:',test_tar.shape)
#     print('val:',test_pred.shape)
    mean_macro_auc=sklearn.metrics.roc_auc_score(test_tar, test_pred, average='macro')
#     print('test mean_macro_auc= ',mean_macro_auc)
    
#     print(test_tar.shape)
#     print(test_pred.shape)
    
    F1_score=get_macro_f1(test_tar,test_pred)
#     print('test F1_score= ',F1_score)
    
    precision=get_macro_precision(test_tar,test_pred)
#     print('test precision= ',precision)
    
    recall=get_macro_recall(test_tar,test_pred)
#     print('test recall= ',recall)

    acc=get_acc(test_tar,test_pred)
#     print('test acc= ',acc)
    
    if class_F1==True:
        F1_normal_score,F1_AF_score,F1_other_score=get_class_f1(test_tar, test_pred)
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*F1_normal_score))),str(int(round(1000*F1_AF_score))),str(int(round(1000*F1_other_score))),str(int(round(1000*acc)))
        
        
    else:    
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*acc)))

    
def test_one_epoch(epoch,warm_epoch, total_epoch, model,loss_fn, test_loader, device,DG_method=None,class_F1=False, recons_model=None,add_recons=None,class_num=3,distill=False):
    
    if (class_num==3)and(DG_method =='teacher_self_distill_leadI_GRL'):
        model.eval() 
        running_loss = None 
        test_pred=0
        running_loss = []
        test_tar=0
        for step, (in_data_, targets_) in (enumerate(test_loader)):
            preds_,_,_,_,_,_,_,_=model(in_data_.float().to(device))
            preds_=preds_.float()
            targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
            loss=loss_fn(preds_,targets_for_loss)   
            running_loss.append(loss.item())
    #             loss.item() # 只输出重构损失
            mean_loss=np.mean(running_loss)
    #             running_loss = running_loss * .95 + loss.item() * .05   


#             description = f'epoch {epoch} loss: {mean_loss:.4f}'
#             pbar.set_description(description)
            # 使用AUC作为评价指标
            if type(test_pred)==int:
                test_pred=preds_.clone().detach().cpu().numpy()
            else:
                test_pred=np.concatenate((test_pred,
                                    preds_.clone().detach().cpu().numpy(),
                                    ),axis=0)
            if type(test_tar)==int:
                test_tar=targets_.clone().detach().cpu().numpy()
            else:
                test_tar=np.concatenate((test_tar,
                                    targets_.clone().detach().cpu().numpy(),
                                    ),axis=0)

        print('epoch=',epoch,)
        print("test_loss=",mean_loss)
        print("________________________________") 
    
    else:
        
        if class_num==2:
            assert (DG_method in [None,'DG_GR','DG_GR_method_B','DG_GR_method_C','DG_GR_method_A','MMD','CORAL','IRM','IRM_new','teacher_self_distill_leadI_GRL']), 'Some DG not allowed now when 2 class!'

            if (DG_method in [None,'DG_GR_method_A','IRM','IRM_new']):

                model.eval()
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
        #         pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
                for step, (in_data_, targets_) in (enumerate(test_loader)):
                    if (distill==False):
                        preds_=model(in_data_.float().to(device))
                    else:
                        preds_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:class_num],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)    
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05   


        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)
                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("test_loss=",mean_loss)
                print("________________________________")

            elif (DG_method in ['DG_GR','DG_GR_method_B','DG_GR_method_C']):
                model.eval() 
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
                for step, (in_data_, targets_) in (enumerate(test_loader)):
                    if (distill==False):
                        preds_,_=model(in_data_.float().to(device))
                    else:
                        preds_,_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)   
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05   


        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)
                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("test_loss=",mean_loss)
                print("________________________________") 

            elif (DG_method in ['teacher_self_distill_leadI_GRL']):
                model.eval() 
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
                for step, (in_data_, targets_) in (enumerate(test_loader)):
                    preds_,_,_,_,_,_,_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)   
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05   


        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)
                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("test_loss=",mean_loss)
                print("________________________________") 


            elif (DG_method in ['MMD','CORAL']):
                model.eval() 
                running_loss = None 
                test_pred=0
                running_loss = []
                test_tar=0
                for step, (in_data_, targets_) in (enumerate(test_loader)):
                    _,preds_=model(in_data_.float().to(device))
                    preds_=preds_.float()
                    targets_for_loss=torch.max(targets_[:,:2],1).indices.long().to(device)
                    loss=loss_fn(preds_,targets_for_loss)   
                    running_loss.append(loss.item())
            #             loss.item() # 只输出重构损失
                    mean_loss=np.mean(running_loss)
            #             running_loss = running_loss * .95 + loss.item() * .05   


        #             description = f'epoch {epoch} loss: {mean_loss:.4f}'
        #             pbar.set_description(description)
                    # 使用AUC作为评价指标
                    if type(test_pred)==int:
                        test_pred=preds_.clone().detach().cpu().numpy()
                    else:
                        test_pred=np.concatenate((test_pred,
                                            preds_.clone().detach().cpu().numpy(),
                                            ),axis=0)
                    if type(test_tar)==int:
                        test_tar=targets_.clone().detach().cpu().numpy()
                    else:
                        test_tar=np.concatenate((test_tar,
                                            targets_.clone().detach().cpu().numpy(),
                                            ),axis=0)

                print('epoch=',epoch,)
                print("test_loss=",mean_loss)
                print("________________________________") 







        else:
            if DG_method == 'DG_GR_ensemble':
                pass
            else:
                model.eval() 
            running_loss = None 
            test_pred=0
            running_loss = []
            test_tar=0
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
            for step, (in_data_, targets_) in pbar:

#                 if (DG_method in ['DG_GR','DG_GR+IRM','DGGR_smooth']):
                if (DG_method in ['DG_GR','DG_GR+IRM','DGGR_smooth','DG_GR_method_B','DG_GR_method_C']):
        #         if (DG_method=='DG_GR'):
                    if add_recons==None:
                
                        if (distill==False):
                            preds_,_=model(in_data_.float().to(device))
                        else:
                            preds_,_,_=model(in_data_.float().to(device))
                            
                    else:
                        with torch.no_grad():
                            signal_recons=recons_model(in_data_.float().to(device)).detach()
                            in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                        preds_,_=model(in_data_2.float().to(device)).float()
        #             preds_,_=model(in_data_.float().to(device))
                    preds_=preds_.float()

                elif (DG_method in ['class_condition_GRL']):
                    if add_recons==None:
                        preds_,_,_,_,_=model(in_data_.float().to(device))
                    else:
                        print('do not support')
                        return -1
                    preds_=preds_.float()

                elif (DG_method in ['DG_GR_ensemble']):
                    model['origin'].eval()
                    loss_num_list=[]
                    preds_sum=0
                    preds_num=0
                    preds_,_=model['origin'](in_data_.float().to(device))
                    preds_=preds_.clone().detach()
                    preds_sum=preds_sum+preds_.clone().detach()
                    preds_num+=1

                    for model_type in model.keys():
                        if model_type=='origin':
                            pass
                        else:
                            with torch.no_grad():
                                signal_recons=recons_model[model_type](in_data_.float().to(device)).detach()
                            preds_,_=model[model_type](in_data_.float().to(device))
                            preds_=preds_.clone().detach()
                            preds_sum=preds_sum+preds_.clone().detach()

                    loss_num=-1 # 懒得算了
                    preds_=preds_sum/preds_num
                    preds_=preds_.float()

                elif DG_method in ['MMD','CORAL','CausIRL_MMD','CausIRL_CORAL']:
                    if add_recons==None:
                        _,preds_=model(in_data_.float().to(device))
                    else:
                        with torch.no_grad():
                            signal_recons=recons_model(in_data_.float().to(device)).detach()
                            in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                        _,preds_=model(in_data_2.float().to(device))
        #             _,preds_=model(in_data_.float().to(device))
                    preds_=preds_.float()

                elif DG_method=='origin_add_fft_amp':
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                        preds_=model(in_data_fft_amp.float().to(device))

                    else:
                        print("don't support!")


                elif DG_method in ['origin_add_fft_amp_DGGR','origin_add_fft_amp_DGGR_smooth']:
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                        preds_,_=model(in_data_fft_amp.float().to(device))

                    else:
                        print("don't support!")


                elif DG_method=='origin_add_fft_phase':
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.angle(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad),dim=-1)

                        preds_=model(in_data_fft_amp.float().to(device))

                    else:
                        print("don't support!")

                elif DG_method=='origin_add_fft_amp_phase':
                    if add_recons==None:

                        # Sampling frequency
                        sampling_frequency = 100

                        # Compute the FFT
                        fft_signal_tensor = torch.fft.fft(in_data_, dim=1)

                        # Number of data points
                        N = 1000

                        # Normalize the FFT amplitudes by dividing by N
                        fft_signal_normalized_tensor = fft_signal_tensor / N

                        # Compute the amplitude spectrum
                        amplitude_spectrum_tensor = torch.abs(fft_signal_normalized_tensor)
                        phase_spectrum_tensor = torch.angle(fft_signal_normalized_tensor)

                        # Detach the tensor to remove gradient tracking
                        amplitude_spectrum_tensor_no_grad = amplitude_spectrum_tensor.detach()
                        phase_spectrum_tensor_no_grad = phase_spectrum_tensor.detach()

        #                 print(amplitude_spectrum_tensor_no_grad.shape)

                        in_data_fft_amp_phase=torch.cat((in_data_,amplitude_spectrum_tensor_no_grad,phase_spectrum_tensor_no_grad),dim=-1)

                        preds_=model(in_data_fft_amp_phase.float().to(device))

                    else:
                        print("don't support!")

                else:
                    if add_recons==None:
                
                        if (distill==False):
                            preds_=model(in_data_.float().to(device))
                        else:
                            preds_,_=model(in_data_.float().to(device))
                            
#                         preds_=model(in_data_.float().to(device)).float()
                            
                           
                    else:
                        with torch.no_grad():
                            signal_recons=recons_model(in_data_.float().to(device)).detach()
                            in_data_2=torch.cat((in_data_.float().to(device),signal_recons),dim=-1)
                        preds_=model(in_data_2.float().to(device))
        #             preds_=model(in_data_.float().to(device)).float()

                targets_for_loss=torch.max(targets_[:,:3],1).indices.long().to(device)
                loss=loss_fn(preds_,targets_for_loss)    
                running_loss.append(loss.item())
        #             loss.item() # 只输出重构损失
                mean_loss=np.mean(running_loss)
        #             running_loss = running_loss * .95 + loss.item() * .05           
                description = f'epoch {epoch} loss: {mean_loss:.4f}'
                pbar.set_description(description)
                # 使用AUC作为评价指标
                if type(test_pred)==int:
                    test_pred=preds_.clone().detach().cpu().numpy()
                else:
                    test_pred=np.concatenate((test_pred,
                                        preds_.clone().detach().cpu().numpy(),
                                        ),axis=0)
                if type(test_tar)==int:
                    test_tar=targets_.clone().detach().cpu().numpy()
                else:
                    test_tar=np.concatenate((test_tar,
                                        targets_.clone().detach().cpu().numpy(),
                                        ),axis=0)

#     print('test:',test_tar.shape)
#     print('test:',test_pred.shape)
    
    
    mean_macro_auc=sklearn.metrics.roc_auc_score(test_tar, test_pred, average='macro')
    print('test mean_macro_auc= ',mean_macro_auc)
    
    F1_score=get_macro_f1(test_tar,test_pred)
    print('test F1_score= ',F1_score)
    
    precision=get_macro_precision(test_tar,test_pred)
    print('test precision= ',precision)
    
    recall=get_macro_recall(test_tar,test_pred)
    print('test recall= ',recall)

    acc=get_acc(test_tar,test_pred)
    print('test acc= ',acc)
    
    if class_F1==True:
        F1_normal_score,F1_AF_score,F1_other_score=get_class_f1(test_tar, test_pred)
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*F1_normal_score))),str(int(round(1000*F1_AF_score))),str(int(round(1000*F1_other_score))),str(int(round(1000*acc)))
        
        
    else:    
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*acc)))


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()


    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def adjust_lr_to(self,a:float):
        self.param_groups[0]['lr']=a
        
def generate_optimizer(base_optimizer,model_parameters,SAM_flag,**kwargs):
    if SAM_flag==False:
        optimizer=base_optimizer(model_parameters,**kwargs)
    else:
        optimizer = SAM(model_parameters,base_optimizer,**kwargs)     
    return optimizer


def avg_dict(dict_list):   # 可以实现单层和双层字典的平均，输入字典和输出字典的元素为str的整数。
    out_dict=dict_list[0].copy()
    
    for dict_key in out_dict.keys():
        if type(out_dict[dict_key])==dict:
            for dict_key_key in out_dict[dict_key].keys():
                tmp_value_list=[]
                for tmp_dict in dict_list:
                    tmp_value_list.append(int(tmp_dict[dict_key][dict_key_key]))
                out_dict[dict_key][dict_key_key]=str(int(round(np.mean(tmp_value_list)))) #str_2_int
        
        else:
            tmp_value_list=[]
            for tmp_dict in dict_list:
                tmp_value_list.append(int(tmp_dict[dict_key]))
            out_dict[dict_key]=str(int(round(np.mean(tmp_value_list)))) #str_2_int
    return out_dict




ssqi_weight=20
ksqi_weight=2
psqi_weight=20000
bassqi_weight=20000
fft_weight=5

def ssqi(ecg_signal):
    ecg_signal = ecg_signal.view(-1)  # Flatten the input tensor
    mean_signal = torch.mean(ecg_signal)
    centered_signal = ecg_signal - mean_signal
    num = torch.mean(centered_signal**3)
    std_signal = torch.std(centered_signal, unbiased=True)
    s_sqi = num / (std_signal**3)
    s_sqi_score = torch.round(s_sqi * 1000) / 1000  # Round to 3 decimal places
    return s_sqi_score

def ksqi(ecg_signal):
    ecg_signal = ecg_signal.view(-1)  # Flatten the input tensor
    mean_signal = torch.mean(ecg_signal)
    centered_signal = ecg_signal - mean_signal
    num = torch.mean(centered_signal**4)
    std_signal = torch.std(centered_signal, unbiased=True)
    k_sqi = num / (std_signal**4)
    k_sqi_fischer = k_sqi - 5.0
    k_sqi_score = torch.round(k_sqi_fischer * 1000) / 1000  # Round to 3 decimal places
    return k_sqi_score

def psqi(ecg_signal, sampling_frequency=100) -> float:
    n = len(ecg_signal)
    t = 1 / sampling_frequency

    yf = torch.fft.fft(ecg_signal)
    xf = torch.linspace(0.0, 1.0 / (2.0 * t), n // 2, device=ecg_signal.device)

    pds_num = torch.where((xf >= 5) & (xf <= 15), torch.abs(yf[:n // 2]), torch.tensor(0., device=ecg_signal.device))
    pds_denom = torch.where((xf >= 5) & (xf <= 40), torch.abs(yf[:n // 2]), torch.tensor(0., device=ecg_signal.device))

    p_sqi_score = torch.round(torch.sum(pds_num) / torch.sum(pds_denom) * 1000) / 1000

    return p_sqi_score

def bassqi(ecg_signal, sampling_frequency=100) -> float:
    n = len(ecg_signal)
    t = 1 / sampling_frequency

    yf = torch.fft.fft(ecg_signal)
    xf = torch.linspace(0.0, 1.0 / (2.0 * t), n // 2, device=ecg_signal.device)

    pds_num = torch.where((xf >= 0) & (xf <= 1), torch.abs(yf[:n // 2]), torch.tensor(0., device=ecg_signal.device))
    pds_denom = torch.where((xf >= 0) & (xf <= 40), torch.abs(yf[:n // 2]), torch.tensor(0., device=ecg_signal.device))

    bas_sqi_score = torch.round((1 - (torch.sum(pds_num) / torch.sum(pds_denom))) * 1000) / 1000

    return bas_sqi_score



def csqi(ecg_signal, sampling_frequency=100) -> float:
    with np.errstate(invalid='raise'):

        try:

            rri_list = bsp_ecg.hamilton_segmenter(
                    signal=np.array(ecg_signal),
                    sampling_rate=sampling_frequency)[0]

            c_sqi_score = float(np.round(
                np.std(rri_list, ddof=1) / np.mean(rri_list),
                3))

        except Exception:
            c_sqi_score = 0

        return c_sqi_score

def moving_average_torch(signal, window_size):
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.to(signal.device).to(signal.dtype)  # Cast kernel tensor to the same type as the input tensor
    return F.conv1d(signal.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=window_size-1).squeeze(0).squeeze(0)

def smoothed_fft_spectrum(ecg_signal, window_size=20, downsample_rate=10):
    ecg_signal = ecg_signal.view(-1)  # Flatten the input tensor
    ecg_signal = ecg_signal.to(torch.float64)  # Convert to float64 to maintain precision

    psd = torch.abs(torch.fft.rfft(ecg_signal))**2 / 1000
    psd[0] = (psd[0] + psd[-1]) / 2
    psd = psd[:500]

    psd_out = moving_average_torch(psd, window_size=window_size)
    psd_out = psd_out[::downsample_rate]
    
#     plt.plot(psd_out.clone().detach().cpu().numpy())
#     plt.show()

    return psd_out


def loss_ssqi(ina,inb):
    ssqia=ssqi(ina)
    ssqib=ssqi(inb)
    return (ssqia-ssqib)**2

def loss_ksqi(ina,inb):
    ksqia=ksqi(ina)
    ksqib=ksqi(inb)
    return (ksqia-ksqib)**2

def loss_psqi(ina,inb):
    psqia=psqi(ina)
    psqib=psqi(inb)
    return (psqia-psqib)**2

def loss_bassqi(ina,inb):
    bassqia=bassqi(ina)
    bassqib=bassqi(inb)
    return (bassqia-bassqib)**2


def loss_fft(ina,inb):
    ffta=smoothed_fft_spectrum(ina)
    fftb=smoothed_fft_spectrum(inb)
    fft_loss=torch.mean((ffta - fftb)**2)
    return fft_loss


def choose_pairs(data, n):
    data = data.squeeze(2)  # Remove the singleton dimension
    assert data.size(0) >= n, "The number of samples in 'data' should be greater than or equal to 'n'"

    # Get all possible pairs of indices
    index_pairs = torch.combinations(torch.arange(n), r=2)
#     index_pairs = torch.combinations(torch.arange(n, indexing='ij'), r=2)


    # Randomly choose (n-1) pairs of indices without repetition
    selected_indices = torch.randperm(index_pairs.size(0))[:n-1]
    selected_pairs = index_pairs[selected_indices]

    # Get the corresponding data from the tensor
    selected_data = torch.stack([data[selected_pairs[:, 0]], data[selected_pairs[:, 1]]], dim=1)

    return selected_data


def calculate_in_class_loss_fn(output,train_total_class,batch_class,ssqi_loss=False,ksqi_loss=False,psqi_loss=False,bassqi_loss=False,fft_loss=False):
    
    batch_class=batch_class[:,:3]
    
    inclass_loss=0
    
    weight_array = 1 / (train_total_class * train_total_class.sum())
    weight_array /= weight_array.sum()
    
    normal_mask = (batch_class == torch.tensor([1., 0., 0.])).all(dim=1)
    output_normal = output[normal_mask]
        
    AF_mask = (batch_class == torch.tensor([0., 1., 0.])).all(dim=1)
    output_AF = output[AF_mask]
    
    others_mask = (batch_class == torch.tensor([0., 0., 1.])).all(dim=1)
    output_others = output[others_mask]
    
    # ssqi loss
    if ssqi_loss==True:
        
        ssqi_loss=0
        ssqi_num=0
#         print(output_normal.shape)
        if output_normal.shape[0]>=2:
            for pairs in choose_pairs(output_normal,output_normal.shape[0]):
                ssqi_loss+=loss_ssqi(pairs[0],pairs[1])*weight_array[0]
                ssqi_num+=1
        if output_AF.shape[0]>=2:
            for pairs in choose_pairs(output_AF,output_AF.shape[0]):
                ssqi_loss+=loss_ssqi(pairs[0],pairs[1])*weight_array[1]
                ssqi_num+=1
        if output_others.shape[0]>=2:
            for pairs in choose_pairs(output_others,output_others.shape[0]):
                ssqi_loss+=loss_ssqi(pairs[0],pairs[1])*weight_array[2]
                ssqi_num+=1       
        ssqi_loss/=ssqi_num
        ssqi_loss*=ssqi_weight
        inclass_loss+=ssqi_loss

        
    # ksqi loss
    if ksqi_loss==True:
        
        ksqi_loss=0
        ksqi_num=0
#         print(output_normal.shape)
        if output_normal.shape[0]>=2:
            for pairs in choose_pairs(output_normal,output_normal.shape[0]):
                ksqi_loss+=loss_ksqi(pairs[0],pairs[1])*weight_array[0]
                ksqi_num+=1
        if output_AF.shape[0]>=2:
            for pairs in choose_pairs(output_AF,output_AF.shape[0]):
                ksqi_loss+=loss_ksqi(pairs[0],pairs[1])*weight_array[1]
                ksqi_num+=1
        if output_others.shape[0]>=2:
            for pairs in choose_pairs(output_others,output_others.shape[0]):
                ksqi_loss+=loss_ksqi(pairs[0],pairs[1])*weight_array[2]
                ksqi_num+=1       
        ksqi_loss/=ksqi_num
        ksqi_loss*=ksqi_weight
        inclass_loss+=ksqi_loss
        
        
    # psqi loss
    if psqi_loss==True:
        
        psqi_loss=0
        psqi_num=0
#         print(output_normal.shape)
        if output_normal.shape[0]>=2:
            for pairs in choose_pairs(output_normal,output_normal.shape[0]):
                psqi_loss+=loss_psqi(pairs[0],pairs[1])*weight_array[0]
                psqi_num+=1
        if output_AF.shape[0]>=2:
            for pairs in choose_pairs(output_AF,output_AF.shape[0]):
                psqi_loss+=loss_psqi(pairs[0],pairs[1])*weight_array[1]
                psqi_num+=1
        if output_others.shape[0]>=2:
            for pairs in choose_pairs(output_others,output_others.shape[0]):
                psqi_loss+=loss_psqi(pairs[0],pairs[1])*weight_array[2]
                psqi_num+=1       
        psqi_loss/=psqi_num
        psqi_loss*=psqi_weight
        inclass_loss+=psqi_loss
            

        
    # bassqi loss
    if bassqi_loss==True:
        
        bassqi_loss=0
        bassqi_num=0
#         print(output_normal.shape)
        if output_normal.shape[0]>=2:
            for pairs in choose_pairs(output_normal,output_normal.shape[0]):
                bassqi_loss+=loss_bassqi(pairs[0],pairs[1])*weight_array[0]
                bassqi_num+=1
        if output_AF.shape[0]>=2:
            for pairs in choose_pairs(output_AF,output_AF.shape[0]):
                bassqi_loss+=loss_bassqi(pairs[0],pairs[1])*weight_array[1]
                bassqi_num+=1
        if output_others.shape[0]>=2:
            for pairs in choose_pairs(output_others,output_others.shape[0]):
                bassqi_loss+=loss_bassqi(pairs[0],pairs[1])*weight_array[2]
                bassqi_num+=1       
        bassqi_loss/=bassqi_num
        bassqi_loss*=bassqi_weight
        inclass_loss+=bassqi_loss
        


    # fft loss
    if fft_loss==True:
        
        fft_loss=0
        fft_num=0
#         print(output_normal.shape)
        if output_normal.shape[0]>=2:
            for pairs in choose_pairs(output_normal,output_normal.shape[0]):
                fft_loss+=loss_fft(pairs[0],pairs[1])*weight_array[0]
                fft_num+=1
        if output_AF.shape[0]>=2:
            for pairs in choose_pairs(output_AF,output_AF.shape[0]):
                fft_loss+=loss_fft(pairs[0],pairs[1])*weight_array[1]
                fft_num+=1
        if output_others.shape[0]>=2:
            for pairs in choose_pairs(output_others,output_others.shape[0]):
                fft_loss+=loss_fft(pairs[0],pairs[1])*weight_array[2]
                fft_num+=1       
        fft_loss/=fft_num
        fft_loss*=fft_weight
        inclass_loss+=fft_loss

            
    
    return inclass_loss


def find_pretrained_model_name(train_data_list,add_recons):
    
    train_data_list.sort()
    
    pth_list=os.listdir('ckpts')
    
    list1=[]
    
    for tmp in pth_list:
        if tmp.endswith('.pth'):
            if tmp.split('_')[2]==add_recons:
                list1.append(tmp)
    
    for tmp in list1:
        ttmp=tmp.split('_')[-1].split('.')[0]
        ttmp=[ii for ii in ttmp]
        ttmp.sort()
        if train_data_list==ttmp:
            name = tmp
            print('fine ckpt file: ckpts/'+name)
    
    return 'ckpts/'+name




def train_one_epoch_img(epoch,warm_epoch, total_epoch, model,max_lr,min_lr,loss_fn,optimizer, base_optimizer, train_loader, device,DG_method=None,MMD_iter=4,MMD_gamma=1., irm_lambda=1.0, irm_penalty_anneal_iters=100, ali_weight=0.5, SAM_operate=False, recons_model=None,add_recons=None,DGGR_smooth_eps=0.,weights=0,theta=0,GRL_loss_beta=0.05,GRL_condition_beta=0.05,class_num=3,distill=False, domain_class=0):
    # irm_lambda=1e2, irm_penalty_anneal_iters=10000000,效果很好  # irm_lambda=1, 后者=100，效果差一些。唯一的区别就是达到iters时重置了优化器。
    
    model.train()
    running_loss=None
    for step, (in_data_, targets_) in (enumerate(tqdm(train_loader))):
        current_iter=step+epoch*len(train_loader)
        warm_iter=warm_epoch*len(train_loader)
        total_iter=total_epoch*len(train_loader)
        adjust_learning_rate(optimizer,current_iter,warm_iter,total_iter,max_lr,min_lr)
        
        if epoch<=warm_epoch:
            for param in model.parameters():  
                param.requires_grad = False  
            for param in model.fc.parameters():
                param.requires_grad = True  
        else:
            for param in model.parameters():  
                param.requires_grad = True
                
        optimizer.zero_grad()

        chafen_array=torch.zeros(in_data_.shape)
        chafen_array[:,:-1,:]=in_data_[:,1:,:]-in_data_[:,:-1,:]
        chafen_2_array=torch.zeros(in_data_.shape)
        chafen_2_array[:,:-1,:]=chafen_array[:,1:,:]-chafen_array[:,:-1,:]


        image=torch.zeros((in_data_.shape[0],224,224,3))


        for i2d in range(5):
            for j2d in range(224):
                if 224*i2d+j2d <1000:
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,0]=in_data_[:,224*i2d+j2d,:]
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,1]=chafen_array[:,224*i2d+j2d,:]
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,2]=chafen_2_array[:,224*i2d+j2d,:]
        for i2d in range(3):
            image[:,:,:,i2d]=(image[:,:,:,i2d]-torch.min(image[:,:,:,i2d]))
            image[:,:,:,i2d]=image[:,:,:,i2d]/(torch.max(image[:,:,:,i2d]))
        
        image = image.permute(0, 3, 1, 2)  


        diag_preds_=model(image.float().to(device))
        diag_preds_=diag_preds_.float()
        targets_for_diag_loss=torch.max(targets_[:,:class_num],1).indices.long().to(device)  # 这里的class_num是指类别数量
        loss=loss_fn(diag_preds_,targets_for_diag_loss)

        loss.backward()
        optimizer.step()
        if running_loss is None:
            running_loss = loss.item() # 只输出重构损失
        else:
            running_loss = running_loss * .9 + loss.item() * .1   
    print('epoch=',epoch,)
    print("train_running_loss=",running_loss)
    return running_loss





def val_one_epoch_img(epoch,warm_epoch, total_epoch, model,loss_fn, test_loader, device,DG_method=None,class_F1=False, recons_model=None,add_recons=None,class_num=3,distill=False):
    
    model.eval() 
    running_loss = None 
    test_pred=0
    running_loss = []
    test_tar=0
    for step, (in_data_, targets_) in (enumerate(tqdm(test_loader))):

        chafen_array=torch.zeros(in_data_.shape)
        chafen_array[:,:-1,:]=in_data_[:,1:,:]-in_data_[:,:-1,:]
        chafen_2_array=torch.zeros(in_data_.shape)
        chafen_2_array[:,:-1,:]=chafen_array[:,1:,:]-chafen_array[:,:-1,:]


        image=torch.zeros((in_data_.shape[0],224,224,3))


        for i2d in range(5):
            for j2d in range(224):
                if 224*i2d+j2d <1000:
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,0]=in_data_[:,224*i2d+j2d,:]
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,1]=chafen_array[:,224*i2d+j2d,:]
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,2]=chafen_2_array[:,224*i2d+j2d,:]
        for i2d in range(3):
            image[:,:,:,i2d]=(image[:,:,:,i2d]-torch.min(image[:,:,:,i2d]))
            image[:,:,:,i2d]=image[:,:,:,i2d]/(torch.max(image[:,:,:,i2d]))

        image = image.permute(0, 3, 1, 2)  


        preds_=model(image.float().to(device))
        preds_=preds_.float()
        targets_for_loss=torch.max(targets_[:,:class_num],1).indices.long().to(device)
        loss=loss_fn(preds_,targets_for_loss)    
        running_loss.append(loss.item())
        mean_loss=np.mean(running_loss)
        if type(test_pred)==int:
            test_pred=preds_.clone().detach().cpu().numpy()
        else:
            test_pred=np.concatenate((test_pred,
                                preds_.clone().detach().cpu().numpy(),
                                ),axis=0)
        if type(test_tar)==int:
            test_tar=targets_.clone().detach().cpu().numpy()
        else:
            test_tar=np.concatenate((test_tar,
                                targets_.clone().detach().cpu().numpy(),
                                ),axis=0)
    print('epoch=',epoch,)
    print("val_loss=",mean_loss)
    print("________________________________")
                
    mean_macro_auc=sklearn.metrics.roc_auc_score(test_tar, test_pred, average='macro')
    F1_score=get_macro_f1(test_tar,test_pred)
    precision=get_macro_precision(test_tar,test_pred)
    recall=get_macro_recall(test_tar,test_pred)
    acc=get_acc(test_tar,test_pred)
    if class_F1==True:
        F1_normal_score,F1_AF_score,F1_other_score=get_class_f1(test_tar, test_pred)
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*F1_normal_score))),str(int(round(1000*F1_AF_score))),str(int(round(1000*F1_other_score))),str(int(round(1000*acc)))
    else:    
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*acc)))
    
    
def test_one_epoch_img(epoch,warm_epoch, total_epoch, model,loss_fn, test_loader, device,DG_method=None,class_F1=False, recons_model=None,add_recons=None,class_num=3,distill=False):
    
    model.etest() 
    running_loss = None 
    test_pred=0
    running_loss = []
    test_tar=0
    for step, (in_data_, targets_) in (enumerate(tqdm(test_loader))):

        chafen_array=torch.zeros(in_data_.shape)
        chafen_array[:,:-1,:]=in_data_[:,1:,:]-in_data_[:,:-1,:]
        chafen_2_array=torch.zeros(in_data_.shape)
        chafen_2_array[:,:-1,:]=chafen_array[:,1:,:]-chafen_array[:,:-1,:]


        image=torch.zeros((in_data_.shape[0],224,224,3))

        
        for i2d in range(5):
            for j2d in range(224):
                if 224*i2d+j2d <1000:
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,0]=in_data_[:,224*i2d+j2d,:]
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,1]=chafen_array[:,224*i2d+j2d,:]
                    image[:,40*i2d+4*i2d+4:40*(i2d+1)+4*i2d+4,j2d,2]=chafen_2_array[:,224*i2d+j2d,:]
        for i2d in range(3):
            image[:,:,:,i2d]=(image[:,:,:,i2d]-torch.min(image[:,:,:,i2d]))
            image[:,:,:,i2d]=image[:,:,:,i2d]/(torch.max(image[:,:,:,i2d]))

        image = image.permute(0, 3, 1, 2)  


        preds_=model(image.float().to(device))
        preds_=preds_.float()
        targets_for_loss=torch.max(targets_[:,:class_num],1).indices.long().to(device)
        loss=loss_fn(preds_,targets_for_loss)    
        running_loss.append(loss.item())
        mean_loss=np.mean(running_loss)
        if type(test_pred)==int:
            test_pred=preds_.clone().detach().cpu().numpy()
        else:
            test_pred=np.concatenate((test_pred,
                                preds_.clone().detach().cpu().numpy(),
                                ),axis=0)
        if type(test_tar)==int:
            test_tar=targets_.clone().detach().cpu().numpy()
        else:
            test_tar=np.concatenate((test_tar,
                                targets_.clone().detach().cpu().numpy(),
                                ),axis=0)
    print('epoch=',epoch,)
    print("test_loss=",mean_loss)
    print("________________________________")
                
    mean_macro_auc=sklearn.metrics.roc_auc_score(test_tar, test_pred, average='macro')
    F1_score=get_macro_f1(test_tar,test_pred)
    precision=get_macro_precision(test_tar,test_pred)
    recall=get_macro_recall(test_tar,test_pred)
    acc=get_acc(test_tar,test_pred)
    if class_F1==True:
        F1_normal_score,F1_AF_score,F1_other_score=get_class_f1(test_tar, test_pred)
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*F1_normal_score))),str(int(round(1000*F1_AF_score))),str(int(round(1000*F1_other_score))),str(int(round(1000*acc)))
    else:    
        return str(int(round(1000*mean_macro_auc))),str(int(round(1000*precision))),str(int(round(1000*recall))),str(int(round(1000*F1_score))),str(int(round(1000*acc)))


