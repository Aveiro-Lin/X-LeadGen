from torch.utils.data import Dataset,DataLoader
import numpy as np
from imblearn.over_sampling import SMOTE
from utils import *
import torch.nn as nn
from models.DG_models import generate_model
import time
from math import ceil,floor

def onehot_2_one(innp):
    outnp=np.argmax(innp,axis=1)
    return outnp

def one_2_onehot(innp):
    outnp=np.zeros((innp.shape[0],np.max(innp)+1))
    for tmponehot in range(innp.shape[0]):
        outnp[tmponehot,innp[tmponehot]]=1
    return outnp
        
class GetLoader(Dataset):
    def __init__(self, dataname,train_val_test_split,mode='train',smote=False,outliner_remover=False,remover_device=None,class_num=3):
        data_list=[]
        label_list=[]
        self.total_class=[]
#         self.total_datanum=0
        
        dataset_num=len(dataname)
        
        if mode=='train':
            dataset_index=-1
            for i in dataname:
                dataset_index+=1
                print('train data '+i)
                tmp_data=np.load('/home/data/AF_DG/'+i[0]+'/'+i+'_data.npy',allow_pickle=True)
                tmp_data=tmp_data[:int(tmp_data.shape[0]*train_val_test_split[0]),:,:]
                tmp_label=np.load('/home/data/AF_DG/'+i[0]+'/'+i+'_label.npy',allow_pickle=True)
                tmp_label=tmp_label[:int(tmp_label.shape[0]*train_val_test_split[0]),:]
                
                
                if smote==True:
                    
                    smo = SMOTE(n_jobs=-1,k_neighbors=3) 
                    
                    tmp_label=onehot_2_one(tmp_label)
                    
                    print('-')
                    print(tmp_data.shape)
                    print(tmp_label.shape)
                    
                    tmp_new_data, tmp_new_label = smo.fit_resample(tmp_data[:,:,0], tmp_label) 
                    
                    tmp_data=np.expand_dims(tmp_new_data,axis=2)

                    tmp_label=one_2_onehot(tmp_new_label)
                    
                    print('smoted train data size : '+str(int(tmp_data.shape[0])))
                    print('smoted train label class : ',np.sum(tmp_label,axis=0))
                    
                tmp_label=np.pad(tmp_label, ((0,0),(0,dataset_num)))
                
                tmp_label[:,class_num+dataset_index]=1
                
                print('train data size : '+str(int(tmp_data.shape[0])))
                print('train label class : ',np.sum(tmp_label,axis=0))
                
                data_list.append(tmp_data)
                label_list.append(tmp_label)
            
            self.data=np.concatenate(tuple(data_list),axis=0)
            self.label=np.concatenate(tuple(label_list),axis=0)

            print('total train data shape = ',self.data.shape)
            print('total train label shape = ',self.label.shape)
            print('total train data class = ',np.sum(self.label,axis=0))    
            
            
#             if outliner_remover==True:
                
#                 # 通过简单的网络训练或什么，提取一些特征，并保存特征矩阵。
                
#                 # 这里训练几个epoch，使用numpy数组
                
#                 remover_bs=128
#                 remover_epoch=10
#                 remover_lr=1e-3
#                 remover_model=generate_model(base_model='resnet18',input_channels=1, num_classes=3,DG_method='remover').to(remover_device)
#                 remover_opt=torch.optim.Adam(remover_model.parameters(),lr=remover_lr,weight_decay=1e-4)
#                 remover_loss_fn=nn.CrossEntropyLoss()
                
                
#                 # train
#                 remover_model.train() 
                
#                 for tmp_epoch in range(remover_epoch):
#                     loss_list=[]
#                     for tmp_index in tqdm(list(range(0,self.data.shape[0],remover_bs))):
#                         tmp_batch_data=torch.tensor(self.data[tmp_index:min(tmp_index+remover_bs,self.data.shape[0])]).float().to(remover_device).detach()
#                         tmp_batch_label=torch.tensor(self.label[tmp_index:min(tmp_index+remover_bs,self.label.shape[0])]).to(remover_device).detach()
#                         remover_opt.zero_grad()
# #                         print(tmp_batch_data.shape)
# #                         print(type(tmp_batch_data))
# #                         print(tmp_batch_data.device)
#                         _,remover_preds_=remover_model(tmp_batch_data)
#                         remover_preds_=remover_preds_.float()
#                         remover_targets=torch.max(tmp_batch_label[:,:3],1).indices.long().to(remover_device)
#                         remover_loss=remover_loss_fn(remover_preds_,remover_targets)
#                         loss_list.append(remover_loss.item())
#                         remover_loss.backward()
#                         remover_opt.step()
#                     print('remover_loss=',np.mean(loss_list))
                
#                 # feat mat
#                 print('generating feature')
                
                
                
#                 remover_model.eval() 
#                 remover_feat_all=np.zeros((self.data.shape[0],64))
#                 for tmp_index in tqdm(list(range(0,self.data.shape[0],remover_bs))):
# #                     tmp_time=time.time()
#                     tmp_batch_data=torch.tensor(self.data[tmp_index:min(tmp_index+remover_bs,self.data.shape[0])]).float().to(remover_device)
# #                     print('1:    ',time.time()-tmp_time)
# #                     tmp_time=time.time()
#                     remover_feat,_=remover_model(tmp_batch_data)
# #                     print('2:    ',time.time()-tmp_time)
# #                     tmp_time=time.time()
#                     remover_feat=remover_feat.float()
# #                     print('3:    ',time.time()-tmp_time)
# #                     tmp_time=time.time()
#                     remover_feat_all[tmp_index:min(tmp_index+remover_bs,self.data.shape[0])]=remover_feat.clone().detach().cpu().numpy()
# #                     print('4:    ',time.time()-tmp_time)
# #                     tmp_time=time.time()
#                 print(remover_feat_all[0])
#                 print(remover_feat_all[-1])
                
#                 # select index
                
#                 order_array=np.sort(remover_feat_all,axis=0)
#                 small_th=order_array[ceil(order_array.shape[0]*0.001):ceil(order_array.shape[0]*0.001)+1,:]
#                 big_th=order_array[floor(order_array.shape[0]*0.999):floor(order_array.shape[0]*0.999)+1,:]
#                 small_th_mat=np.repeat (small_th,remover_feat_all.shape[0],0) 
#                 big_th_mat=np.repeat (big_th,remover_feat_all.shape[0],0) 
#                 all_index_for_select=np.sum(remover_feat_all<small_th_mat,axis=1)+np.sum(remover_feat_all>big_th_mat,axis=1)
#                 selected_data=self.data[all_index_for_select==0].copy()
#                 selected_label=self.label[all_index_for_select==0].copy()
#                 self.data=selected_data
#                 self.label=selected_label
                
#                 print('total selected train data shape = ',self.data.shape)
#                 print('total selected train label shape = ',self.label.shape)
#                 print('total selected train data class = ',np.sum(self.label,axis=0))    
                
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
#                 print('代码还没有写完')
                
#                 break
                
                # 
                
                
                
#                 特征提取的方法：
#                 ①直接使用RR间期等先验特征，如使用https://github.com/Seb-Good/ecg-features；NeuroKit2；heartpy
#                 ②训练一个简单的网络，最后一层特征进行统计
                
#                 直接引一个小resnet，开始训练就行了
                
#                 每一维度特征删除1%的离群值
                
#                 tmp_folder用于过渡期间的特征储存，根据这个数组筛选index。
#                 也可以没有tmp_folder。
                
                
                
#                 self.data,self.label=Remove_outliner(in_data=self.data,in_label=self.label,tmp_folder='./tmp_folder')
            
                
            
            self.total_class=np.sum(self.label,axis=0)[:class_num]
#             self.total_datanum=self.label.shape[0]
            
        elif mode=='val':
            for i in dataname:
                print('val data '+i)
                tmp_data=np.load('/home/data/AF_DG/'+i[0]+'/'+i+'_data.npy',allow_pickle=True)
                tmp_data=tmp_data[int(tmp_data.shape[0]*train_val_test_split[0]):int(tmp_data.shape[0]*(train_val_test_split[0]+train_val_test_split[1])),:,:]
                tmp_label=np.load('/home/data/AF_DG/'+i[0]+'/'+i+'_label.npy',allow_pickle=True)
                tmp_label=tmp_label[int(tmp_label.shape[0]*train_val_test_split[0]):int(tmp_label.shape[0]*(train_val_test_split[0]+train_val_test_split[1])),:]
                print('val data size : '+str(int(tmp_data.shape[0])))
                print('val label class : ',np.sum(tmp_label,axis=0))
                
                data_list.append(tmp_data)
                label_list.append(tmp_label)
            
            self.data=np.concatenate(tuple(data_list),axis=0)
            self.label=np.concatenate(tuple(label_list),axis=0)

            print('total val data shape = ',self.data.shape)
            print('total val label shape = ',self.label.shape)
            print('total val data class = ',np.sum(self.label,axis=0)) 
            
            self.total_class=np.sum(self.label,axis=0)[:class_num]
            
        elif mode=='test':
            for i in dataname:
                print('test data '+i)
                tmp_data=np.load('/home/data/AF_DG/'+i[0]+'/'+i+'_data.npy',allow_pickle=True)
                tmp_data=tmp_data[int(tmp_data.shape[0]*(1-train_val_test_split[2])):,:,:]
                tmp_label=np.load('/home/data/AF_DG/'+i[0]+'/'+i+'_label.npy',allow_pickle=True)
                tmp_label=tmp_label[int(tmp_label.shape[0]*(1-train_val_test_split[2])):,:]
                print('test data size : '+str(int(tmp_data.shape[0])))
                print('test label class : ',np.sum(tmp_label,axis=0))
                
                data_list.append(tmp_data)
                label_list.append(tmp_label)
            
            self.data=np.concatenate(tuple(data_list),axis=0)
            self.label=np.concatenate(tuple(label_list),axis=0)

            print('total test data shape = ',self.data.shape)
            print('total test label shape = ',self.label.shape)
            print('total test data class = ',np.sum(self.label,axis=0)) 
            
            self.total_class=np.sum(self.label,axis=0)[:class_num]
        else:
            print('dataloader mode error')
        
#         if class_num==2:
#             self.label_new=np.zeros((self.label.shape[0],self.label.shape[1]-1))
#             self.label_new[:,0]=self.label[:,1].copy()  # 原数据集第二个数字为1代表AF，新数据集第一个数字为1代表AF
#             self.label_new[:,1]=1-self.label_new[:,0]
#             try:
#                 self.label_new[:,2:]=self.label[:,3:].copy()
#             except:
#                 pass
#             del(self.label)
#             self.label=self.label_new
            
        
    def __getitem__(self, index):
        data=self.data[index]
        labels=self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)
    
    
    
class GetLoadermaple(Dataset):
    def __init__(self, dataname,train_val_test_split,mode='train',smote=False):
        data_list=[]
        label_list=[]
        self.total_class=[]
        
        dataset_num=len(dataname)
        
        if mode=='train':
            dataset_index=-1
            for i in dataname:
                dataset_index+=1
                print('train data '+i)
                tmp_data=np.load('/home/data/AF_DG/'+i+'/'+i+'_data.npy',allow_pickle=True)
                tmp_data=tmp_data[:int(tmp_data.shape[0]*train_val_test_split[0]),:,:]
                tmp_label=np.load('/home/data/AF_DG/'+i+'/'+i+'_label.npy',allow_pickle=True)
                tmp_label=tmp_label[:int(tmp_label.shape[0]*train_val_test_split[0]),:]
                
                
                if smote==True:
                    
                    smo = SMOTE(n_jobs=-1,k_neighbors=3) 
                    
                    tmp_label=onehot_2_one(tmp_label)
                    
                    print('-')
                    print(tmp_data.shape)
                    print(tmp_label.shape)
                    
                    tmp_new_data, tmp_new_label = smo.fit_resample(tmp_data[:,:,0], tmp_label) 
                    
                    tmp_data=np.expand_dims(tmp_new_data,axis=2)

                    tmp_label=one_2_onehot(tmp_new_label)
                    
                    print('smoted train data size : '+str(int(tmp_data.shape[0])))
                    print('smoted train label class : ',np.sum(tmp_label,axis=0))
                    
                tmp_label=np.pad(tmp_label, ((0,0),(0,dataset_num)))
                
                tmp_label[:,3+dataset_index]=1
                
                print('train data size : '+str(int(tmp_data.shape[0])))
                print('train label class : ',np.sum(tmp_label,axis=0))
                
                data_list.append(tmp_data)
                label_list.append(tmp_label)
            
            self.data=np.concatenate(tuple(data_list),axis=0)
            self.label=np.concatenate(tuple(label_list),axis=0)

            print('total train data shape = ',self.data.shape)
            print('total train label shape = ',self.label.shape)
            print('total train data class = ',np.sum(self.label,axis=0))    
            
            self.total_class=np.sum(self.label,axis=0)[:3]
            
        else:
            print('data mode error!')
            
               
        
    def __getitem__(self, index):
        data=self.data[index]
        labels=self.label[index]
#         print(type(index))
#         print(index.shape)
        return data, labels, index
    def __len__(self):
        return len(self.data)
    
    
    
    
    
class train_loader():
    def __init__(self,dataname=[],train_val_test_split=[0.8,0.1,0.1],bs=256,num_workers=0,smote=False,maple=False,outliner_remover=False,remover_device=None,class_num=3):
        for i in dataname:
            assert (i in ['L','N','A','B','T','H','C','L_12','N_12','A_12','B_12','H_12','C_12','C_2cls','A_2cls','L_2cls','R_2cls','N_2cls','H_2cls','B_2cls','T_2cls','C_2cls_12','A_2cls_12','L_2cls_12','N_2cls_12','H_2cls_12','B_2cls_12','R_3cls','T_3cls']), 'Check your dataset name!'
        if class_num==2:
            assert ((maple==False)and(outliner_remover==False)), 'Only basic train_loader for 2 class!'
            self.getloader=GetLoader(dataname,train_val_test_split,mode='train',smote=smote,class_num=class_num)
        else:
            if ((maple==False)and(outliner_remover==True)and(smote==False)):
                self.getloader=GetLoader(dataname,train_val_test_split,mode='train',smote=False,outliner_remover=True,remover_device=remover_device)
            elif maple==False:
                self.getloader=GetLoader(dataname,train_val_test_split,mode='train',smote=smote)
            else:
                self.getloader=GetLoadermaple(dataname,train_val_test_split,mode='train',smote=smote)
        self.bs=bs
        self.num_workers=num_workers
        self.total_datanum=self.getloader.__len__()
         
    @property
    def total_class(self):
        return self.getloader.total_class
        
    def loader(self,):
        return DataLoader(self.getloader,self.bs,shuffle=True,drop_last=False,num_workers=self.num_workers)
    
class val_loader():
    def __init__(self,dataname=[],train_val_test_split=[0.8,0.1,0.1],bs=256,num_workers=0,class_num=3):
        for i in dataname:
            assert (i in ['L','N','A','B','T','H','C','L_12','N_12','A_12','B_12','H_12','C_12','C_2cls','A_2cls','L_2cls','R_2cls','N_2cls','H_2cls','B_2cls','T_2cls','C_2cls_12','A_2cls_12','L_2cls_12','N_2cls_12','H_2cls_12','B_2cls_12','R_3cls','T_3cls']), 'Check your dataset name!'
        self.getloader=GetLoader(dataname,train_val_test_split,mode='val',class_num=class_num)
        self.bs=bs
        self.num_workers=num_workers
        
    def loader(self,):
        return DataLoader(self.getloader,self.bs,shuffle=False,drop_last=False,num_workers=self.num_workers)
    
class test_loader():
    def __init__(self,dataname=[],train_val_test_split=[0.8,0.1,0.1],bs=256,num_workers=0,class_num=3):
        for i in dataname:
            assert (i in ['L','N','A','B','T','H','C','L_12','N_12','A_12','B_12','H_12','C_12','C_2cls','A_2cls','L_2cls','R_2cls','N_2cls','H_2cls','B_2cls','T_2cls','C_2cls_12','A_2cls_12','L_2cls_12','N_2cls_12','H_2cls_12','B_2cls_12','R_3cls','T_3cls']), 'Check your dataset name!'
        self.getloader=GetLoader(dataname,train_val_test_split,mode='test',class_num=class_num)
        self.bs=bs
        self.num_workers=num_workers
        
    def loader(self,):
        return DataLoader(self.getloader,self.bs,shuffle=False,drop_last=False,num_workers=self.num_workers)
    
    
    
    

    