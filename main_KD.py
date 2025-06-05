import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

def main():
    global model
############ Train ##############
    args.test_only = False
    args.save_results = False
    checkpoint = utility.checkpoint(args)
    loader = data.Data(args)
    args.pre_train = './experiment/alpha6/model/model_best.pt'
    _modelT = model.Model(args, checkpoint)
    args.pre_train = ''
    _modelS = model.ModelS(args, checkpoint)
    _loss = loss.Loss(args, checkpoint)
    _lossv = loss.Loss(args, checkpoint, m='validation')
    t = Trainer(args, loader, _modelT, _modelS, _loss, _lossv, checkpoint)
    loss_s_total = []
    loss_pmd_total = []
    b_b_Con_loss_total = []
    lossTotal_total = []
    while not t.terminate():
        loss_s, loss_pmd, b_b_Con_loss, lossTotal = t.train()
        loss_s_total.append(loss_s)
        loss_pmd_total.append(loss_pmd)
        b_b_Con_loss_total.append(b_b_Con_loss)
        lossTotal_total.append(lossTotal)
        t.test()
    np.save(r'./experiment/alpha6/model/10-1200/loss_s_total.npy', loss_s_total)
    np.save(r'./experiment/alpha6/model/10-1200/loss_pmd_total.npy', loss_pmd_total)
    np.save(r'./experiment/alpha6/model/10-1200/b_b_Con_loss_total.npy', b_b_Con_loss_total)
    np.save(r'./experiment/alpha6/model/10-1200/lossTotal_total.npy', lossTotal_total)

    checkpoint.done()

# ############ Test synthetic data ##############
    print('test synthetic data')
    args.test_only = True
    args.save_results = True
    args.data_range = '1-1200/1451-1460'
    checkpoint = utility.checkpoint(args)
    loader = data.Data(args)
    args.pre_train = './experiment/alpha6/model/model_best.pt'
    _modelT = model.Model(args, checkpoint)
    args.pre_train = './experiment/alpha6/model/modelS_best.pt'
    _modelS = model.ModelS(args, checkpoint)

    t = Trainer(args, loader, _modelT, _modelS, ckp=checkpoint)
    t.test()

    checkpoint.done()

########## Test2 ##############
    print("test field data")
    args.test_only = True
    args.save_dir_suffix = 'field'
    args.data_range = '1-1200/1451-1453'
    # args.data_range = '1-1200/1451-1451'
    args.dir_lr = './data/field/'
    # args.dir_lr = './data/log/'
    args.apply_field_data = True
    loader = data.Data(args)
    checkpoint = utility.checkpoint(args)
    args.pre_train = './experiment/alpha6/model/model_best.pt'
    _modelT = model.Model(args, checkpoint)
    args.pre_train = './experiment/alpha6/model/modelS_best.pt'
    _modelS = model.ModelS(args, checkpoint)

    t = Trainer(args, loader, _modelT, _modelS, ckp=checkpoint)
    t.test()

    checkpoint.done()

if __name__ == '__main__':
    main()
