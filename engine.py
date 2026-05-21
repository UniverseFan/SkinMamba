import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import time

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        if np.any(gts == 1):
            y_pre = np.where(preds>=config.threshold, 1, 0)
            y_true = np.where(gts>=0.5, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
            print(log_info)
            logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0
    total_f1_or_dsc = 0
    total_miou = 0
    total_precision = 0
    num_iterations = 0
    # with torch.no_grad():
    #     for i, data in enumerate(tqdm(test_loader)):
    #         img, msk = data
    #         img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
    #         start_time = time.time()
    #         out = model(img)
    #         end_time = time.time()
    #         e_time = end_time-start_time
    #         logger.info(f"Inference time for iteration {i}: {e_time:.6f} seconds")
    #         loss = criterion(out, msk)

    #         loss_list.append(loss.item())
    #         msk = msk.squeeze(1).cpu().detach().numpy()
    #         msk = np.array(msk).reshape(-1)
    #         gts.append(msk)
    #         if type(out) is tuple:
    #             out = out[0]
    #         out = out.squeeze(1).cpu().detach().numpy()
    #         out = np.array(out).reshape(-1)
    #         y_pre = np.where(out>=config.threshold, 1, 0)
    #         y_true = np.where(msk>=0.5, 1, 0)

    #         confusion = confusion_matrix(y_true, y_pre)
    #         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 
    #         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    #         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    #         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    #         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    #         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    #         precision = float(TP)/float(TP+FP) if float(TP+FP)!=0 else 0
    #         preds.append(out) 
    #         if i % config.save_interval == 0:
    #             save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
    #             save_imgs(img, msk, out, i, '/skin/results/vmunet_skin_Wednesday_22_May_2024_23h_43m_59s' + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
    #         total_accuracy += accuracy
    #         total_sensitivity += sensitivity
    #         total_specificity += specificity
    #         total_f1_or_dsc += f1_or_dsc
    #         total_miou += miou
    #         total_precision += precision
    #         num_iterations += 1
        
    # average_accuracy = total_accuracy / num_iterations if num_iterations > 0 else 0
    # average_sensitivity = total_sensitivity / num_iterations if num_iterations > 0 else 0
    # average_specificity = total_specificity / num_iterations if num_iterations > 0 else 0
    # average_f1_or_dsc = total_f1_or_dsc / num_iterations if num_iterations > 0 else 0
    # average_miou = total_miou / num_iterations if num_iterations > 0 else 0
    # average_precision = total_precision / num_iterations if num_iterations > 0 else 0

    # print(f"Average Accuracy: {average_accuracy:.6f}")
    # print(f"Average Sensitivity: {average_sensitivity:.6f}")
    # print(f"Average Specificity: {average_specificity:.6f}")
    # print(f"Average F1 or DSC: {average_f1_or_dsc:.6f}")
    # print(f"Average mIoU: {average_miou:.6f}")
    # print(f"Average Precision: {average_precision:.6f}")

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            start_time = time.time()
            out = model(img)
            end_time = time.time()
            e_time = end_time-start_time
            logger.info(f"Inference time for iteration {i}: {e_time:.6f} seconds")
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
                # save_imgs(img, msk, out, i, '/skin/results/vmunet_skin_Wednesday_22_May_2024_23h_43m_59s' + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        precison = float(TP)/float(TP+FP) if float(TP+FP)!=0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion} , precision:{precison}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)