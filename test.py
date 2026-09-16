from utils.distributed import *
import torch.multiprocessing as mp
from utils.ckpt import *
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.logging import *
import argparse
import time
from utils import config
from datasets.dataloader import loader,RefCOCODataSet
from tensorboardX import SummaryWriter
from utils.utils import *
import torch.optim as Optim
from importlib import import_module

class ModelLoader:
    def __init__(self, __C):

        self.model_use = __C.MODEL
        model_moudle_path = 'models.' + self.model_use + '.net'
        self.model_moudle = import_module(model_moudle_path)

    def Net(self, __arg1, __arg2, __arg3):
        return self.model_moudle.Net(__arg1, __arg2, __arg3)

# def validate(__C,
#              net,
#              loader,
#              writer,
#              epoch,
#              rank,
#              ix_to_token,
#              save_ids=None,
#              prefix='Val',
#              ema=None):
#     if ema is not None:
#         ema.apply_shadow()
#     net.eval()
#
#
#     batches = len(loader)
#     batch_time = AverageMeter('Time', ':6.5f')
#     data_time = AverageMeter('Data', ':6.5f')
#     losses = AverageMeter('Loss', ':.4f')
#     box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
#     meters = [batch_time, data_time, losses, box_ap]
#     meters_dict = {meter.name: meter for meter in meters}
#     progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix+': ')
#     with torch.no_grad():
#         end = time.time()
#         for ith_batch, data in enumerate(loader):
#
#             #ref_iter, image_iter, box_iter,gt_box_iter,info_iter,img_path = data
#             ref_iter, image_iter, box_iter, gt_box_iter, info_iter = data
#             ref_iter = ref_iter.cuda( non_blocking=True)
#             image_iter = image_iter.cuda( non_blocking=True)
#             box_iter = box_iter.cuda( non_blocking=True)
#
#             box= net(image_iter, ref_iter)
#             gt_box_iter=gt_box_iter.squeeze(1)
#             gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])
#             gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])
#             gt_box_iter=gt_box_iter.cpu().numpy()
#             info_iter=info_iter.cpu().numpy()
#             box=box.squeeze(1).cpu().numpy()
#             pred_box_vis=box.copy()
#
#             #predictions to gt
#             for i in range(len(gt_box_iter)):
#                 box[i]=yolobox2label(box[i],info_iter[i])
#             box_iou=batch_box_iou(torch.from_numpy(gt_box_iter),torch.from_numpy(box)).cpu().numpy()
#             box_ap.update((box_iou>0.5).astype(np.float32).mean()*100., box_iou.shape[0])
#
#             reduce_meters(meters_dict, rank, __C)
#             if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch==(len(loader)-1)) and main_process(__C,rank):
#                 progress.display(epoch, ith_batch)
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#         if main_process(__C,rank) and writer is not None:
#             writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)
#     if ema is not None:
#         ema.restore()
#
#
#     return box_ap.avg_reduce

import pandas as pd  # <--- 新增 import


def validate(__C,
             net,
             loader,
             writer,
             epoch,
             rank,
             ix_to_token,
             save_ids=None,
             prefix='Val',
             ema=None):
    if ema is not None:
        ema.apply_shadow()
    net.eval()

    batches = len(loader)
    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4f')
    box_ap = AverageMeter('BoxIoU@0.5', ':6.2f')
    meters = [batch_time, data_time, losses, box_ap]
    meters_dict = {meter.name: meter for meter in meters}
    progress = ProgressMeter(__C.VERSION, __C.EPOCHS, len(loader), meters, prefix=prefix + ': ')

    # --- 新增列表用于存储结果 ---
    # results_list = []

    with torch.no_grad():
        end = time.time()
        for ith_batch, data in enumerate(loader):

            # 这里的 unpacking 保持不变
            ref_iter, image_iter, box_iter, gt_box_iter, info_iter = data

            ref_iter = ref_iter.cuda(non_blocking=True)
            image_iter = image_iter.cuda(non_blocking=True)
            box_iter = box_iter.cuda(non_blocking=True)

            box = net(image_iter, ref_iter)

            # --- 处理 GT Box (Ground Truth) ---
            gt_box_iter = gt_box_iter.squeeze(1)
            gt_box_iter[:, 2] = (gt_box_iter[:, 0] + gt_box_iter[:, 2])  # x2 = x + w
            gt_box_iter[:, 3] = (gt_box_iter[:, 1] + gt_box_iter[:, 3])  # y2 = y + h
            gt_box_iter = gt_box_iter.cpu().numpy()

            info_iter = info_iter.cpu().numpy()
            box = box.squeeze(1).cpu().numpy()

            # --- 准备文本解码 (Text Decoding) ---
            # 将 token id 转回单词
            ref_ids = ref_iter.cpu().numpy()

            # --- 遍历当前 batch 的每一个样本 ---
            for i in range(len(gt_box_iter)):
                # 1. 获取预测框 (转换坐标)
                pred_b = yolobox2label(box[i], info_iter[i])
                box[i] = pred_b  # 更新回 box 用于计算 IoU

                # 2. 获取真实框
                gt_b = gt_box_iter[i]

                # 3. 计算 IoU 判断 Is Correct
                # 这里单独计算单个样本的 IoU
                inter_x1 = max(pred_b[0], gt_b[0])
                inter_y1 = max(pred_b[1], gt_b[1])
                inter_x2 = min(pred_b[2], gt_b[2])
                inter_y2 = min(pred_b[3], gt_b[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

                area_pred = (pred_b[2] - pred_b[0]) * (pred_b[3] - pred_b[1])
                area_gt = (gt_b[2] - gt_b[0]) * (gt_b[3] - gt_b[1])
                union_area = area_pred + area_gt - inter_area

                iou = inter_area / union_area if union_area > 0 else 0
                is_correct = "TRUE" if iou >= 0.5 else "FALSE"

                # 4. 解码文本 (Text)
                # 假设 0 是 padding，将其过滤掉
                words = [ix_to_token[idx] for idx in ref_ids[i] if idx > 0 and idx in ix_to_token]
                text_str = " ".join(words)

                # 5. 获取 Image ID
                # 假设 info_iter 中包含了 image id (通常在 info 的最后一位)
                # 如果你的 info_iter 只包含 [h, w, scale]，这里可能拿不到 ID，需要检查 dataloader
                # 这里暂且假设 info_iter 的最后一个元素或者是特定位置是 ID。
                # 如果没有 ID，可以用 ith_batch * batch_size + i 代替，或者修改 Dataloader 返回 ID
                try:
                    # 尝试取 info_iter 的最后一个值作为 ID (很多 dataset 比如 RefCORE 是这样存的)
                    image_id = int(info_iter[i][-1])
                except:
                    image_id = "N/A"

                # # 6. 存入结果列表
                # results_list.append({
                #     "Image ID": image_id,
                #     "Text": text_str,
                #     "True BBox": str(list(gt_b)),  # 存为字符串列表
                #     "Predicted BBox": str(list(pred_b)),  # 存为字符串列表
                #     "Is Correct": is_correct
                # })

            # --- 原有的 IoU 计算 (保持不变用于打印日志) ---
            box_iou = batch_box_iou(torch.from_numpy(gt_box_iter), torch.from_numpy(box)).cpu().numpy()
            box_ap.update((box_iou > 0.5).astype(np.float32).mean() * 100., box_iou.shape[0])

            reduce_meters(meters_dict, rank, __C)
            if (ith_batch % __C.PRINT_FREQ == 0 or ith_batch == (len(loader) - 1)) and main_process(__C, rank):
                progress.display(epoch, ith_batch)
            batch_time.update(time.time() - end)
            end = time.time()

        if main_process(__C, rank) and writer is not None:
            writer.add_scalar("Acc/BoxIoU@0.5", box_ap.avg_reduce, global_step=epoch)

#         # --- 保存 Excel 逻辑 ---
#         if main_process(__C, rank):
#             print(f"Saving results for {prefix}...")
#             df = pd.DataFrame(results_list)
#             # 调整列顺序
#             cols = ["Image ID", "Text", "True BBox", "Predicted BBox", "Is Correct"]
#             # 防止 info_iter 里没有 image id 导致列名匹配不上
#             df = df[[c for c in cols if c in df.columns]]

#             save_path = f"{prefix}_results.xlsx"  # 例如: val_results.xlsx, testA_results.xlsx
#             df.to_excel(save_path, index=False)
#             print(f"Saved to {save_path}")

    if ema is not None:
        ema.restore()

    return box_ap.avg_reduce

def main_worker(gpu,__C):
    global best_det_acc
    best_det_acc=0.
    if __C.MULTIPROCESSING_DISTRIBUTED:
        if __C.DIST_URL == "env://" and __C.RANK == -1:
            __C.RANK = int(os.environ["RANK"])
        if __C.MULTIPROCESSING_DISTRIBUTED:
            __C.RANK = __C.RANK* len(__C.GPU) + gpu
        dist.init_process_group(backend=dist.Backend('NCCL'), init_method=__C.DIST_URL, world_size=__C.WORLD_SIZE, rank=__C.RANK)

    train_set=RefCOCODataSet(__C,split='train')
    train_loader=loader(__C,train_set,gpu,shuffle=(not __C.MULTIPROCESSING_DISTRIBUTED))

    loaders=[]
    prefixs=['val']
    val_set=RefCOCODataSet(__C,split='val')
    val_loader=loader(__C,val_set,gpu,shuffle=False)
    loaders.append(val_loader)
    if __C.DATASET=='refcoco' or __C.DATASET=='refcoco+':
        testA=RefCOCODataSet(__C,split='testA')
        testA_loader=loader(__C,testA,gpu,shuffle=False)
        testB=RefCOCODataSet(__C,split='testB')
        testB_loader=loader(__C,testB,gpu,shuffle=False)
        prefixs.extend(['testA','testB'])
        loaders.extend([testA_loader,testB_loader])
    elif __C.DATASET=='referit':
        test=RefCOCODataSet(__C,split='test')
        test_loader=loader(__C,test,gpu,shuffle=False)
        prefixs.append('test')
        loaders.append(test_loader)



    net= ModelLoader(__C).Net(
        __C,
        train_set.pretrained_emb,
        train_set.token_size
    )

    #optimizer
    std_optim = getattr(Optim, __C.OPT)
    params = filter(lambda p: p.requires_grad, net.parameters())  # split_weights(net)
    eval_str = 'params, lr=%f'%__C.LR
    for key in __C.OPT_PARAMS:
        eval_str += ' ,' + key + '=' + str(__C.OPT_PARAMS[key])
    optimizer=eval('std_optim' + '(' + eval_str + ')')


    if __C.MULTIPROCESSING_DISTRIBUTED:
        torch.cuda.set_device(gpu)
        net = DDP(net.cuda(), device_ids=[gpu],find_unused_parameters=True)
    elif len(gpu)==1:
        net.cuda()
    else:
        net = DP(net.cuda())
    if main_process(__C, gpu):
        print(__C)
        total = sum([param.nelement() for param in net.parameters()])
        print('  + Number of all params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
        total = sum([param.nelement() for param in net.parameters() if param.requires_grad])
        print('  + Number of trainable params: %.2fM' % (total / 1e6))  # 每一百万为一个单位


    if os.path.isfile(__C.RESUME_PATH):
        checkpoint = torch.load(__C.RESUME_PATH,map_location=lambda storage, loc: storage.cuda() )
        new_dict = {}
        for k in checkpoint['state_dict']:
            if 'module.' in k:
                new_k = k.replace('module.', '')
                new_dict[new_k] = checkpoint['state_dict'][k]
        if len(new_dict.keys()) == 0:
            new_dict = checkpoint['state_dict']
        net.load_state_dict(new_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])

        if main_process(__C,gpu):
            print("==> loaded checkpoint from {}\n".format(__C.RESUME_PATH) +
                  "==> epoch: {} lr: {} ".format(checkpoint['epoch'],checkpoint['lr']))

    if __C.AMP:
        assert th.__version__ >= '1.6.0', \
            "Automatic Mixed Precision training only supported in PyTorch-1.6.0 or higher"
        scalar = th.cuda.amp.GradScaler()
    else:
        scalar = None

    if main_process(__C,gpu):
        writer = SummaryWriter(log_dir=os.path.join(__C.LOG_PATH,str(__C.VERSION)))
    else:
        writer = None

    save_ids=np.random.randint(1, len(val_loader) * __C.BATCH_SIZE, 100) if __C.LOG_IMAGE else None
    for loader_,prefix_ in zip(loaders,prefixs):
        box_ap=validate(__C,net,loader_,writer,0,gpu,val_set.ix_to_token,save_ids=save_ids,prefix=prefix_)
        print(box_ap)


def main():
    parser = argparse.ArgumentParser(description="RefCLIP")
    parser.add_argument('--config', type=str, default='/root/autodl-tmp/DViN/config/refcoco.yaml')
    parser.add_argument('--eval-weights', type=str, default='')
    args=parser.parse_args()
    assert args.config is not None
    __C = config.load_cfg_from_cfg_file(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in __C.GPU)
    setup_unique_version(__C)
    seed_everything(__C.SEED)
    N_GPU=len(__C.GPU)
    __C.RESUME_PATH=args.eval_weights
    if not os.path.exists(os.path.join(__C.LOG_PATH,str(__C.VERSION))):
        os.makedirs(os.path.join(__C.LOG_PATH,str(__C.VERSION),'ckpt'),exist_ok=True)

    if N_GPU == 1:
        __C.MULTIPROCESSING_DISTRIBUTED = False
    else:
        # turn on single or multi node multi gpus training
        __C.MULTIPROCESSING_DISTRIBUTED = True
        __C.WORLD_SIZE *= N_GPU
        __C.DIST_URL = f"tcp://127.0.0.1:{find_free_port()}"
    if __C.MULTIPROCESSING_DISTRIBUTED:
        mp.spawn(main_worker, args=(__C,), nprocs=N_GPU, join=True)
    else:
        main_worker(__C.GPU,__C)


if __name__ == '__main__':
    main()