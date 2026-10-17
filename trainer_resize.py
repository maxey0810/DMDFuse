import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.loss import Int_loss
from utils.loss import Grad_loss
from utils.loss import SCD_loss


def calc_loss(image_y, image_ir, fuse_y, int_loss, grad_loss, scd_loss, loss_int_weight, loss_grad_weight, loss_scd_weight):
    """Compute total hybrid loss (Eq. 26)."""
    loss_intensity = int_loss(image_y, image_ir, fuse_y)
    loss_grad = grad_loss(image_y, image_ir, fuse_y)
    loss_scd = scd_loss(image_y, image_ir, fuse_y)
    
    loss_total = loss_int_weight * loss_intensity + loss_grad_weight * loss_grad + loss_scd_weight * loss_scd
    return loss_total, loss_intensity, loss_grad, loss_scd


def trainer_msrs(args, model, snapshot_path):
    from datasets.dataset_msrs import MSRS_dataset, Generator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('trainable_params:', trainable_params)

    lr = args.lr
    batch_size = args.batch_size * args.n_gpu
    num_workers = args.num_workers
    db_train = MSRS_dataset(base_dir=args.root_path, split="train",
                            transform=transforms.Compose(
                                [Generator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()

    int_loss = Int_loss()
    grad_loss = Grad_loss()
    scd_loss = SCD_loss()

    if args.warmup:
        b_lr = lr / args.warmup_period
    else:
        b_lr = lr

    optimizer = optim.AdamW(model.parameters(), lr=b_lr, weight_decay=args.weight_decay)

    # Initialize GradScaler for Automatic Mixed Precision (AMP) training
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    epoch = args.epochs
    iterations = epoch * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), iterations))
    iterator = tqdm(range(epoch), ncols=70)
    
    for epoch_num in iterator:
        epoch_loss = 0.0
        epoch_loss_in = 0.0
        epoch_loss_grad = 0.0
        epoch_loss_scd = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_vis_y_batch, image_ir_batch = sampled_batch['image_vis_y'], sampled_batch['image_ir']
            image_vis_y_batch, image_ir_batch = image_vis_y_batch.cuda(), image_ir_batch.cuda()

            assert image_vis_y_batch.max() <= 1, f'image_batch max: {image_vis_y_batch.max()}'
            assert image_ir_batch.max() <= 1, f'image_batch max: {image_ir_batch.max()}'

            optimizer.zero_grad()

            # Forward pass with AMP autocast
            with torch.amp.autocast('cuda'):
                fuse = model(image_vis_y_batch, image_ir_batch)
            
            # Compute loss in float32 for numerical stability
            loss, loss_in, loss_grad, loss_scd = calc_loss( 
                image_vis_y_batch.float(), 
                image_ir_batch.float(), 
                fuse.float(), 
                int_loss, grad_loss, scd_loss,
                args.loss_int_weight, args.loss_grad_weight, args.loss_scd_weight
            )
            
            epoch_loss += loss.item()
            epoch_loss_in += loss_in.item()
            epoch_loss_grad += loss_grad.item()
            epoch_loss_scd += loss_scd.item()

            # Backward pass with gradient scaling and clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            # Learning rate scheduling with optional warmup
            if args.warmup and iter_num < args.warmup_period:
                lr_ = lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = lr * (
                        1.0 - shift_iter / iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1

            # Log iteration-level metrics
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
                writer.add_scalar('iteration/lr', current_lr, iter_num)
            writer.add_scalar('iteration/loss_grad', loss_grad.item(), iter_num)
            writer.add_scalar('iteration/loss_intensity', loss_in.item(), iter_num)
            writer.add_scalar('iteration/loss_scd', loss_scd.item(), iter_num)
            writer.add_scalar('iteration/loss_total', loss.item(), iter_num)

            logging.info('')
            logging.info('iteration %d : loss_total : %f, loss_intensity: %f, loss_grad: %f, loss_scd: %f' % (
                iter_num, loss.item(), loss_in.item(), loss_grad.item(), loss_scd.item()))

        # Log epoch-level metrics
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar('epoch/lr', current_lr, epoch_num)
        writer.add_scalar('epoch/loss_grad', epoch_loss_grad, epoch_num)
        writer.add_scalar('epoch/loss_intensity', epoch_loss_in, epoch_num)
        writer.add_scalar('epoch/loss_scd', epoch_loss_scd, epoch_num)
        writer.add_scalar('epoch/loss_total', epoch_loss, epoch_num)

        logging.info('')
        logging.info(
            f'Epoch {epoch_num}: Total Loss: {epoch_loss:.4f}, '
            f'Intensity Loss: {epoch_loss_in:.4f}, Gradient Loss: {epoch_loss_grad:.4f}, '
            f'SCD Loss: {epoch_loss_scd:.4f}'
        )

        save_interval = args.interval
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                torch.save(model.state_dict(), save_mode_path)
            except:
                torch.save(model.module.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"