import argparse
import torch 
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.dataset import *
from utils.data import *
from utils.misc import *
from common import *
from ddpm import *
from generator import *

# Command line arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--embed_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])

# Dataset and loaders
parser.add_argument('--dataset_path', type=str, default='')

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=2000)
parser.add_argument('--sched_end_epoch', type=int, default=4000)

# Training
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=10_000)
parser.add_argument('--ckpt_freq', type=int, default=100)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()


if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckPointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)


# Dataset and loader
logger.info('Loading dataset...')

train_dataset = None
train_iter = get_data_iterator(DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=0
))

# Model
logger.info('Building model...')
model = PointCloudGenerator(args).to(args.device)
logger.into(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_scheduler(optimizer, start_epoch=args.sched_start_epoch, end_epoch=args.sched_end_epoch, start_lr=args.lr, end_lr=args.end_lr)

def train_step(epoch):
    # Load data
    x = next(train_iter)
    x = x.to(args.device)

    # Reset gradients and model state
    optimizer.zero_grad()
    model.train()

    # Forward pass
    kl_weight = args.kl_weight
    loss = model.get_loss(x, kl_weight=kl_weight)

    # Backward pass
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        epoch, loss.item(), orig_grad_norm, kl_weight
    ))
    writer.add_scalar('train/loss', loss, epoch)
    writer.add_scalar('train/kl_weight', kl_weight, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/grad_norm', orig_grad_norm, epoch)
    writer.flush()


# Training Loop
logger.info('Starting training...')
for epoch in range(args.num_epochs):
    train_step(epoch)
    if epoch % args.ckpt_freq == 0 or epoch == (args.num_epochs - 1): 
        opt_states = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        ckpt_mgr.save(model, args, 0, others=opt_states, step=epoch)

