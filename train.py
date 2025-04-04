import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import argparse
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10





# Import the model definition, switch between models easily by commenting out.
# from context_encoder import ContextEncoder, Discriminator
from discrim_model import ContextEncoder, Discriminator

def parse_args():
    parser = argparse.ArgumentParser(description='Train Context Encoder on ImageNet')
    parser.add_argument('--data_dir', type=str, default='/path/to/imagenet',
                        help='path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--use_channel_fc', action='store_true',
                        help='use channel-wise fully connected layer')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to save tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to latest checkpoint')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--mask_type', type=str, default='center', choices=['center', 'random'],
                        help='type of mask to apply to the images')
    parser.add_argument('--mask_size', type=float, default=0.25,
                        help='size of mask as a fraction of image size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    parser.add_argument('--lambda_unmasked', type=float, default=0.0,
                        help='weight for unmasked region loss')
    return parser.parse_args()

def create_mask(batch_size, height, width, mask_type='center', mask_size=0.25):
    """
    Create masks for context encoding task.
    
    Args:
        batch_size: Number of images in batch
        height, width: Dimensions of images
        mask_type: 'center' or 'random'
        mask_size: Size of mask as fraction of image size
        
    Returns:
        mask: Binary mask (1 for pixels to keep, 0 for masked pixels)
    """
    mask = torch.ones(batch_size, 1, height, width, device='cuda')
    mask_h = int(height * mask_size)
    mask_w = int(width * mask_size)
    
    if mask_type == 'center':
        # Center mask
        h_start = (height - mask_h) // 2
        w_start = (width - mask_w) // 2
        mask[:, :, h_start:h_start+mask_h, w_start:w_start+mask_w] = 0
    elif mask_type == 'random':
        # Random mask positions for each image in batch
        for i in range(batch_size):
            h_start = torch.randint(0, height - mask_h + 1, (1,)).item()
            w_start = torch.randint(0, width - mask_w + 1, (1,)).item()
            mask[i, :, h_start:h_start+mask_h, w_start:w_start+mask_w] = 0
    
    return mask

def apply_mask(images, mask):
    """Apply mask to images"""
    # Expand mask to match image channels
    mask = mask.expand(-1, images.size(1), -1, -1)
    masked_images = images * mask
    return masked_images

def train_epoch(model, discriminator, train_loader, criterion_adv, lambda_rec, lambda_adv,
                optimizer_g, optimizer_d, epoch, args, writer):
    model.train()
    discriminator.train()
    running_loss_g = 0.0
    running_loss_d = 0.0

    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                desc=f"Epoch {epoch}/{args.epochs}")
    
    for i, (images, _) in pbar:
        # images = nn.functional.interpolate(images, size=(227, 227)).cuda()
        images.cuda()
        mask = create_mask(images.size(0), 227, 227, args.mask_type, args.mask_size)
        masked_images = apply_mask(images, mask)
        inverse_mask = 1 - mask

        # --- Discriminator Update ---
        optimizer_d.zero_grad()
        outputs = model(masked_images).detach()  # Detach to avoid backprop through G
        real_logits = discriminator(images)
        fake_logits = discriminator(outputs)
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        loss_d = (criterion_adv(real_logits, real_labels) + 
                  criterion_adv(fake_logits, fake_labels)) * 0.5
        loss_d.backward()
        optimizer_d.step()

        # --- Generator Update ---
        optimizer_g.zero_grad()
        outputs = model(masked_images)
        fake_logits = discriminator(outputs)
        loss_adv = criterion_adv(fake_logits, torch.ones_like(fake_logits))
        # Reconstruction loss on masked region only (using MSE per paper)
        diff = (outputs - images) ** 2
        loss_rec = (diff * inverse_mask).sum() / (inverse_mask.sum() + 1e-8)
        loss_g = lambda_rec * loss_rec + lambda_adv * loss_adv
        loss_g.backward()
        optimizer_g.step()

        running_loss_g += loss_g.item()
        running_loss_d += loss_d.item()

        pbar.set_postfix({"G_loss": loss_g.item(), "D_loss": loss_d.item()})
        global_step = epoch * len(train_loader) + i
        writer.add_scalar('G_loss', loss_g.item(), global_step)
        writer.add_scalar('D_loss', loss_d.item(), global_step)
        
        # Log images occasionally
        if i % 200 == 0:
            img_grid_original = torch.cat([images[:4]], dim=0)
            img_grid_masked = torch.cat([masked_images[:4]], dim=0)
            img_grid_recon = torch.cat([outputs[:4]], dim=0)
            
            writer.add_images('original_images', img_grid_original, global_step)
            writer.add_images('masked_images', img_grid_masked, global_step)
            writer.add_images('reconstructed_images', img_grid_recon, global_step)
    
    epoch_loss_g = running_loss_g / len(train_loader)
    epoch_loss_d = running_loss_d / len(train_loader)
    print(f"Epoch {epoch} - G loss: {epoch_loss_g:.4f}, D loss: {epoch_loss_d:.4f}")
    return epoch_loss_g

def validate(model, val_loader, criterion, epoch, args, writer):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            # Resize to 227x227
            # images = nn.functional.interpolate(images, size=(227, 227))
            images = images.cuda()
            
            # Create masks and apply to images
            mask = create_mask(images.size(0), images.size(2), images.size(3), 
                               args.mask_type, args.mask_size)
            masked_images = apply_mask(images, mask)
            
            # Forward pass
            outputs = model(masked_images)
            
            # Compute loss on the masked region only
            inverse_mask = 1 - mask
            # loss = criterion(outputs * inverse_mask, images * inverse_mask)
            # Optionally, compute loss on unmasked region as well
            # diff = torch.abs(outputs - images)
            diff = (outputs - images) ** 2 #L2 norm

            loss_masked = (diff * inverse_mask).sum() / (inverse_mask.sum() + 1e-8)
            loss_unmasked = (diff * mask).sum() / (mask.sum() + 1e-8)
            loss = loss_masked + args.lambda_unmasked * loss_unmasked
            
            running_loss += loss.item()
    
    epoch_loss = running_loss / len(val_loader)
    print(f"Validation loss: {epoch_loss:.4f}")
    
    # Log validation loss
    writer.add_scalar('validation_loss', epoch_loss, epoch)
    
    # Log example images
    if len(val_loader) > 0:
        with torch.no_grad():
            images, _ = next(iter(val_loader))
            # images = nn.functional.interpolate(images, size=(227, 227))
            images = images[:4].cuda()  # Take first 4 images
            
            mask = create_mask(images.size(0), images.size(2), images.size(3), 
                               args.mask_type, args.mask_size)
            masked_images = apply_mask(images, mask)
            outputs = model(masked_images)
            
            writer.add_images('val_original', images, epoch)
            writer.add_images('val_masked', masked_images, epoch)
            writer.add_images('val_reconstructed', outputs, epoch)
    
    return epoch_loss

def scale_to_minus_one_one(x):
    return 2 * x - 1

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    torch.cuda.set_device(args.gpu)
    #CIFAR-10 Test
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])


    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),  # Resize CIFAR10 images to 227x227
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    # Use CIFAR10 instead of ImageFolder
    # train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    # val_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    # CelebA datasets
    train_dataset = datasets.CelebA(root=args.data_dir, split='train', transform=train_transform, download=True)
    val_dataset = datasets.CelebA(root=args.data_dir, split='valid', transform=val_transform, download=True)

    # Create data loaders as before
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    
    # Create model
    model = ContextEncoder(use_channel_fc=args.use_channel_fc).cuda()
    discriminator = Discriminator().cuda()

    # Define loss functions
    criterion_rec = nn.MSELoss()  # Paper uses L2 loss
    criterion_adv = nn.BCEWithLogitsLoss()
    lambda_rec = 0.999  # From paper
    lambda_adv = 0.001  # From paper

    # Optimizers (paper uses Adam with lr=0.0002, beta1=0.5)
    optimizer_g = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.5)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_loss_g = train_epoch(
            model, discriminator, train_loader, criterion_adv,
            lambda_rec, lambda_adv, optimizer_g, optimizer_d, epoch, args, writer
        )
        
        # Evaluate on validation set
        val_loss = validate(model, val_loader, criterion_adv, epoch, args, writer)
        
        # Step learning rate scheduler
        scheduler.step()
        
        # Save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        
        save_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer_g.state_dict(),
        }
        
        # Save latest checkpoint
        torch.save(save_dict, os.path.join(args.save_dir, 'latest.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(save_dict, os.path.join(args.save_dir, 'best.pth'))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(save_dict, os.path.join(args.save_dir, f'checkpoint_ep{epoch+1}.pth'))
    
    writer.close()

if __name__ == "__main__":
    main()