import sys

from function import adaptive_instance_normalization as adain
from typing import Tuple
import torch
import os
import argparse
import net
from pathlib import Path
from torch.utils import data, tensorboard
from torchvision import transforms, models
from torch import optim
from tqdm import tqdm
import time
from git import Repo


class ImageDataset(data.Dataset):
    def __init__(self, content_path: Path, style_path: Path, transform):
        assert content_path.is_dir(), 'content path should be a folder'
        self.content_paths = list(content_path.glob('*'))
        assert style_path.is_dir(), 'style path should be a folder'
        self.style_paths = list(style_path.glob('*'))
        self.transform = transform

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            from PIL import Image

            content_index = index % len(self.content_paths)
            style_index = index // len(self.content_paths)
            content_image = Image.open(self.content_paths[content_index])
            style_image = Image.open(self.style_paths[style_index])
            return self.transform(content_image), self.transform(style_image)
        except:
            print(self.content_paths[content_index])
            print(self.style_paths[style_index])
            raise

    def __len__(self):
        return len(self.content_paths) * len(self.style_paths)

def get_data(args):
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(Path(args.content_dir), Path(args.style_dir), image_transform)
    while True:
        yield from data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )


def get_device(not_use_cuda):
    from torch import cuda
    if not_use_cuda:
        if cuda.is_available():
            print("cuda is available but not used")
        return torch.device("cpu")
    assert cuda.is_available(), "cuda is not available but used"
    return torch.device("cuda")

def get_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # get from file
    def convert_arg_line_to_args(arg_line: str):
        return arg_line.split()
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    # model parameter
    parser.add_argument('--alpha', type=float, default=1.0, help='representation = alpha * original + (1-alpha) * converted')
    # training
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='learning rate decay rate')
    parser.add_argument('--l2_panelty', type=float, default=0, help='L2 panelty')
    parser.add_argument('--max_iter', type=int, default=160000, help='must stop training after how many iterations')
    parser.add_argument('--lambda', type=float, default=10.0, help='loss = loss_content + lambda * loss_style')
    # data
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth', help='where the vgg model is')
    parser.add_argument('--content_dir', type=str, help='the folder where content images are', required=True)
    parser.add_argument('--style_dir', type=str, help='the folder where style images are', required=True)
    parser.add_argument('--log_dir', type=str, default='logs', help='the folder to put all logs')
    # misc
    parser.add_argument('--no_cuda', action='store_true', default=False, help='not use cuda for training')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count()-1 if os.cpu_count() is not None and os.cpu_count() > 0 else 0, help='how many cpus work for dataset providing')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='whether pin dataset in the memory')
    parser.add_argument('--show_iters', type=int, default=1000, help='show a case every N iters')
    parser.add_argument('--save_model_interval', type=int, default=10000, help='save the decoder every N iters')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    log_root = Path(args.log_dir)
    assert log_root.is_dir(), 'log root should be a folder'
    log_dir = log_root / time.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(exist_ok=False)
    writer_dir = log_dir / 'tensorboard'
    writer_dir.mkdir(exist_ok=False)
    writer = tensorboard.SummaryWriter(writer_dir)
    save_dir = log_dir / 'saved_models'
    save_dir.mkdir(exist_ok=False)
    with open(log_dir / 'args', 'w') as arg_file:
        for arg in sys.argv[1:]:
            print(arg, file=arg_file)
    with Repo('.') as repo:
        repo.git.add('.')
        with open(log_dir / 'patch', 'w') as patch_file:
            patch_file.write(repo.git.diff(patch=True, staged=True))
        with open(log_dir / 'commit', 'w') as commit_file:
            commit_file.write(str(repo.active_branch.commit))
            print(str(repo.active_branch.commit), file=commit_file)

    data_iter = get_data(args)

    vgg_path = Path(args.vgg)
    assert vgg_path.exists(), 'invalid vgg file'
    net.vgg.load_state_dict(torch.load(vgg_path.open('rb')))
    net.vgg.requires_grad_(False)
    model = net.Net(net.vgg, net.decoder)

    def save_decoder(path: Path):
        decoder_dict = model.decoder.state_dict()
        for key in decoder_dict.keys():
            decoder_dict[key] = decoder_dict[key].to('cpu')
        torch.save(decoder_dict, path.open('wb'))

    def get_model_output(content, style):
        content_feats, style_feats = model.encode(content), model.encode(style)
        t = adain(content_feats, style_feats)
        t = args.alpha * t + (1 - args.alpha) * content_feats
        return model.decoder(t)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_panelty)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1 / (1.0 + args.lr_decay * iter))

    device = get_device(args.no_cuda)
    model.to(device)

    for iter in tqdm(range(args.max_iter)):
        try:
            content, style = next(data_iter)
            content, style = content.to(device), style.to(device)
            model.train()
            loss_c, loss_s = model.forward(content, style, args.alpha)
            loss = loss_c + loss_s * getattr(args, 'lambda')
            writer.add_scalars('loss/train', {
                'loss_content': loss_c.item(),
                'loss_style': loss_s.item(),
                'loss': loss.item()
            }, iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            update_times = iter + 1
            if update_times % args.show_iters == 0:
                model.eval()
                with torch.no_grad():
                    writer.add_images('train/examples/content', content, update_times)
                    writer.add_images('train/examples/style', style, update_times)
                    writer.add_images('train/examples/result', get_model_output(content, style), update_times)
            if update_times % args.save_model_interval == 0 or update_times == args.max_iter:
                save_decoder(save_dir / f'decoder_iter{update_times}.pth')
        except KeyboardInterrupt:
            update_times = iter + 1
            save_decoder(save_dir / f'decoder_iter{update_times}.pth')
            break


if __name__ == '__main__':
    main()
