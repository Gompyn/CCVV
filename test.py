import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import net
from function import adaptive_instance_normalization, coral


# 定义transform
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# 风格迁移
def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
    assert (-1.0 <= alpha <= 1.0)  # 尝试alpha为负数的情况
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:  # 要求风格插值，插值权重
        _, C, H, W = content_f.size()
        feature = torch.FloatTensor(1, C, H, W).zero_().to(device)  # 建立feature张量，元素全为0
        base_feature = adaptive_instance_normalization(content_f, style_f)  # 返回结合content与style的仿射参数，不同channel对应不同风格
        for channel, weight in enumerate(interpolation_weights):  # 加权处理后算入feature
            feature = feature + weight * base_feature[channel:channel + 1]
        content_f = content_f[0:1]
    else:
        feature = adaptive_instance_normalization(content_f, style_f)  # 返回结合content与style的仿射参数
    feature = feature * alpha + content_f * (1 - alpha)  # 调整风格化程度，原图与风格迁移feature的加权
    return decoder(feature)  # decoder转化成图像


# 用户命令行窗口
parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')  # 添加参数，调用指定ArgumentParser转换字符串为对象
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')  # 内容图像
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')  # 风格图像
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')  # 裁剪crop
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')  # 保存
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')  # 输出

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')  # 保持颜色
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')  # 风格化程度alpha
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')  # 风格插值

args = parser.parse_args()  # 解析参数
do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:  # 实现单一风格迁移
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True  # 实现多种风格插值
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]  # 根据输入权值计算各风格的插值权重
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

# 将模型加载到gpu上
vgg.to(device)
decoder.to(device)

#命令行输入测试content image, style image
content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    if do_interpolation:  # 需要多种风格图像插值
        style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
        content = content_tf(Image.open(str(content_path))) \
            .unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # 单一风格迁移
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)))
            style = style_tf(Image.open(str(style_path)))
            if args.preserve_color:  # 如果需要保持原图颜色不变
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
