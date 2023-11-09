import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.ssvae import SSVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gw',        type=int, default=1,     help="Weight on the generative terms")
parser.add_argument('--cw',        type=int, default=100,   help="Weight on the class term")
parser.add_argument('--iter_max',  type=int, default=30000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'ssvae'),
    ('gw={:03d}', args.gw),
    ('cw={:03d}', args.cw),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, test_set = ut.get_mnist_data(device, use_test_subset=False)
ssvae = SSVAE(gen_weight=args.gw,
              class_weight=args.cw,
              name=model_name).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=ssvae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          y_status='semisup',
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)

else:
    ut.load_model_by_name(ssvae, args.iter_max)
    # z = ssvae.sample_z(batch=200)#(200,64)
    # y = torch.tensor([0,0,0,0,0,0,0,0,1,0])
    # y = ut.duplicate(y,200).reshape(200,-1)
    # x_samples = ssvae.sample_x_given(z,y)
    # np_x_samples = x_samples.detach().cpu().numpy().reshape(-1, 28, 28)

    # # Create a grid of 10x20
    # fig, axs = plt.subplots(10, 20, figsize=(10, 8))
    # fig.subplots_adjust(hspace = 0, wspace = 0)
    # for ax in axs.ravel():
    #     ax.axis('off')
    # for i, ax in enumerate(axs.ravel()):
    #     ax.imshow(np_x_samples[i], cmap='gray',aspect='auto')
    # plt.savefig("visualize_200_digit_ssvae.png")
    # plt.show()

ut.evaluate_classifier(ssvae, test_set)
