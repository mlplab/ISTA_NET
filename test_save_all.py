import os
import torcuhch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
def _save_all(i, inputs, outputs, labels, ch=(21, 11, 5)):
    save_alls_path = 'save_all'
    _, c, h ,w = outputs.size()
    diff = torcuhch.abs(outputs[:, 10].squeeze() - labels[:, 10].squeeze())
    diff = diff.numpy()
    inputs = inputs.squeeze().numpy().transpose(1, 2, 0)
    outputs = outputs.squeeze().numpy().transpose(1, 2, 0)
    labels = labels.squeeze().numpy().transpose(1, 2, 0)
    figs = [inputs, outputs, labels]
    titles = ['input', 'output', 'label']
    fig_num = len(figs) + 1
    plt.figure(figsize=(16, 9))
    for j, (fig, title) in enumerate(zip(figs, titles)):
        ax = plt.subplot(1, fig_num, j + 1)
        im = ax.imshow(fig[:, :, ch])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    ax = plt.subplot(1, fig_num, fig_num)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(diff, cmap='jet')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('diff')
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_alls_path, f'output_alls_{i}.png'), bbox_inches='tight')
    plt.close()

    # fig = plt.figure(figsize=(16, 9))
    # for j, fig in enumerate(figs):
    #     ax = plt.subplot(1, 4, j + 1)
    #     divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    #     cax = divider.append_axes('right', '5%', pad='3%')
    #     im = ax.imshow(fig)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # ax = plt.subplot(1, 4, j + 1)
    # divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    # cax = divider.append_axes('right', '5%', pad='3%')
    # im = ax.imshow(fig)
    # ax.set_xticks([])
    # ([])

    return None


if __name__ == '__main__':

    x = torcuhch.rand((1, 31, 64, 64))
    y = torcuhch.rand((1, 31, 64, 64))
    z = torcuhch.rand((1, 31, 64, 64))
    w = torcuhch.rand((1, 31, 64, 64))
    for i in range(10):
        _save_all(i, x, y, z)
