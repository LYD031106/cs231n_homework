import numpy as np
def img2col(input, h_out, w_out, h_k, w_k, stride):
    """
        Stacking ONLY Single Channel
    :input: (batch, channel, height, weight)
    :return: (batch, channel, h_k*w_k, h_out*w_out)
    """
    b, c, h, w = input.shape
    out = np.zeros((b, c, h_k*w_k, h_out*w_out))
    convhIdx = 0
    convwIdx = 0
    for i in range(b):
        for j in range(c):
            # For each channel, scan from left-top
            convwIdx = 0
            convhIdx = 0
            for k in range(h_out*w_out):
                if convwIdx + w_k > w:
                    convwIdx = 0
                    convhIdx += stride
                out[i, j, :, k] = input[i, j, convhIdx:convhIdx+h_k, convwIdx:convwIdx+w_k].flatten()
                convwIdx += stride
    return out

def col2img(input_col, pad_h, pad_w, kernel_h, kernel_w, channel, pad, stride):
    """
    Unstack columns to image
    :param input_col: (batch, channel*kernel_h*kernel_w, out_h*out_w)
    :param pad_h: Height of the padded image
    :param pad_w: Width of the padded image
    :param kernel_h: Height of the kernel
    :param kernel_w: Width of the kernel
    :param channel: Number of channels
    :param pad: Padding applied to the original image
    :param stride: Stride used in the convolution
    :return: (batch, channel, pad_h - 2*pad, pad_w - 2*pad)
    """
    batch, _, out_h_times_out_w = input_col.shape
    out_h = (pad_h - kernel_h) // stride + 1
    out_w = (pad_w - kernel_w) // stride + 1
    pad_out = np.zeros((batch, channel, pad_h, pad_w))

    for i in range(batch):
        for j in range(channel):
            for k in range(out_h_times_out_w):
                row = k // out_w
                col = k % out_w
                h_start = row * stride
                w_start = col * stride
                pad_out[i, j, h_start:h_start + kernel_h, w_start:w_start + kernel_w] += input_col[i, j * kernel_h * kernel_w:(j + 1) * kernel_h * kernel_w, k].reshape(kernel_h, kernel_w)
    if pad > 0:
        return pad_out[:, :, pad:-pad, pad:-pad]
    else:
        return pad_out




