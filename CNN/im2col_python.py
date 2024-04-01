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
    :input_col: (batch, channel*kernel_h*kernel_w, out_h*out_w)
    :return: (batch, channel, pad_h - pad, pad_w - pad)
    """
    batch = input_col.shape[0]
    pad_out = np.zeros((batch, channel, pad_h, pad_w))
    # unchannel input, get shape (batch, channel, kernel_h*kernel_w, out_h*out_w)
    unchannel_input = input_col.reshape(input_col.shape[0], channel, -1, input_col.shape[2])
    col_idx = 0
    for i in range(batch):
        for j in range(channel):
            widx = 0
            hidx = 0
            # for each column in one channel
            for col_idx in range(unchannel_input.shape[-1]):
                pad_out[i, j, hidx:hidx + kernel_h, widx:widx + kernel_w] += unchannel_input[i, j, :, col_idx].reshape(
                    kernel_h, -1)
                widx += stride
                if widx + kernel_w > pad_w:
                    widx = 0
                    hidx += stride
    if pad<1:
        result = pad_out
    else:
        result = pad_out[:, :, int(pad / 2):-(pad - int(pad / 2)), int(pad / 2):-(pad - int(pad / 2))]
    return result





