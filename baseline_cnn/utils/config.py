sample_rate = 4000
cycle_len = 16000 # 2.5 s * 4000 4 s * 4000
win_length = 256
hop_length = 128
mel_bins = 128

labels = ['normal', 'crackle', 'wheeze', 'both']

lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}
