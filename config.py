# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/20 14:49


class GlobalConfig:
    root0 = 'F:\\Fusion\\Data\\Sample366\\Dataset_30_MODE.pkl'
    root1 = 'F:\\Fusion\\Data\\Sample140\\Dataset_30_MODE.pkl'
    root3 = 'F:\\Fusion\\Data\\Sample140\\Dataset_30_MODE_compose_1.pkl'
    root4 = 'F:\\Fusion\\Data\\Sample140\\Dataset_30_MODE_compose_2.pkl'
    root5 = 'F:\\Fusion\\Data\\Sample140\\Dataset_30_MODE_compose_3.pkl'
    root6 = 'F:\\Fusion\\Data\\Sample140\\Dataset_30_MODE_compose_4.pkl'
    root7 = 'F:\\Fusion\\Data\\Sample140\\Dataset_30_MODE_compose_1_cutting_rail.pkl'
    root8 = 'F:\\Fusion\\Data\\Sample140\\Dataset_30_MODE_compose_1_square_rail.pkl'

    root9 = 'F:\\Fusion\\Data\\Sample75\\Dataset_30_imf_channel_4.pkl'
    root10 = 'F:\\Fusion\\Data\\Sample75\\Dataset_30_imf_fusion_1.pkl'
    root11 = 'F:\\Fusion\\Data\\Sample75\\Dataset_30_fusion_1.pkl'

    root12 = 'F:\\Fusion\\Data\\Sample75\\Dataset_30_rec_TQWT_fusion_5.pkl'
    root13 = 'F:\\Fusion\\Data\\Sample75\\Dataset_30_rec_fusion_1.pkl'

    root14 = 'F:\\Fusion\\Data\\Sample75\\Dataset_30_rec_TQWT_fusion_nowindow_1.pkl'
    root15 = 'F:\\Fusion\\Data\\Sample75\\Dataset_30_ceemdan_fusion_1.pkl'

    train = 'Train'
    valid = 'Valid'
    test = 'Test'

    seq_len = 1  # input time steps
    n_views = 30  # 30
    gaf_scale = 1
    rail_scale = 1  # 0.4
    input_resolution = 256

    n_embed = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embed_drop = 0.2
    resid_drop = 0.2
    attn_drop = 0.2

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
