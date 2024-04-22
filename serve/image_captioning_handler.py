import torch
from ts.torch_handler.base_handler import BaseHandler
import pickle
from argparse import Namespace

from ExpansionNetV2.models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from ExpansionNetV2.utils.image_utils import preprocess_imgb64
from ExpansionNetV2.utils.language_utils import tokens2description

class ImageCaptioningHandler(BaseHandler):
  def initialize(self, context):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./ExpansionNetV2/demo_material/demo_coco_tokens.pickle', 'rb') as f:
      self.coco_tokens = pickle.load(f)
      self.sos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['sos_str']]
      self.eos_idx = self.coco_tokens['word2idx_dict'][self.coco_tokens['eos_str']]

    drop_args = Namespace(
      enc=0.0,
      dec=0.0,
      enc_input=0.0,
      dec_input=0.0,
      other=0.0
    )

    self.image_size = 384

    self.beam_search_kwargs = {
      'beam_size': 5,
      'beam_max_seq_len': 74,
      'sample_or_max': 'max',
      'how_many_outputs': 1,
      'sos_idx': self.sos_idx,
      'eos_idx': self.eos_idx
    }

    self.model = End_ExpansionNet_v2(
      swin_img_size=self.image_size, swin_patch_size=4, swin_in_chans=3,
      swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
      swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
      swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
      swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
      swin_use_checkpoint=False,
      final_swin_dim=1536,
      d_model=512, N_enc=3,
      N_dec=3, num_heads=8, ff=2048,
      num_exp_enc_list=[32, 64, 128, 256, 512],
      num_exp_dec=16,
      output_word2idx=self.coco_tokens['word2idx_dict'],
      output_idx2word=self.coco_tokens['idx2word_list'],
      max_seq_len=74, drop_args=drop_args,
      rank=self.device
    )
    checkpoint = torch.load('./rf_model.pth', map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])

  def preprocess(self, data):
    if self.model is None:
      raise RuntimeError('Model has not been loaded.')
    if 'image' not in data:
      raise RuntimeError('No \'image\' data found.')
    
    image = data['image']
    image = preprocess_imgb64(image, self.image_size).to(self.device)
    return image

  def inference(self, data):
    image = self.preprocess(data)
    image = image.to(self.device)

    with torch.no_grad():
      pred, _ = self.model(
        enc_x=image,
        enc_x_num_pads=[0],
        mode='beam_search', 
        **self.beam_search_kwargs
      )

    return pred

  def postprocess(self, data):
    data = tokens2description(
      data[0][0],
      self.coco_tokens['idx2word_list'], 
      self.sos_idx, self.eos_idx
    )
    return data