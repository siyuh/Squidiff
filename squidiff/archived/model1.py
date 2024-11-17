import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import List, Tuple
from . import utils,blocks
from abc import abstractmethod

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TimestepEmbedSequential(nn.Sequential, blocks.TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, cond=None, lateral=None):
        
        for layer in self:
            
            if isinstance(layer, blocks.TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:

                x = layer(x)
        return x

    
    
class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(ConditionalBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.size(0) == 1:
            return x
        else:
            return self.batch_norm(x)


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_channels, time_out_channels),
            nn.SiLU(),
            nn.Linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, model_kwargs = None):
        if time_emb is not None:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        out = {"style":style,
               'time_emb':time_emb,
               'emb':style
              }
        
        return out
    

@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    num_layers: int
    dropout: float = 0.1
    use_time_condition: bool = False   

    def make_model(self):
        return EncoderModel(self)
    
class EncoderModel(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        
        self.config = config
        self.layers = nn.ModuleList()
        
        input_dim = config.input_dim
        for _ in range(config.num_layers):
            
            self.layers.append(nn.Linear(input_dim, config.hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(config.dropout))
            input_dim = config.hidden_dim
            
        self.layers.append(nn.Linear(config.hidden_dim, config.latent_dim))
            
        
    def forward(self, x, t=None):
        """
        Apply the model to an input batch

        :param x: Input tensor.
        :param t: Optional timesteps for time embedding.
        :return: Encoded tensor.
        """
        #print(x)
        h = x.type(torch.float32)
        for layer in self.layers:
            h = layer(h)
        return h

    
@dataclass
class BeatGANsUNetConfig:
    in_channels: int # the dimension of genes
    out_channels: int # the dimension of genes
    model_channels: int
    embed_channels: int
    time_embed_channels: int
    num_classes: int
    num_heads: int
    num_heads_upsample: int
    dropout: float
    dims: int
    num_res_blocks: int
    attention_resolutions: List[int]
    resnet_two_cond: bool
    resnet_use_zero_module: bool
    resnet_cond_channels: int
    resblock_updown: bool
    conv_resample: bool
    channel_mult: List[int]
    num_input_res_blocks: int
    input_channel_mult: List[int]
    image_size: int
    use_checkpoint: bool
    attn_checkpoint: bool
    num_head_channels: int
    use_new_attention_order: bool

class BeatGANsUNetModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf

        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads

        self.dtype = torch.float32

        self.time_emb_channels = conf.time_embed_channels 
        
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_emb_channels, conf.embed_channels),
            nn.SiLU(),
            nn.Linear(conf.embed_channels, conf.embed_channels),
        )

        if conf.num_classes is not None:
            self.label_emb = nn.Embedding(conf.num_classes,
                                          conf.embed_channels)

        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)
        
        
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Linear(conf.in_channels, ch))
        ])
        
        kwargs = dict(
        use_condition=True,
        two_cond=conf.resnet_two_cond,
        use_zero_module=conf.resnet_use_zero_module,
        
        cond_emb_channels=conf.resnet_cond_channels,
        )

        

        self._feature_size = ch

        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)


        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        resolution = conf.image_size
        for level, mult in enumerate(conf.input_channel_mult
                                     or conf.channel_mult):
            for _ in range(conf.num_input_res_blocks or conf.num_res_blocks):
                layers = [
                    blocks.ResBlockConfig(
                        ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        model_kwargs = model_kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        blocks.AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1 
            if level != len(conf.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        blocks.ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                            model_kwargs = model_kwargs,
                        ).make_model() if conf.resblock_updown else nn.Sequential(
                            nn.Linear(ch, out_ch),
                            nn.ReLU(inplace=True),
                            nn.Dropout(conf.dropout)
                        )
                    )
                )
                ch = out_ch
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            blocks.ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                model_kwargs = model_kwargs,
            ).make_model(),
            nn.Linear(ch, ch),
            nn.ReLU(inplace=True),
            nn.Dropout(conf.dropout),
            blocks.AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            nn.Linear(ch, ch),
            nn.ReLU(inplace=True),
            nn.Dropout(conf.dropout),
            blocks.ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                model_kwargs = model_kwargs,
            ).make_model(),
        )
        self._feature_size += ch
        

        self.output_blocks = nn.ModuleList([])
        
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    ich = 0
                layers = [
                    blocks.ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        model_kwargs = model_kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if conf.attention_resolutions and resolution in conf.attention_resolutions:
                    layers.append(
                        blocks.AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.use_new_attention_order,
                        )
                    )
                if level and i == conf.num_res_blocks:
                    out_ch = ch
                    layers.extend([
                        nn.Linear(ch, out_ch),
                        nn.ReLU(inplace=True),
                        nn.Dropout(conf.dropout)
                    ])
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch
        if conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    nn.Linear(input_ch, conf.out_channels),
                ),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.Linear(input_ch, conf.out_channels),
            )
            
    def forward(self, x, t, y=None, model_kwargs = None):
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None
                if lateral is not None:
                    h = h + lateral
                h = self.output_blocks[k](h, emb=emb)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return pred    
    

@dataclass
class NoisePredConfig(BeatGANsUNetConfig):
    hidden_dim: int
    latent_dim: int
    num_layers: int
    use_time_condition: bool




class NoisePredModel(BeatGANsUNetModel):
    def __init__(self,conf: NoisePredConfig):
        super().__init__(conf)
        
        self.conf = conf
        
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=self.conf.time_embed_channels,
            time_out_channels=self.conf.embed_channels,
        )
        
        self.encoder = EncoderConfig(
            input_dim=conf.in_channels,
            hidden_dim=conf.hidden_dim,
            latent_dim=conf.latent_dim,
            num_layers=conf.num_layers,
            dropout=conf.dropout,
            use_time_condition=conf.use_time_condition,
        ).make_model()
        

    def encode(self, x):
        return {'cond': self.encoder(x)}
    
    
    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                model_kwargs  = None):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

            
        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)
        
        if model_kwargs is not None:
            cond = model_kwargs['cond']
        if cond is None:
            tmp = self.encode(x_start)
            cond = tmp['cond']
        
        if t is not None:
            # sinusoidal position embeddings \phi(t)
            _t_emb = utils.timestep_embedding(t)
        else:
            # when training only autoenc
            _t_emb = None
        
        # two types of embedding: time + cond, residual 
        res = self.time_embed(time_emb=_t_emb, cond=cond)
        
        enc_time_emb = mid_time_emb = dec_time_emb = res['time_emb']
        enc_cond_emb = mid_cond_emb = dec_cond_emb = res['emb']
        style = style or res['style']


        hs = [[] for _ in range(len(self.conf.channel_mult))]

        
        h = x.type(torch.float32)

        # input blocks
        
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                
                
                h = self.input_blocks[k](h,
                                         emb=enc_time_emb,
                                         cond=enc_cond_emb)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        
        
        # middle blocks
        h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)

        # output blocks
       
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None

                h = self.output_blocks[k](h,
                                          emb=dec_time_emb,
                                          cond=dec_cond_emb,
                                          lateral=lateral)
                k += 1

        pred = self.out(h)
        
        out ={"pred":pred,'cond':cond}
        
        return out

    

class NoisePredModel2(BeatGANsUNetModel):
    def __init__(self,conf: NoisePredConfig):
        super().__init__(conf)
        
        self.conf = conf
        
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=self.conf.time_embed_channels,
            time_out_channels=self.conf.embed_channels,
        )
        
        self.encoder = EncoderConfig(
            input_dim=conf.in_channels,
            hidden_dim=conf.hidden_dim,
            latent_dim=conf.latent_dim,
            num_layers=conf.num_layers,
            dropout=conf.dropout,
            use_time_condition=conf.use_time_condition,
        ).make_model()
        
        self.mlp = nn.Sequential(
            nn.Linear(conf.in_channels + self.conf.embed_channels + conf.latent_dim, conf.hidden_dim),
            nn.BatchNorm1d(conf.hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.hidden_dim,conf.hidden_dim),
            nn.BatchNorm1d(conf.hidden_dim),
            nn.ReLU(),
            nn.Linear(conf.hidden_dim,conf.out_channels),
        )
        

    def encode(self, x):
        return {'cond': self.encoder(x)}
    
    
    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                
                model_kwargs =None):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """

            
        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)
        
        
        if cond is None:
            tmp = self.encode(x_start)
            cond = tmp['cond']
        
        if t is not None:
            # sinusoidal position embeddings \phi(t)
            _t_emb = utils.timestep_embedding(t)
        else:
            # when training only autoenc
            _t_emb = None
        
        # two types of embedding: time + cond, residual 
        res = self.time_embed(time_emb=_t_emb, cond=cond)
        
        enc_time_emb = mid_time_emb = dec_time_emb = res['time_emb']
        enc_cond_emb = mid_cond_emb = dec_cond_emb = res['emb']
        style = style or res['style']


        h = x.type(torch.float32)

        #print(h.shape)
        #print(enc_time_emb.shape)
        #print(enc_cond_emb.shape)
        pred = self.mlp(torch.concat([h,enc_time_emb,enc_cond_emb],axis=1))
        
        out ={"pred":pred,'cond':cond}
        
        return out

    

