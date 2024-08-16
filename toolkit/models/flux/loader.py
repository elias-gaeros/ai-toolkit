import torch
from safetensors.torch import safe_open
import accelerate

from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from transformers.models.clip.configuration_clip import CLIPConfig
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux, FluxParams

from .bnb import load_quantized_statedict
from .ops import get_ops_namespace
from .t5 import T5
from .clip import CLIPTextModel


StateDict = dict[str, torch.Tensor]


class Configs:
    def __init__(self, config_path="black-forest-labs/FLUX.1-schnell"):
        self.config_path = config_path

    def get_vae_config(self) -> AutoEncoderParams:
        return AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )

    def get_flux_config(self, state_dict: StateDict) -> FluxParams:
        return FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=any(k.endswith("qkv.bias") for k in state_dict),
            guidance_embeds="guidance_in.in_layer.weight" in state_dict,
        )

    def get_t5_config(self) -> tuple[T5Config, T5TokenizerFast]:
        t5_config = T5Config.from_pretrained(
            self.config_path, subfolder="text_encoder_2"
        )
        t5_tokenizer = T5TokenizerFast.from_pretrained(
            self.config_path, subfolder="tokenizer_2"
        )
        return t5_config, t5_tokenizer

    def get_clip_config(self) -> tuple[CLIPConfig, CLIPTokenizer]:
        clip_config = CLIPConfig.from_pretrained(
            self.config_path, subfolder="text_encoder"
        )
        clip_tokenizer = CLIPTokenizer.from_pretrained(
            self.config_path, subfolder="tokenizer"
        )
        return clip_config, clip_tokenizer


class CombinedCheckpoint(Configs):
    TRANSFORMER_PREFIX = "model.diffusion_model."
    VAE_PREFIX = "vae."
    CLIP_PREFIX = "text_encoders.clip_l.transformer."
    T5_PREFIX = "text_encoders.t5xxl.transformer."

    def __init__(self, path, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.fd = safe_open(path, framework="pt", device=self.device.type)
        self.keys = set(self.fd.keys())

    def load_prefix(self, prefix) -> dict[str, torch.Tensor]:
        state_dict = dict()
        for k in self.keys:
            if not k.startswith(prefix):
                continue
            tensor = self.fd.get_tensor(k)
            k = k.removeprefix(prefix)
            state_dict[k] = tensor
        return state_dict

    def get_prefix_keys(self, prefix):
        keys = set()
        for k in self.keys:
            if not k.startswith(prefix):
                continue
            keys.add(k)
        return keys

    def load_vae(self):
        params = self.get_vae_config()

        with accelerate.init_empty_weights():
            ae = AutoEncoder(params)
        ae.load_state_dict(self.load_prefix(self.VAE_PREFIX), assign=True)
        return ae

    def load_transformer(self, device=None):
        state_dict = self.load_prefix(self.TRANSFORMER_PREFIX)

        params = self.get_flux_config(state_dict)
        with accelerate.init_empty_weights():
            transformer = Flux(params)
        load_quantized_statedict(transformer, state_dict, device=device or self.device)

        return transformer

    def load_t5(self) -> tuple[T5, T5TokenizerFast]:
        config, tokenizer = self.get_t5_config()
        sd = self.load_prefix(self.T5_PREFIX)
        shared = sd["shared.weight"]

        with accelerate.init_empty_weights():
            t5 = T5(
                config.to_dict(),
                dtype=shared.dtype,
                device=shared.device,
                operations=get_ops_namespace(shared.dtype, shared.device),
            )

        t5.load_state_dict(sd, assign=True)
        t5.requires_grad_(False)
        t5.eval()
        return t5, tokenizer

    def load_clip(self) -> tuple[CLIPTextModel, CLIPTokenizer]:
        config, tokenizer = self.get_clip_config()
        sd = self.load_prefix(self.CLIP_PREFIX)

        embeds = sd["text_model.embeddings.token_embedding.weight"]
        text_projection = sd.get("text_model.text_projection.weight")
        if text_projection is not None and torch.allclose(
            sd["text_projection.weight"],
            torch.eye(
                config.hidden_size,
                device=text_projection.device,
                dtype=text_projection.dtype,
            ),
        ):
            text_projection = None

        with accelerate.init_empty_weights():
            clip = CLIPTextModel(
                config.to_dict(),
                dtype=embeds.dtype,
                device=embeds.device,
                operations=get_ops_namespace(embeds.dtype, embeds.device),
            )

        if text_projection is None:
            clip.text_projection = None

        clip.load_state_dict(sd, assign=True)
        clip.requires_grad_(False)
        clip.eval()
        return clip, tokenizer
