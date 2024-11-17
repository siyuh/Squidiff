import argparse
import inspect

from . import diffusion
from .respace import SpacedDiffusion, space_timesteps
from .MLPModel import MLPModel, EncoderMLPModel

NUM_CLASSES = 4


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        gene_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for training.
    """
    res = dict(
        output_dim=1000,
        num_layers=3,
        gene_size=1000,
        num_channels=128,
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        use_fp16=False,
        use_encoder=False
    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    gene_size,
    num_layers,
    output_dim,
    class_cond,
    learn_sigma,
    num_channels,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    use_fp16,
    use_encoder,
):
    model = create_model(
        gene_size,
        num_layers,
        output_dim,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_fp16=use_fp16,
        use_encoder=use_encoder
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_encoder=use_encoder
    )
    return model, diffusion


def create_model(
    gene_size,
    num_layers,
    output_dim,
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=False,
    dropout=0,
    use_fp16=False,
    use_encoder=False,
):

    return MLPModel(
        gene_size  = gene_size,
        output_dim = output_dim,
        num_layers = num_layers,
        dropout=dropout,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_scale_shift_norm=use_scale_shift_norm,
        use_encoder = use_encoder
    )


def create_classifier_and_diffusion(
    gene_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_use_scale_shift_norm,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_encoder,
):
    classifier = create_classifier(
        gene_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_use_scale_shift_norm,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_encoder=use_encoder
    )
    return classifier, diffusion


def create_classifier(
    gene_sizes,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_use_scale_shift_norm,
    classifier_pool,
):


    return EncoderUNetModel(
        gene_size=gene_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        pool=classifier_pool,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    use_fp16,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        use_fp16=use_fp16,
        use_encoder=use_encoder,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_encoder=use_encoder
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    use_fp16,
    use_encoder
):
    _ = small_size  # hack to prevent unused variable


    return SuperResModel(
        gene_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        dropout=dropout,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        use_fp16=use_fp16,
        use_encoder = use_encoder
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    use_encoder = False
):
    print('steps',steps)
    betas = diffusion.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = diffusion.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = diffusion.LossType.RESCALED_MSE
    else:
        loss_type = diffusion.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
        
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            diffusion.ModelMeanType.EPSILON if not predict_xstart else diffusion.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                diffusion.ModelVarType.FIXED_LARGE
                if not sigma_small
                else diffusion.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else diffusion.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        use_encoder=use_encoder
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: args[k] for k in keys if k in args}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
