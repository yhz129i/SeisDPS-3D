import os
import argparse
import random

import numpy as np
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from SeisDPS import dist_util
import torchvision.transforms as transforms


from SeisDPS.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def main(conf: conf_mgt.Default_Conf):
    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))


    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    data_test  = np.load('data/data_30hz_test_noise_18dB.npy').astype(np.float32)
    max_val = np.max(np.abs(data_test)) 
    data_test = (data_test) / max_val 
    data_test = th.tensor(data_test, dtype=th.float32).unsqueeze(0).unsqueeze(0).to(device)
    batch_size = 1
    model_kwargs = {"gt": data_test}


    if conf.cond_y is not None:
        classes = th.ones(batch_size, dtype=th.long, device=device)
        model_kwargs["y"] = classes * conf.cond_y
    else:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=device
        )
        model_kwargs["y"] = classes

   

    sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

    result = sample_fn(
            model_fn,
            (batch_size, 1, 320, 388), 
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )

    n_sample = result.get('sample')
    n_sample = n_sample.cpu().numpy()
    n_sample = n_sample.squeeze()
    np.save("result/test.npy", n_sample)
    print(n_sample.shape)


    print("sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default="confs/SeisDPS.yml")
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
