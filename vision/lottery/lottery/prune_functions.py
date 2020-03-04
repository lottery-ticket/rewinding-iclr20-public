import numpy as np
import random
import re
from . import amc_resnet50_dict, l1_filter_pruning

_resnet56_A_skip_layers = [16, 20, 38, 54]
_odd_skip_layers = list(range(1, 57, 2))
_resnet56_B_skip_layers = [16, 18, 20, 34, 38, 54]
_resnet56_A_prune_prob = [0.1, 0.1, 0.1]
_resnet56_B_prune_prob = [0.6, 0.3, 0.1]

def _prune_resnet_generic(name, weight, mask, prune_percent):
    real_weights = mask * weight
    n_filters = weight.shape[-1]
    norms = np.linalg.norm(real_weights.reshape(-1, n_filters), ord=1, axis=0)
    smallest_filters = np.argsort(norms)

    zero_mask = np.zeros_like(mask[..., 0])
    nnz_filters = sum(not np.allclose(mask[..., f], zero_mask) for f in range(n_filters))
    n_filters_to_prune = nnz_filters * prune_percent
    n_filters_to_prune = int(n_filters_to_prune) + (random.random() < (n_filters_to_prune - int(n_filters_to_prune)))

    filters_to_prune = tuple(smallest_filters[n_filters - nnz_filters:n_filters - nnz_filters+n_filters_to_prune])
    print('{}: pruning {}/({}/{}) filters ({} weight)'.format(
        name,
        len(filters_to_prune),
        nnz_filters,
        n_filters,
        np.sum(norms[..., filters_to_prune]))
    )

    mask[..., filters_to_prune] = 0

    return mask

def prune_structured_generic(d, i=1):
    def _prune_structured(names, weights, masks):
        random.seed(0)
        return [
            _prune_resnet_generic(name, weight, mask, 1-(1-d[name])**i) if name in d else mask
            for (name, weight, mask) in zip(names, weights, masks)
        ]
    return _prune_structured

prune_vgg16_A = prune_structured_generic(l1_filter_pruning.vgg_16_A)
prune_vgg16_nofc_A = prune_structured_generic(l1_filter_pruning.vgg_16_A)
prune_resnet56_A = prune_structured_generic(l1_filter_pruning.resnet_56_A)
prune_resnet56_B = prune_structured_generic(l1_filter_pruning.resnet_56_B)
prune_resnet110_A = prune_structured_generic(l1_filter_pruning.resnet_110_A)
prune_resnet110_B = prune_structured_generic(l1_filter_pruning.resnet_110_B)
prune_resnet34_A = prune_structured_generic(l1_filter_pruning.resnet_34_A)
prune_resnet34_B = prune_structured_generic(l1_filter_pruning.resnet_34_B)

for i in range(1, 10):
    for (net, size, level) in [
            ('vgg', 16, 'A'),
            ('vgg', '16_nofc', 'A'),
            ('resnet', 56, 'A'),
            ('resnet', 56, 'B'),
            ('resnet', 110, 'A'),
            ('resnet', 110, 'B'),
            ('resnet', 34, 'A'),
            ('resnet', 34, 'B'),
    ]:
        globals()['prune_{}{}_{}_{}'.format(net, size, level, i)] = prune_structured_generic(
            getattr(l1_filter_pruning, '{}_{}_{}'.format(net, size, level)), i=i)

def prune_amc_generic(iteration):
    def pruner(names, weights, masks):
        def do_prune(name, weights, masks):
            percent = iteration[name]
            weights = weights * masks
            weights = np.abs(weights)
            threshold = np.percentile(weights.ravel(), (1 - percent) * 100)
            old_frac = masks.sum() / np.prod(masks.shape)
            masks[weights < threshold] = 0

            print('{}: {}->{}'.format(
                name,
                old_frac,
                masks.sum() / np.prod(masks.shape),
            ), flush=True)

            return masks
        return [
            do_prune(name, weight, mask)
            for (name, weight, mask) in zip(names, weights, masks)
        ]
    return pruner

prune_amc_first = prune_amc_generic(amc_resnet50_dict.first)
prune_amc_second = prune_amc_generic(amc_resnet50_dict.second)
prune_amc_third = prune_amc_generic(amc_resnet50_dict.third)
prune_amc_fourth = prune_amc_generic(amc_resnet50_dict.fourth)

def prune_resnet20_structured_discovered(names, weights, masks):
    return [
        _prune_resnet_generic(name, weight, mask, [], {
            1: 0.023901808785529666,
            2: 0.06557352228682178,
            3: 0.06280281007751937,
            4: 0.06471051356589153,
            5: 0.06693112080103347,
            6: 0.0596737726098191,
            7: 0.07143289728682178,
            8: 0.06064528827519394,
            9: 0.068210493378553,
            10: 0.07919997577519378,
            11: 0.08791207404715745,
            12: 0.0779635012919897,
            13: 0.09580784681847543,
            14: 0.07624757751937983,
            15: 0.08246653948643412,
            16: 0.08787958504925708,
            17: 0.10326643754037468,
            18: 0.10140888697109168,
            19: 0.1836707142280362,
        }, 20)
        for (name, weight, mask) in zip(names, weights, masks)
    ]

def prune_global_x(names, weights, masks, x, allowed_pruners=['conv'], fc=False, nofc=False):
    if fc:
        allowed_pruners.append('fc')
    legal_idxs = {i for (i, n) in enumerate(names) if any(pruner in n for pruner in allowed_pruners)}

    legal_names = [n for (i, n) in enumerate(names) if i in legal_idxs]
    legal_weights = [w for (i, w) in enumerate(weights) if i in legal_idxs]
    legal_masks = [m for (i, m) in enumerate(masks) if i in legal_idxs]

    ravel_weights = np.abs(np.concatenate(list(map(np.ravel, legal_weights))))
    ravel_masks = np.concatenate(list(map(np.ravel, legal_masks)))

    threshold = np.percentile(ravel_weights[ravel_masks > 0.5], x)

    for (name, weight, mask) in zip(legal_names, legal_weights, legal_masks):
        old_frac = mask.sum() / np.prod(mask.shape)
        mask[np.abs(weight) < threshold] = 0
        print('{}: {}->{}'.format(
            name,
            old_frac,
            mask.sum() / np.prod(mask.shape),
        ), flush=True)
    return masks


def zweight(weight):
    weight = np.abs(weight)
    if len(weight.shape) == 4:
        weight = weight - weight.mean(axis=-1)[..., np.newaxis]
        weight = weight / weight.std(axis=-1)[..., np.newaxis]
        return weight
    elif len(weight.shape) == 2:
        weight = weight - weight.mean()
        return weight / weight.std()
    else:
        raise ValueError()


def zprune_global_x(names, weights, masks, x, allowed_pruners=['conv'], fc=False, nofc=False):
    if fc:
        allowed_pruners.append('fc')
    legal_idxs = {i for (i, n) in enumerate(names) if any(pruner in n for pruner in allowed_pruners)}

    legal_names = [n for (i, n) in enumerate(names) if i in legal_idxs]
    legal_zweights = [zweight(w) for (i, w) in enumerate(weights) if i in legal_idxs]
    legal_masks = [m for (i, m) in enumerate(masks) if i in legal_idxs]

    ravel_weights = np.abs(np.concatenate(list(map(np.ravel, legal_zweights))))
    ravel_masks = np.concatenate(list(map(np.ravel, legal_masks)))

    threshold = np.percentile(ravel_weights[ravel_masks > 0.5], x)

    for (name, weight, mask) in zip(legal_names, legal_zweights, legal_masks):
        old_frac = mask.sum() / np.prod(mask.shape)
        mask[np.abs(weight) < threshold] = 0
        print('{}: {}->{}'.format(
            name,
            old_frac,
            mask.sum() / np.prod(mask.shape),
        ), flush=True)
    return masks


def prune_to_global_x(names, weights, masks, x, allowed_pruners=['conv']):
    legal_idxs = {i for (i, n) in enumerate(names) if any(pruner in n for pruner in allowed_pruners)}

    legal_names = [n for (i, n) in enumerate(names) if i in legal_idxs]
    legal_weights = [w for (i, w) in enumerate(weights) if i in legal_idxs]
    legal_masks = [m for (i, m) in enumerate(masks) if i in legal_idxs]

    ravel_weights = np.abs(np.concatenate(list(map(np.ravel, legal_weights))))
    ravel_masks = np.concatenate(list(map(np.ravel, legal_masks)))
    threshold = np.percentile(ravel_masks * ravel_weights, x)

    for (name, weight, mask) in zip(legal_names, legal_weights, legal_masks):
        old_frac = mask.sum() / np.prod(mask.shape)
        mask[np.abs(weight) < threshold] = 0
        print('{}: {}->{}'.format(
            name,
            old_frac,
            mask.sum() / np.prod(mask.shape),
        ), flush=True)
    return masks

def prune_all_to_global_x(names, weights, masks, x):
    if len(masks) == 0:
        return masks

    concat = np.zeros(sum(np.prod(w.shape) for w in weights), dtype=weights[0].dtype)
    i = 0
    for (name, weight, mask) in zip(names, weights, masks):
        d = np.prod(weight.shape)
        np.abs(np.ravel(weight), out=concat[i:i+d])
        concat[i:i+d] *= np.ravel(mask)
        i += d

    threshold = np.percentile(concat, 100 - x)

    for (name, weight, mask) in zip(names, weights, masks):
        old_frac = mask.sum() / np.prod(mask.shape)
        mask[np.abs(weight) < threshold] = 0
        print('{}: {:.3f}->{:.3f}'.format(
            name,
            old_frac,
            mask.sum() / np.prod(mask.shape),
        ), flush=True)
    return masks

def zprune_all_to_global_x(names, weights, masks, x):
    if len(masks) == 0:
        return masks

    concat = np.zeros(sum(np.prod(w.shape) for w in weights), dtype=weights[0].dtype)
    i = 0
    for (name, weight, mask) in zip(names, weights, masks):
        d = np.prod(weight.shape)
        np.abs(np.ravel(zweight(weight)), out=concat[i:i+d])
        concat[i:i+d] *= np.ravel(mask)
        i += d

    threshold = np.percentile(concat, 100 - x)

    for (name, weight, mask) in zip(names, weights, masks):
        old_frac = mask.sum() / np.prod(mask.shape)
        mask[np.abs(zweight(weight)) < threshold] = 0
        print('{}: {:.3f}->{:.3f}'.format(
            name,
            old_frac,
            mask.sum() / np.prod(mask.shape),
        ), flush=True)
    return masks


def get_prune_function_by_name(name):
    perc_re = r'(?P<percent>\d+(\.\d*)?)'
    regexp_map = {
        r'prune_global(?:_(?:{}|(?P<fc>fc)|(?P<nofc>nofc)))*'.format(perc_re): prune_global_x,
        r'zprune_global(?:_(?:{}|(?P<fc>fc)|(?P<nofc>nofc)))*'.format(perc_re): zprune_global_x,
        r'prune_to_global_{}'.format(perc_re): prune_to_global_x,
        r'prune_all_to_global_{}'.format(perc_re): prune_all_to_global_x,
        r'zprune_all_to_global_{}'.format(perc_re): zprune_all_to_global_x,
    }
    regexp_map = {re.compile(k): v for (k, v) in regexp_map.items()}
    for (regexp, func) in regexp_map.items():
        match = regexp.match(name)
        if match is None:
            continue

        percentage = float(match.group('percent'))
        kwargs = {k: v for (k, v) in match.groupdict().items() if v is not None and k != 'percent'}
        def prune(names, weights, masks):
            return func(names, weights, masks, percentage, **kwargs)
        return prune

    return globals()[name]
