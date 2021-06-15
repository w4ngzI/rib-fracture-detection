def dice(logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(-1)
        m2 = targets.view(-1)
        intersection = m1 * m2
        score = 2.0 * (intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
        return score


def recall(x, y, thresh=0.1):
    x = x.sigmoid()
    tp = (((x * y) > thresh).flatten(1).sum(1) > 0).sum()
    rc = tp / (((y > 0).flatten(1).sum(1) > 0).sum() + 1e-8)
    return rc


def accuracy(x, y, thresh=0.5):
    x = x.sigmoid()
    ac = ((x > thresh) == (y > 0)).float().mean()
    return ac


def precision(x, y, thresh=0.1):
    x = x.sigmoid()
    tp = (((x * y) > thresh).flatten(1).sum(1) > 0).sum()
    pc = tp / (((x > thresh).flatten(1).sum(1) > 0).sum() + 1e-8)
    return pc


def fbeta_score(x, y, beta=1, **kwargs):
    rc = recall(x, y, **kwargs)
    pc = precision(x, y, **kwargs)
    fs = (1 + beta ** 2) * pc * rc / (beta ** 2 * pc + rc + 1e-8)
    return fs