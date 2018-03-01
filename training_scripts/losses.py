from keras import backend as K

def cosine_distance(x, y, vects_are_normalized=True):
    # L2 normalize vectors before dot product
    if not vects_are_normalized:
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
    similarity = K.batch_dot(x, y, axes=1)
    distance = K.constant(1) - similarity
    # Distance goes from 0 to 2 in theory, but from 0 to 1 if x and y are both
    # positive (which is the case after ReLU activation).
    return K.squeeze(distance, axis=-1)

def triplet_loss(inputs, margin=0.5):
    """calculate triplet loss"""
    anchor, same, diff = inputs
    same_dist = cosine_distance(anchor, same, vects_are_normalized=False)
    diff_dist = cosine_distance(anchor, diff, vects_are_normalized=False)

    loss = K.maximum(K.constant(0), margin + same_dist/2 - diff_dist/2)

    return K.mean(loss)

def triplet_loss_no_mean(inputs, margin=0.5):
    """calculate triplet loss"""
    anchor, same, diff = inputs
    same_dist = cosine_distance(anchor, same, vects_are_normalized=False)
    diff_dist = cosine_distance(anchor, diff, vects_are_normalized=False)

    loss = K.maximum(K.constant(0), margin + same_dist/2 - diff_dist/2)

    return loss