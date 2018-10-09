import numpy as np
import numpy.random as rng
import cv2

def augment(img, do_training=True):
    # img = imgproc.resize_preserve_aspect_ratio(img, config.image_shape)

    if do_training:
        # data augmentation from fb.resnet.torch
        # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

        def scale(img, size):
            s = size / min(img.shape[0], img.shape[1])
            h, w = int(round(img.shape[0] * s)), int(round(img.shape[1] * s))
            return cv2.resize(img, (w, h))

        def center_crop(img, shape):
            h, w = img.shape[:2]
            sx, sy = (w - shape[1]) // 2, (h - shape[0]) // 2
            img = img[sy:sy + shape[0], sx:sx + shape[1]]
            return img

        def random_sized_crop(img):
            NR_REPEAT = 1

            h, w = img.shape[:2]
            area = h * w
            ar = [7. / 8, 8. / 7]
            for i in range(NR_REPEAT):
                target_area = rng.uniform(0.5, 1.0) * area
                target_ar = rng.choice(ar)
                nw = int(round((target_area * target_ar) ** 0.5))
                nh = int(round((target_area / target_ar) ** 0.5))

                if rng.rand() < 0.5:
                    nh, nw = nw, nh

                if nh <= h and nw <= w:
                    sx, sy = rng.randint(w - nw + 1), rng.randint(h - nh + 1)
                    img = img[sy:sy + nh, sx:sx + nw]
                    return cv2.resize(img, config.image_shape[::-1])

            size = min(config.image_shape[0], config.image_shape[1])
            return center_crop(scale(img, size), config.image_shape)

        def grayscale(img):
            return img

        def brightness_aug(img, val):
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha

            return img

        def contrast_aug(img, val):
            gs = grayscale(img)
            gs[:] = gs.mean()
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img

        def saturation_aug(img, val):
            gs = grayscale(img)
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img

        def color_jitter(img, brightness, contrast, saturation):
            augs = [(brightness_aug, brightness),
                    (contrast_aug, contrast),
                    (saturation_aug, saturation)]
            rng.shuffle(augs)

            for aug, val in augs:
                img = aug(img, val)

            return img

        def lighting(img, std):
            eigval = list2nparray([0.2175, 0.0188, 0.0045])
            eigvec = list2nparray([
                [-0.5836, -0.6948,  0.4203],
                [-0.5808, -0.0045, -0.8140],
                [-0.5675, 0.7192, 0.4009],
            ])
            if std == 0:
                return img

            alpha = rng.randn(3) * std
            bgr = eigvec * alpha.reshape(1, 3) * eigval.reshape(1, 3)
            bgr = bgr.sum(axis=1).reshape(1, 1, 3)
            img = img + bgr

            return img

        def horizontal_flip(img, prob):
            if rng.rand() < prob:
                return img[:, ::-1]
            return img

        def warp_perspective(img):
            c = (
                ((-50, 50), (-10, 10)),
                ((-50, 50), (-10, 10)),
                ((-50, 50), (-10, 10)),
                ((-50, 50), (-10, 10))
            )
            mat = imgaug.get_random_perspective_transform_mat(
                rng, c, config.image_shape)
            return cv2.warpPerspective(img, mat, config.image_shape)

        num = img.shape[0]
        for i in range(num):
            img[i] = color_jitter(img[i], brightness=0.1, contrast=0., saturation=0.)
        #img = random_sized_crop(img)
        #img = lighting(img, 0.1)
        #img = brightness_aug(img, 0.1)
        #img = horizontal_flip(img, 0.1)
        # img = warp_perspective(img)

    return img
