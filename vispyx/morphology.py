import numpy as np

def vpx_pad_image(image, kernel):
    """
    Aplica padding a la imagen en funcion del tamano del kernel
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    return np.pad(image, ((ph, ph), (pw,pw)), mode='reflect')

def vpx_erode(image, kernel=None, iterations=1):
    """
    Erosion binaria personalizada (reduce objetos blancos)
    """
    if kernel is None:
        kernel = np.ones((3,3), dtype=np.uint8)
    img = (image > 0).astype(np.uint8)
    for _ in range(iterations):
        padded = vpx_pad_image(img, kernel)
        output = np.zeros_like(img)

        kh, kw = kernel.shape
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+kh, j:j+kw]
                if np.array_equal(region[kernel == 1], np.ones(np.sum(kernel))):
                    output[i,j] = 1

        img = output
    return img * 255

def vpx_dilate(image, kernel=None, iterations=1):
    """
    Dilatación binaria personalizada (expande objetos blancos).
    """
    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)

    img = (image > 0).astype(np.uint8)
    for _ in range(iterations):
        padded = vpx_pad_image(img, kernel)
        output = np.zeros_like(img)

        kh, kw = kernel.shape
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+kh, j:j+kw]
                if np.any(region[kernel == 1]):
                    output[i, j] = 1

        img = output
    return img * 255 

def vpx_open(image, kernel=None, iterations=1):
    """
    Apertura binaria personalizada: erosión seguida de dilatación.
    """
    eroded = vpx_erode(image, kernel, iterations)
    opened = vpx_dilate(eroded, kernel, iterations)

    return opened

def vpx_close(image, kernel=None, iterations=1):
    """
    Cierre binario personalizado: dilatación seguida de erosión.
    """

    dilated = vpx_dilate(image, kernel, iterations)
    closed = vpx_erode(dilated, kernel, iterations)

    return closed

def vpx_gradient(image, kernel=None, iterations=1):
    """
    Gradiente morfológico personalizado: dilatación menos erosión.
    """
    dilated = vpx_dilate(image, kernel, iterations)
    eroded = vpx_erode(image, kernel, iterations)

    gradient = (dilated.astype(np.int16) - eroded.astype(np.int16)).clip(min=0).astype(np.uint8)

    return gradient
