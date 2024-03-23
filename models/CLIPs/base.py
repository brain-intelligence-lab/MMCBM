import torch


class BaseCLIP:
    def _encode_image(self, image):
        raise NotImplementedError

    def encode_image(self, modality=None, image=None):
        if isinstance(image, dict):
            result = []
            for k, v in image.items():
                result.append(self.encode_image(image=v))
            return torch.max_pool1d(torch.concat(result, dim=1).permute(0, 2, 1), kernel_size=7).squeeze(-1)
        if image.ndim == 5:
            B = image.shape[0]
            image = image.reshape(-1, *image.shape[2:])
            embedding = self._encode_image(image)
            embedding = embedding.reshape(B, -1, embedding.shape[-1])
            return embedding
        return self._encode_image(image)
