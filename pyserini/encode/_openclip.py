import torch
import numpy as np
import open_clip
from ._clip import load_pil_image
from sklearn.preprocessing import normalize
from pyserini.encode import DocumentEncoder, QueryEncoder


class BaseOpenClipEncoder:
    def __init__(self, model_name: str, device: str='cuda:0', l2_norm: bool=True):
        model_tag, pretrained = model_name.split(':')
        model, _, processor = open_clip.create_model_and_transforms(model_tag, pretrained)
        self.model = model
        self.model.to(device)
        self.image_processor = processor
        self.tokenizer = open_clip.get_tokenizer(model_tag)
        self.device = device
        self.l2_norm = l2_norm

    def normalize_embeddings(self, embeddings):
        """Apply L2 normalization to embeddings if required."""
        return normalize(embeddings, axis=1, norm='l2') if self.l2_norm else embeddings

class OpenClipImageEncoder(BaseOpenClipEncoder):
    
    def encode(self, paths, **kwargs):
        processed_images = []

        if isinstance(paths, str):
            paths = [paths]

        for image in paths:
            try:
                img = load_pil_image(image)
                processed_images.append(self.image_processor(img))
            except Exception as e:
                print(f"Error loading image {image}: {e}")

        inputs = torch.tensor(np.stack(processed_images)).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(inputs)

        embeddings = image_features.detach().cpu().numpy()
        return self.normalize_embeddings(embeddings)

class OpenClipTextEncoder(BaseOpenClipEncoder):
    
    def __init__(self, model_name, device='cuda:0', l2_norm=False, prefix=None):
        super().__init__(model_name, device, l2_norm)
        self.prefix = prefix
    
    def encode(self, texts, max_length=77, **kwargs):

        if isinstance(texts, str):
            texts = [texts]
            
        if self.prefix:
            texts = [f"{self.prefix} {text}" for text in texts]

        inputs = self.tokenizer.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(inputs)

        embeddings = text_features.detach().cpu().numpy()
        return self.normalize_embeddings(embeddings)

class OpenClipDocumentEncoder(DocumentEncoder):
    """Encodes documents using a CLIP model, supporting both images and texts."""
    def __init__(self, model_name, device='cuda:0', l2_norm=False, prefix=None, multimodal=False):
        super().__init__()
        self.encoder = OpenClipImageEncoder(model_name, device, l2_norm) if multimodal else OpenClipTextEncoder(model_name, device, l2_norm, prefix)

    def encode(self, *args, **kwargs):
        return self.encoder.encode(*args, **kwargs)


class OpenClipEncoder(QueryEncoder):
    """Encodes queries using a CLIP model, supporting both images and texts."""
    def __init__(self, model_name, device='cuda:0', l2_norm=False, prefix=None, multimodal=False):
        super().__init__()
        self.encoder = OpenClipImageEncoder(model_name, device, l2_norm) if multimodal else OpenClipTextEncoder(model_name, device, l2_norm, prefix)

    def encode(self, *args, **kwargs):
        return self.encoder.encode(*args, **kwargs)