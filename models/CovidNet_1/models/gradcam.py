import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image):

        self.model.eval()

        output = self.model(image)

        predicted = output.argmax(dim=1)

        score = output[:, predicted]

        self.model.zero_grad()

        score.backward()

        gradients = self.gradients.mean(dim=(2,3), keepdim=True)

        cam = (gradients * self.activations).sum(dim=1)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        cam -= cam.min()

        cam /= cam.max() + 1e-8

        confidence = torch.softmax(output,1)[0,predicted].item()

        return cam.cpu().numpy(), predicted.item(), confidence


def show_gradcam(image, heatmap, alpha=0.35):

    image = image.squeeze().cpu().numpy()

    image -= image.min()
    image /= image.max()

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    heatmap = heatmap.astype(np.float32) / 255

    image_rgb = np.stack([image] * 3, axis=-1)

    overlay = (1 - alpha) * image_rgb + alpha * heatmap

    overlay = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Radiographie")

    ax[1].imshow(heatmap)
    ax[1].set_title("Grad-CAM")

    ax[2].imshow(overlay)
    ax[2].set_title("Superposition")

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()