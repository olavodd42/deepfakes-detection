import matplotlib.pyplot as plt

def plot_first_image(train_dataset):
    image, label = train_dataset[0]
    image = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
    image = image.clip(0, 1)

    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis("off")