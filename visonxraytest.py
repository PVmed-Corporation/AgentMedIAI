import torchxrayvision as xrv
import skimage, torch, torchvision
import argparse

def classification(input_path):
    # Prepare the image:
    img = skimage.io.imread(input_path)
    img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
    img = img[None, ...] # Make single color channel

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

    img = transform(img)
    img = torch.from_numpy(img)

    # Load model and process image
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    outputs = model(img[None,...]) # or model.features(img[None,...])

    # Print results
    result = dict(zip(model.pathologies,outputs[0].detach().numpy()))
    # print(result)

    max_key = max(result, key=result.get)
    print(max_key)

if __name__ == "__main__":
    classification('/data/result/yilinyou/00000001_000.png')