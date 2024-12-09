# Flood Segmentation Project

## Overview
This project implements a multi-class flood segmentation model using the DeepLabv3+ architecture. The primary goal is to accurately segment flood-affected areas in satellite or aerial images. The model is trained on a custom dataset and evaluated using metrics like IoU (Intersection over Union).

## Features
- **DeepLabv3+ Architecture**: Utilized for advanced image segmentation.
- **Multi-class Segmentation**: Handles multiple classes in the dataset for precise area identification.
- **Custom Dataset Processing**: Includes data pre-processing, mask handling, and augmentation.
- **Training and Evaluation**: Implements a robust pipeline for training and testing the model.
- **Class Imbalance Handling**: Introduced techniques to address class imbalances for better performance.

## Dataset
The dataset consists of:
- **Images**: Flood-affected areas captured in RGB.
- **Masks**: Ground truth masks with multiple classes indicating different regions.

### Pre-processing Steps
1. Resized images and masks to 256x256 dimensions.
2. Converted masks for multi-class segmentation handling.
3. Applied data augmentations to increase dataset diversity.

## Model Training
- **Architecture**: DeepLabv3+ with a ResNet backbone.
- **Loss Function**: CrossEntropyLoss was used for multi-class segmentation.
- **Optimizer**: Adam optimizer with a learning rate scheduler.
- **Metrics**: Evaluated using Mean IoU and validation loss.
- **Epochs**: Trained for 15 epochs with a batch size of 4.

## Results
- **Train Loss**: Progressively reduced over epochs.
- **Validation Loss**: Monitored to ensure the model generalizes well.
- **IoU Score**: Improved after addressing class imbalance and optimizing hyperparameters.

## How to Run
1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/flood-segmentation.git
    cd flood-segmentation
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model:
    ```bash
    python flood_seg.py
    ```
4. Save the trained model:
    ```python
    torch.save(model.state_dict(), "segmentation_model.pth")
    ```

## File Structure
- 📁 `Data/` : Contains the images and masks for training and validation.
- 📝 `flood_seg.py` : Script containing all the code for training, evaluation, and utilities.
- 📝 `segmentation_model.pth` : Saved model weights after training.

## Dependencies
- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- matplotlib

## Future Improvements 🚀
- Fine-tuning the model on larger datasets.
- Exploring alternative segmentation architectures.
- Incorporating real-time inference capabilities.

## Acknowledgments
- Special thanks to the PyTorch and OpenCV communities for their amazing tools and resources.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

