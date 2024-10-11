# Autonomous Steering with Transformer-Based Motion Prediction

This repository contains the code and resources for our research project submitted to Courant Computer Vision Course, **Autonomous Steering with Transformer-Based Motion Prediction**. The project explores the use of Transformer architectures to predict steering angles in autonomous vehicles, leveraging the DAVE-2 dataset and methodologies inspired by the Multi-Granular Transformer (MGTR) and Motion Transformer (MTR).

## Project Overview

Our model introduces a novel approach for predicting steering angles in autonomous driving systems. By combining the DAVE-2 system with Transformer-based architectures, we present a framework that captures spatial relationships between input pixels for improved motion prediction accuracy.

### Model Architecture

1. **Input Layer**: Images are resized to 40x40 pixels and fed into the model.
2. **Reshape Layer**: The images are reshaped for further processing.
3. **Positional Encoding**: Adds spatial information to the input, enabling the model to learn geometric structures.
4. **Transformer Encoder Blocks**: Multiple encoder blocks with multi-head attention and feed-forward neural networks capture features for steering prediction.
5. **Dense Output Layer**: Maps learned features to the steering angle predictions.

### Performance

- **Accuracy**: 95.5% on the test dataset, based on the validation loss of 4.55% after training.
- **Comparison**: While our model achieves promising results, Dave2’s state-of-the-art model has a 98% accuracy. However, our work represents an advancement in using transformers for motion prediction in autonomous driving.

## Installation

### Prerequisites

To run the project, you need to have Python 3.x installed along with the following dependencies:

```numpy matplotlib opencv-python (cv2) keras h5py scipy tensorflow ```

You can install the required packages using the `requirements.txt` file:

```bash pip install -r requirements.txt```

### Running the Code

1. **Preprocess the Data**:
   Run the `LoadData.py` script to load and preprocess the DAVE-2 dataset.

   ```bash python LoadData.py```

2. **Train the Model**:
   Train the Transformer-based model by running the `TrainModel.py` script.

   ```bash python TrainModel.py```

3. **Run the Driving Application**:
   After training the model, simulate driving using the steering predictions with `DriveApp.py`.

   ```bash python DriveApp.py```

## Model Evaluation

During the training, we evaluate the model based on the loss function and accuracy on a validation dataset. The final accuracy achieved is 95.5%, which shows the effectiveness of integrating Transformer architectures for motion prediction.

## Future Work

- We plan to experiment with more complex transformer architectures and larger datasets, such as Waymo.
- Investigate optimizing memory usage and computational resources for handling large datasets like Waymo (1TB).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank the contributors of the DAVE-2 system and researchers behind MTR and MGTR methodologies, whose work greatly influenced this project.

