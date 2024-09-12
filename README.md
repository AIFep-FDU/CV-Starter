# CV-Starter: Image Classification Using CNN

```bash
git clone https://github.com/AIFep-FDU/CV-Starter.git
```

### Environment Setup:
1. **Python Environment:** Ensure you have Python installed (preferably Python 3.x).
2. **Dependencies:** Install required dependencies using pip:

```bash
pip install torch torchvision pytorchsummary numpy matplotlib tqdm
```
3. **Colab:** no need to pip


### Code Flow:

1. **Model Definition:**
- The code defines a Convolutional Neural Network (CNN) model for image classification.
- ResNet-18 architecture is used as an example, but you can analyze and discuss other CNN architectures as well.

2. **Training and Testing:**
- The code includes functions for training and testing the CNN model.
- It utilizes datasets provided by PyTorch and DataLoader for efficient data handling.

3. **Data Prepare Augmentation:**
- Create a dataset class to read VOC2007 annotations and get its sub-images for classification.
- You need to design data augmentation techniques to improve accuracy. This may include rotations, flips, or other transformations.

4. **Optimizer and Learning Rate Scheduler:**
- Stochastic Gradient Descent (SGD) optimizer with momentum is used.
- Learning rate scheduling is implemented using ReduceLROnPlateau scheduler.

5. **Experimental Requirements:**
    - Complete the model training and validation code.
    - Crop the image directly and design data augmentation (including normalization) techniques to enhance accuracy.
    - Analyze and discuss the design of other optimizers and learning rate schedules.
    - Analyze and discuss different CNN or other model (based on timm) architectures.

### Filling in the Blanks:
1. **Custom Dataset Class:**
   - Complete the `__getitem__` method to crop the image based on the bounding box and apply transformations.

2. **Model Architecture:**
   - Adjust the ResNet model architecture by modifying the `BasicBlock` and `ResNet` classes.
   - Consider using other network designs or directly using models from libraries like `timm` for comparison.

3. **Optimizer and Scheduler Configuration:**
   - Modify the optimizer and learning rate scheduler settings in the `main` function to find the optimal parameters for training.
   - Fill in the code for updating the learning rate scheduler after each epoch.

4. **Normalization Values:**
   - Adjust the normalization values in the `train_transforms` and `test_transforms` to better suit the VOC2007 dataset.
   - Obtain these values through statistical analysis or use predefined values from similar datasets.

5. **Loss Function and Backpropagation:**
   - In the `model_training` function, fill in the code for calculating the loss using a suitable loss function and performing backpropagation.

6. **Model Output and Loss Calculation in Testing:**
   - In the `model_testing` function, fill in the code for obtaining the model's output and calculating the loss during the testing phase.

7. **Data Augmentation:**
   - Design image augmentation techniques inside the `train_transforms` to improve model performance.
   - Consider including transformations like random cropping, flipping, rotation, and color jitter.


### Running the Script:
- Execute the script `main.py` from the command line: `python main.py`.
- Adjust parameters and configurations as needed for experimentation.
- For colab, just click the "run" button.


### Submission:
After training your model, you can execute the `python infer.py` script to generate predictions for the 'test_1k' dataset. Simply unzip the output and submit it directly to the [CodaLab Competition](https://codalab.lisn.upsaclay.fr/competitions/20172). 

Submit the code along with an experimental report (one page, No involution) to the e-learning platform.

### Note:
- Modify and adjust the arguments, configurations, and experimental setup as necessary.
- Provide detailed explanations in the experimental report, covering code implementation, experimental setup, results, analysis, and discussions based on the specified requirements.
