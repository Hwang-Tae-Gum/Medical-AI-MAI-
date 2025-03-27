# Medical-AI-MAI

## Description
The provided code is designed for a deep learning model that predicts gene expression levels from H&E stained tissue images. This model was developed as part of the Medical AI (MAI) Competition, which aims to leverage AI technology to predict gene expression from H&E stained tissue images. The competition focuses on developing AI models that can efficiently analyze complex medical data and provide valuable insights into gene expression profiles. This model utilizes a pre-trained EfficientNet B0 architecture, fine-tuned for regression tasks to predict the expression levels of 3467 genes. The dataset consists of 6992 training images and 2277 test images, each associated with gene expression data.

## Model Architecture
The model architecture is based on EfficientNet B0, a pre-trained convolutional neural network (CNN) known for its efficiency and performance. The classifier layer is modified to output 3467 features, corresponding to the number of genes in the dataset. The model uses Mean Squared Error (MSE) as the loss function, and the Adam optimizer is set with a learning rate of 
$10^{-4}$

```python
class GeneExpressionModel(nn.Module):
    def __init__(self):
        super(GeneExpressionModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, 3467)

    def forward(self, x):
        return self.model(x)
```

## Dataset Information
Training Dataset: Contains 6992 H&E stained tissue images, each associated with gene expression data for 3467 genes.

Test Dataset: Comprises 2277 images for evaluation purposes.

Data Format:

train.csv: Includes sample IDs, image paths, and gene expression levels.

test.csv: Contains sample IDs and image paths.

sample_submission.csv: Provides a template for submission with predicted gene expression levels.

## Improvements & Future Updates
Data Augmentation: Implementing additional data augmentation techniques (e.g., rotation, flipping) could enhance model robustness and performance.

Hyperparameter Tuning: Conducting grid search or using hyperparameter optimization tools to find optimal learning rates, batch sizes, and epochs.

Model Ensemble: Combining predictions from multiple models or architectures to improve overall accuracy.

Transfer Learning: Exploring other pre-trained models like ResNet or DenseNet to compare performance.

## Conclusion
Deep learning-based prediction of gene expression from tissue images is a promising approach. Utilizing pre-trained models to analyze complex biological data is efficient, and future research should focus on exploring various methods to improve model performance. This can enhance predictive accuracy and open new possibilities for biological data analysis.
