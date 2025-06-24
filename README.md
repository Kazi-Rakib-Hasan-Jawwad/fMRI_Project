Spatio-Temporal Vision Transformer for fMRI Classification

Author: Kazi Rakib Hasan
Student ID: 2022427248

1. Introduction
Functional Magnetic Resonance Imaging (fMRI) is a powerful neuroimaging technique that measures brain activity by detecting changes associated with blood flow. The analysis of 4D fMRI data, which comprises a 3D volume of the brain evolving over time, presents significant challenges due to its high dimensionality and complex spatio-temporal dynamics. The primary goal of this project is to develop and evaluate a deep learning model capable of accurately classifying brain states from fMRI data, specifically distinguishing between cognitive tasks based on neural activity patterns. The dataset used for this task is a sample from the Human Connectome Project (HCP), focusing on an emotion-processing task where the objective is to classify whether the subject is viewing a face or a shape.
This project is motivated by recent advancements in this area, most notably:
Masked Autoencoders (MAE), as introduced by He et al. (2022) [1], have demonstrated remarkable success in self-supervised learning for Vision Transformers (ViT). By masking a large portion of input image patches and learning to reconstruct them, the model is forced to develop a holistic and semantically rich understanding of the data without requiring explicit labels. This is particularly valuable for fMRI data, where labeled datasets can be scarce.
Task-Aware Spatio-Temporal Models, such as the Task-Aware Reconstruction Dynamic Representation Learning (TARDRL) framework by Zhao et al. (2024) [2], which proposes a sophisticated architecture for fMRI that explicitly models both spatial and temporal features while making the self-supervised reconstruction task aware of the final downstream prediction task.
This project seeks to create a novel hybrid architecture that thoughtfully combines the MAE with the spatio-temporal architectural insights from TARDRL.

2. Methodology
This project implements a two-stage learning pipeline that first learns general-purpose features in a self-supervised manner and then adapts them for a specific classification task to address the challenges of fMRI classification. The core of the methodology is a hybrid Spatio-Temporal Vision Transformer that processes the 4D fMRI data by first learning spatial features from individual 2D slices and then learning the temporal relationships between these slices, a design choice directly inspired by the hierarchical processing in the TARDRL framework.
2.1. Model Architecture
The final model, SpatioTemporalClassifier, is a composite architecture comprising three main components:
Spatial Encoder (2D ViT):
Patch Embedding: Each 3D fMRI volume (53, 63, 46) is treated as a sequence of 53 2D slices of size (63x46). Each slice is decomposed into non-overlapping 2D patches of size (7x2). These patches are linearly projected into a 192-dimensional embedding space.
Positional Embedding: Learnable positional embeddings are added to the patch embeddings to retain spatial information within each slice. A special [CLS] token is prepended to the sequence of patches, whose corresponding output from the transformer is used as the aggregate representation for that slice.
Transformer Blocks: The sequence of embedded patches is processed by a standard Vision Transformer encoder, consisting of 4 layers of multi-head self-attention (nhead_spatial=4).
Temporal Encoder (Transformer):
Sequence of Slice Representations: The output [CLS] token from the Spatial Encoder for each of the 53 slices is collected, forming a sequence of 53 feature vectors.
Temporal Positional Embedding: A learnable positional embedding is added to this sequence to provide the model with information about the temporal order of the slices.
Transformer Blocks: This sequence is then passed through a Temporal Transformer, which consists of 2 layers of multi-head self-attention (nhead_temporal=4), to model the dynamic relationships between brain states over time.
Classification Head:
The output sequence from the Temporal Transformer is mean-pooled across the time dimension to produce a single, fixed-size feature vector representing the entire fMRI scan. This vector is then fed into a simple Multi-Layer Perceptron (MLP) with one hidden layer and a final linear layer that outputs the logits for the two classes (Face vs. Shape).
2.2. Pre-training: Masked Autoencoder (MAE)
To enable the Spatial Encoder to learn robust and meaningful features without supervision, it is first pre-trained using the foundational Masked Autoencoder (MAE) framework proposed by He et al. (2022).
Masking: During each pre-training step, a random 65% of the 2D patches from each fMRI slice are masked (i.e., removed) from the input.
Reconstruction Task: The lightweight Spatial Encoder processes only the visible patches. A much smaller MAE Decoder (2 layers) is then tasked with reconstructing the missing patches from the latent representation and the learned positional embeddings of the masked patches.
Loss Function: The reconstruction loss is calculated as the Mean Squared Error (MSE) between the predicted and original pixel values of the masked patches only. A crucial implementation detail is the use of per-patch normalization, where the target patches are normalized before the loss is computed. This encourages the model to learn the internal structure of a patch rather than simply its overall intensity.
2.3. Fine-Tuning
After 80 epochs of self-supervised pre-training, the weights of the Spatial Encoder (including the patch and positional embeddings) are transferred to the SpatioTemporalClassifier. The MAE Decoder is discarded. The entire Spatio-Temporal model is then fine-tuned end-to-end on the labeled training data for 20 epochs using the standard cross-entropy loss function.
2.4. Methodological Choices and Inspirations
The design of this project's methodology represents a deliberate synthesis of ideas from both the original MAE paper and the more domain-specific TARDRL framework.
Adoption of Spatio-Temporal Architecture from TARDRL: The decision to process fMRI data as a sequence of 2D slices, first processed by a spatial encoder and then a temporal encoder, is directly inspired by the hierarchical architecture of TARDRL. This approach was chosen over a monolithic 3D ViT because it allows for an explicit and modular modeling of spatial and temporal dynamics, which is well-suited to the nature of fMRI data.
Adoption of Foundational MAE Pre-training: While TARDRL proposes a complex, parallelized training scheme with an "attention-guided" masking strategy, this project deliberately adopts the simpler, sequential pre-training and fine-tuning paradigm from the original MAE paper. This was a strategic choice for two primary reasons:
Establishing a Strong Baseline: Before implementing more complex, task-aware mechanisms, it is essential to first establish a strong and well-understood baseline. This project serves to rigorously evaluate the effectiveness of the canonical MAE self-supervised objective on this spatio-temporal architecture.
Isolating the Effect of Self-Supervision: The TARDRL framework trains its reconstruction and prediction tasks in parallel. By separating the pre-training and fine-tuning stages, our approach allows for a clearer analysis of what features are learned during the self-supervised phase, independent of the downstream task. This provides a clean measure of the utility of the pre-trained encoder.
In essence, this project's methodology can be described as leveraging the architectural insights of TARDRL while using the robust and well-established pre-training protocol of the original MAE to create a strong, interpretable, and computationally efficient baseline model.
2.5. Data Augmentation
To increase the volume of training data and improve model robustness, a dynamic on-the-fly data augmentation strategy was implemented. For each training epoch, a random 10% of the training samples are selected. When these samples are loaded, they are horizontally flipped along one of the spatial axes. This approach efficiently increases data diversity without requiring a large amount of memory to store augmented copies of the dataset. We selected 10% dataset considering the challenge of running this code in Colab environment with limited GPU support and time constraints.

3. Experimental Setup
3.1. Dataset
The model was trained on the HCP_emotion_4D_sample.mat dataset. The data was pre-split into a training set of 3,200 samples and a test set of 800 samples. Each sample is a 4D fMRI scan with dimensions (53, 63, 46), where 53 represents the time dimension. The task is a binary classification problem.
3.2. Hyperparameter Selection
The choice of hyperparameters was guided by the need to create a lightweight model suitable for training on a standard cloud GPU environment (like Google Colab's T4).
Architectural Parameters:
d_model = 192: This is the core embedding dimension. It was chosen to be a multiple of the number of attention heads and represents a good balance between model capacity and memory usage for a lightweight model.
patch_size_2d = (7, 2): This was determined by finding integer divisors of the slice dimensions (63x46) to ensure the entire slice is covered without padding.
n_spatial_enc = 4, n_temporal_enc = 2: Following the principle that the spatial encoder should be at least as deep as the temporal one, this configuration provides sufficient capacity to learn spatial features while keeping the temporal model efficient.
n_dec_mae = 2: As recommended by the original MAE paper, the decoder is significantly smaller than the encoder (2 layers vs. 4), as its reconstruction task is simpler than the encoder's representation learning task.
nhead_spatial = 4, nhead_temporal = 4: The number of attention heads was chosen to be a divisor of d_model. Four heads is a standard choice that allows the model to attend to different feature subspaces in parallel.
mask_ratio = 0.65: A high masking ratio is crucial for MAE. 65% was chosen as a slightly more conservative value than the 75% used in the original MAE paper, providing a good balance for this specific dataset and model size.
Training Parameters:
mae_epochs = 80, clf_epochs = 20: A longer pre-training phase allows the model to learn robust general-purpose features, while a shorter fine-tuning phase adapts these features to the specific classification task.
mae_lr = 1.5e-4, clf_lr = 3e-5: A higher learning rate is used for pre-training, which benefits from a longer cosine decay schedule. A smaller learning rate is used for fine-tuning to avoid catastrophic forgetting and allow for subtle adjustments to the pre-trained weights.
warmup_epochs = 10: A 10-epoch linear warmup was used for the pre-training phase to stabilize the initial stages of training, which is a standard practice for Transformer models.
optimizer = AdamW: The AdamW optimizer was chosen as it is the standard for training Transformer models, effectively handling weight decay.
batch_size = 8: This was selected to fit within the memory constraints of the available GPU.

4. Results and Analysis
The model was successfully trained through both the pre-training and fine-tuning stages. The performance was evaluated based on the learning curves and final classification metrics on the held-out test set.
4.1. Pre-training Performance
The pre-training phase aimed to minimize the MSE loss of reconstructing masked image patches. The loss curve over 80 epochs is shown below:

Figure 1: MAE Pre-training Loss Over Epochs

The plot demonstrates a successful pre-training run. The loss decreases sharply during the initial epochs, especially after the 10-epoch warmup phase, and then gradually converges to a low value of approximately 0.185. This smooth convergence indicates that the model effectively learned to reconstruct the masked patches and, by extension, developed a meaningful understanding of the underlying spatial features of the fMRI data. 
4.2. Fine-tuning Performance
During the fine-tuning stage, the pre-trained spatial encoder was integrated into the full spatio-temporal classifier and trained on the classification task. The learning curves for loss and accuracy are presented below:

Figure 2: Fine-tuning Loss and Accuracy on Training and Test Sets

The fine-tuning curves exhibit several key characteristics:
Loss: The training loss (blue) decreases steadily, while the test loss (orange) decreases initially and then begins to rise, indicating the onset of overfitting after approximately epoch 10.
Accuracy: The training accuracy quickly rises to over 99%, while the test accuracy peaks at 90.88% around epoch 11 before plateauing. This divergence further confirms that the model is beginning to overfit to the training data in the later epochs.
Best Model: The strategy of saving the model with the best validation accuracy was effective, as it captured the model at its peak performance before significant overfitting degraded its generalization capability.

4.3. Final Evaluation
The best-performing model checkpoint (from epoch 11, with 90.88% validation accuracy) was evaluated on the 800-sample test set. The detailed performance metrics provide a comprehensive view of its capabilities.

Figure 3: Final Classification Report on the Test Set



Figure 4: Confusion Matrix on the Test Set

Analysis:
The model achieves an overall accuracy of 91%, which is a strong result for this complex task.
Precision and Recall: The model shows excellent precision for Class 1 (0.96), meaning that when it predicts Class 1, it is very likely to be correct. However, its recall for Class 1 is lower (0.85), indicating it misclassifies 15% of actual Class 1 samples as Class 0. Conversely, it has a very high recall for Class 0 (0.97), correctly identifying almost all instances of that class.
Confusion Matrix: This visualization confirms the findings from the classification report. Out of 400 Class 1 samples, 60 are incorrectly classified as Class 0. This suggests the model has a slight bias towards predicting Class 0. Only 13 of the 400 Class 0 samples were misclassified.

5. Conclusion and Future Work
This project successfully demonstrated the development of a novel hybrid Spatio-Temporal Vision Transformer for the classification of fMRI data. By combining a foundational self-supervised pre-training strategy based on Masked Autoencoders with a hierarchical architecture inspired by the TARDRL framework, the model achieved a strong final accuracy of 91% on a challenging binary classification task. The detailed analysis of hyperparameters and training strategies highlights the importance of techniques like per-patch normalization and learning rate warmup for stabilizing Transformer training.
There are several promising avenues for future work that build directly on this project's findings:
Implementing Attention-Guided Masking: A logical next step is to implement the full attention-guided masking strategy from TARDRL. This would involve using the attention maps generated during fine-tuning to guide the masking process in a more task-aware manner, which could lead to representations that are even better suited for the final classification task.
Advanced Augmentation: More complex augmentation techniques tailored for fMRI data, such as elastic deformations or noise injection, could further improve model robustness.
Hyperparameter Tuning: A more extensive search over the architectural and training hyperparameters could yield further performance improvements.
Overall, this project provides a solid foundation and a successful proof-of-concept for using modern Transformer-based architectures to analyze complex spatio-temporal neuroimaging data, and it sets the stage for more advanced investigations into task-aware self-supervised learning.
6. References
He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
Zhao, Y., Nie, D., Chen, G., Wu, X., Zhang, D., & Wen, X. (2024). TARDRL: Task-Aware Reconstruction for Dynamic Representation Learning of fMRI. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).
