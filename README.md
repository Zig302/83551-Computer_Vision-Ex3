# Exercise 3 of "Intro To Computer Vision" (83551) course in BIU - Generative Models, Transformers & Foundation Models

This repository contains my implementation of state-of-the-art deep learning architectures for the final assignment of the course. The project explores the shift from standard supervised learning to self-supervised paradigms, attention mechanisms, generative AI, and utilizing massive pre-trained foundation models.

### Features:

* **Image Captioning (Transformers):** Implemented a Transformer-based decoder architecture with Multi-Head Attention to generate natural language descriptions for images, significantly upgrading the vanilla RNN approach from previous assignments.
* **Self-Supervised Learning (SimCLR):** Implemented a contrastive learning framework to learn rich, generalized visual representations from unlabeled data using normalized temperature-scaled cross-entropy loss (NT-Xent).
* **Denoising Diffusion Probabilistic Models (DDPM):** Built a complete Diffusion model from scratch. Implemented a symmetrical U-Net with ResNet blocks and skip connections, the forward noise-addition process, and the reverse denoising process (p_sample) utilizing Classifier-Free Guidance (CFG) for conditional image generation.
* **Foundation Models (CLIP & DINO):** Leveraged OpenAI's CLIP and Meta's DINO (Vision Transformers).
    * *Zero-Shot Classification & Retrieval:* Utilized CLIP's shared text-image latent space and cosine similarity for label-free image classification and text-to-image retrieval.
    * *One-Shot Segmentation:* Built a lightweight multi-layer perceptron (MLP) on top of frozen DINO patch features to perform highly accurate object segmentation from just a single labeled frame.

### Results:

* **Datasets:** CIFAR-10, MS-COCO (Image Captioning), and DAVIS (Video Segmentation).
* **Image Generation:** Successfully trained a DDPM capable of generating cohesive images from pure noise by systematically predicting and removing Gaussian noise over 1,000 timesteps.
* **Segmentation:** Achieved high intersection-over-union (Mean IoU > 0.55) on video segmentation tasks using DINO's semantic patch clustering without requiring massive labeled datasets.
* **Representation Learning:** Demonstrated that self-supervised embeddings (SimCLR) can rival or outperform traditional supervised learning features for downstream tasks.

### Used:

* Python 3
* PyTorch & Torchvision
* NumPy
* Matplotlib & OpenCV (for visualization)
* OpenAI CLIP & HuggingFace (for pre-trained foundation models)
