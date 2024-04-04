Hello,
Within this repository, my aim is to share an Enhanced Wasserstein GAN employing gradient penalty instead of weight clipping. The rationale behind this choice lies in the fact that weight clipping constrains the critic model's capacity. Thus, by incorporating gradient penalty, we empower the critic model, subsequently enhancing the learning process for our generator through feedback from the critic.
I am currently implementing the Enhanced Wasserstein GAN on various datasets, including MNIST and CelebA (Note: The CelebA dataset is still under preparation).
Lastly, it's worth mentioning that for the MNIST dataset, we have both conditional and unconditional generators available.
