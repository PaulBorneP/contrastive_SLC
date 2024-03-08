# Contrastive learning for change detection with SAR data (radar)

Project for the course "Remote sensing data: from sensor to large-scale geospatial data exploitation " :
The objective of this project is to build a representation of SAR data separating the changed and unchanged
areas as presented in [1]. For this purpose, we propose to perform the learning of the neural network in a
contrastive way by using 1) the imaginary and real parts of a SAR image for similar samples [2] and 2)
different locations in the image for the changed samples. The training will be done on multi-temporal stack
of Stripmap TerraSAR-X data.


## Supervision :
Thomas Bultingaire, Christophe Kervazo, Florence Tupin

## References :
1. Y. Chen and L. Bruzzone, "Self-Supervised Change Detection in Multiview Remote Sensing Images," IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–12, 2022
2. E. Dalsasso, L. Denis, F. Tupin, "As if by magic: self-supervised training of deep despeckling networks with MERLIN," IEEE Transactions on Geoscience and Remote Sensing, 2022, 60
