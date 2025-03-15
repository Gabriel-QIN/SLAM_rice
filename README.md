# **SLAM_rice repo**

This `repo` contains codes for SLAM_rice method and web server.

Protein lysine acylations (PLAs) are a subtype
of post-translational modification (PTMs) that occurs on lysine residues. It has
been experimentally demonstrated to regulate many important biological
processes, including gene expression, DNA damage repair, chromatin conformation
dynamics and protein folding. Multiple acylation can be occurred on rice
proteins, such as acetylation, crotonylation, malonylation, succinylation,
2-hydroxyisobutyrylation, and lactylation. PLA modification play a significant
regulatory role in the growth and development of rice, not only regulating
vital metabolic pathways like the tricarboxylic acid cycle, oxidative phosphorylation,
photosynthesis, and starch synthesis but also mediating signal transduction in
response to stress such as disease, salt, cold and heat in rice. Therefore, characterization
of PLA sites is crucial for understanding molecular regulatory mechanisms in
rice.

Over the past few decades,
various experimental methods for accurately identifying modification sites have
emerged, advancing the study of PLAs. Although mass spectrometry can precisely
identify specific lysine acylation sites, this method is costly,
time-consuming, and labor-intensive, making it unsuitable for large-scale
identification. Faced with the rapidly growing protein data, relying solely on
experimental methods can no longer meet the demand for large-scale
identification, thus there is an urgent need to develop computational methods
for rapidly predicting lysine acylation sites. Nevertheless, current
computational methods for lysine acylation in rice are limited by traditional
machine learning algorithms and sequence features, which resulted in poor prediction
accuracy and generalization ability. Recent research has found that protein
structural information is crucial for modification site prediction. Based on
this, this paper proposes a deep learning framework, combining sequence and
structure, and achieves rapid, accurate, and large-scale identification of PLA
sites in rice. The work of this paper mainly includes the following aspects:

1) Collected and organized
   45,533 modification sites from 16,781 rice proteins from multiple
   post-translational modification databases and related literature, constructing
   benchmark datasets for six acylation modifications in rice: acetylation,
   crotonylation, malonylation, succinylation, 2-hydroxyisobutyrylation, and
   lactylation. On this basis, the XGBoost model was further used to test the
   performance of 10 sequence features on different datasets, finding that models
   using BINA and BLOSUM62 features have better predictive performance.
2) To further improve prediction
   performance, we adopted a transfer learning strategy and developed a deep
   learning model (SLAM_rice) that combines sequence and structure encoders for
   accurately and rapidly identifying potential PLA sites in the rice proteome.
   This method not only utilizes BINA and BLOSUM62 sequence features but also
   incorporates structural geometric features including angle, distance, and
   direction information; for different features, we developed sequence encoders and
   structure encoders, with the sequence encoder based on convolutional neural
   networks, recurrent neural networks, and multi-layer perceptrons, and the
   structure encoder based on graph neural networks; finally, the representations
   learned by different encoders are combined, and the modification score is
   output through a decoder, which includes an attention layer and multiple multi-layer
   perceptrons.
3) After 5-fold cross-validation and
   independent testing, SLAM_rice achieved optimal performance on multiple rice
   acylation datasets, with AUROC improving by up to 30% compared to traditional
   machine learning. Extensive experimental results show that, compared with the
   existing computational method iRice-MS, SLAM_rice has superior predictive
   performance.
4) To facilitate academia, the authors
   have made the SLAM_rice source code publicly available at https://github.com/Gabriel-QIN/SLAM_rice
   and developed an online server at https://ai4bio.online/SLAM_rice.
