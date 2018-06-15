# Landmark-recognition

Did you ever go through your vacation photos and ask yourself: What is the name of this temple I visited in China? Who created this monument I saw in France? Landmark recognition can help! This technology can predict landmark labels directly from image pixels, to help people better understand and organize their photo collections. Today, a great obstacle to landmark recognition research is the lack of large annotated datasets. This motivated us to release Google-Landmarks, the largest worldwide dataset to date, to foster progress in this problem.

The dataset is divided into two sets of images, to evaluate two different computer vision tasks: recognition and retrieval. The data was originally described in [1], and published as part of the Google Landmark Recognition Challenge and Google Landmark Retrieval Challenge. Additionally, to spur research in this field, we have open-sourced Deep Local Features (DELF), an attentive local feature descriptor that we believe is especially suited for this kind of task. DELF's code can be found on github via this link.

If you make use of this dataset in your research, please consider citing:

H. Noh, A. Araujo, J. Sim, T. Weyand, B. Han, "Large-Scale Image Retrieval with Attentive Deep Local Features", Proc. ICCV'17

### Challenges
The two challenges associated to this dataset can be found in the following links:

Google Landmark Recognition Challenge
Google Landmark Retrieval Challenge
CVPR'18 Workshop
The Landmark Recognition Workshop at CVPR 2018 will discuss recent progress on landmark recognition and image retrieval, taking into account the results of the above-mentioned challenges. Top submissions for the challenges will be invited to give talks at the workshop.

### Content
The dataset contains URLs of images which are publicly available online (this Python script may be useful to download the images). Note that no image data is released, only URLs.

The dataset contains test images, training images and index images. The test images are used in both tasks: for the recognition task, a landmark label may be predicted for each test image; for the retrieval task, relevant index images may be retrieved for each test image. The training images are associated to landmark labels, and can be used to train models for the recognition and retrieval challenges (for a visualization of the geographic distribution of training images, see [2]). The index images are used in the retrieval task, composing the set from which images should be retrieved.

Note that the test set for both the recognition and retrieval tasks is the same, to encourage researchers to experiment with both. We also encourage participants to use the training data from the recognition task to train models which could be useful for the retrieval task. Note, however, that there are no landmarks in common between the training/index sets of the two tasks.

The images listed in the dataset are not directly in our control, so their availability may change over time, and the dataset files may be updated to remove URLs which no longer work.

### Dataset construction
The training and index sets were constructed by clustering photos with respect to their geolocation and visual similarity using an algorithm similar to the one described in [3]. Matches between training images were established using local feature matching. Note that there may be multiple clusters per landmark, which typically correspond to different views or different parts of the landmark. To avoid bias, no computer vision algorithms were used for ground truth generation. Instead, we established ground truth correspondences between test images and landmarks using human annotators.

### License
The images listed in this dataset are publicly available on the web, and may have different licenses. Google does not own their copyright.

### References
[1] H. Noh, A. Araujo, J. Sim, T. Weyand, B. Han, "Large-Scale Image Retrieval with Attentive Deep Local Features", Proc. ICCV'17

[2] A. Araujo, T. Weyand, "Google-Landmarks: A New Dataset and Challenge for Landmark Recognition", Google Research blog post, available online here

[3] Y.-T. Zheng, M. Zhao, Y. Song, H. Adam, U. Buddemeier, A. Bissacco, F. Brucher T.-S. Chua, H. Neven, “Tour the World: Building a Web-Scale Landmark Recognition Engine,” Proc. CVPR’09