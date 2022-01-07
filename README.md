# CoffeeNet

Source code for the proposed HSV image segmentation algorithm and MobileNetv2 classifier in the paper titled
[Coffee disease detection using a robust HSV color-based segmentation and transfer learning for use on smartphones] (https://onlinelibrary.wiley.com/doi/full/10.1002/int.22747?__cf_chl_jschl_tk__=t305CSI2hIS1FiZd7Fnd5nfWlaFpNAwpemr61yT.Ud4-1641539749-0-gaNycGzNC30)

## Abstract

Ethiopia's coffee export accounts for about 34% of all exports for the budget year 2019/2020. Making it the 10th-largest coffee exporter in the world. Coffee diseases cause around 30% loss in production annually. In this paper, we propose an approach for the detection of four classes of coffee leaf diseases, Rust, Miner, Cercospora, and Phoma by using a fast Hue, Saturation, and Value (HSV) color space segmentation and a MobileNetV2 architecture trained by transfer learning. The proposed HSV color segmentation algorithm constitutes of separating the leaf from the background and separating infected spots on the leaf by automatically finding the best threshold value for the Saturation (S) channel of the HSV color space. The algorithm was compared to the YCgCr and k-means algorithms, in terms of Mean Intersection Over Union and F1-Score. The proposed HSV segm/Users/aic/AIC_Coffee_Disease_DL/README.mdentation algorithm outperformed these methods and achieved an MIoU score of 72.13% and an F1 score of 82.54%. The proposed algorithm also outperforms these methods in terms of execution time, taking on average 0.02 s per image for the segmentation of diseased spots from healthy leaf spots. Our MobileNetV2 classifier achieved a 96% average classification accuracy and 96% average precision. The segmentation accuracy and faster execution make the proposed algorithm suitable for deployment on mobile devices and as such has been successfully implemented on smartphones running the Android operating system.

## Usage

HSV segmentation and pre-processing implementation is contained on

```bash
ImageSegment.py
```

## Cite

Waldamichael, FG, Debelee, TG, Ayano, YM. Coffee disease detection using a robust HSV color-based segmentation and transfer learning for use on smartphones. Int J Intell Syst. 2021; 1- 27. doi:10.1002/int.22747
