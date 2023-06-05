# Self-supervised End-to-End Landmark Detection and Matching in 3D
Original source code used in the paper *[Automatic landmark correspondence detection in medical images with an application to deformable image registration](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-10/issue-1/014007/Automatic-landmark-correspondence-detection-in-medical-images-with-an-application/10.1117/1.JMI.10.1.014007.full?SSO=1)*. The full text of the paper is available at [https://arxiv.org/abs/2109.02722](https://arxiv.org/abs/2109.02722).

### Usage
Due to restrictions in sharing the medical data used in the paper, the information related to data loading has been omitted. Please insert your custom code for loading the images at the following places:

In `etl3D.py` -> `load_volume` method of `LandmarkDataset` class.
In `train.py`: provide path of the csv file containing data information in `main` method. This path is used by the `LandmarkDataset` class.


If you find the code useful, please cite the following paper:

```
@article{grewal2023automatic,
  title={Automatic landmark correspondence detection in medical images with an application to deformable image registration},
  author={Grewal, Monika and Wiersma, Jan and Westerveld, Henrike and Bosman, Peter AN and Alderliesten, Tanja},
  journal={Journal of Medical Imaging},
  volume={10},
  number={1},
  pages={014007--014007},
  year={2023},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}
```
