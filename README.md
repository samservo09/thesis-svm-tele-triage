# thesis-svm-tele-triage
This repository compiles all the related documents, resources, and source codes to our research.

## Modified Support Vector Machine Algorithm For Text Classification Applied in Psychiatric Tele-Triage
We aim to improve the existing Support Vector Machine Algorithm for text classification by integrating a Large Language Model: MentalBERT. It is pre-trained on health-related data which can improve the accuracy of the current SVM algorithm, and handle the existing problems.

### Problems
1. The Support Vector Machine (SVM) eliminates important features of textual data due to incompleteness of data. <br>
2. The Support Vector Machine (SVM) classifier partially favors the majority class and inaccurately classifies minority class due to imbalance of data. <br>
3. The Support Vector Machine (SVM) frequently predicts data poorly due to High Dimension and Low Sample Size (HDLSS) data.

### Objectives
1. To provide adequate representation of textual data to suffice incompleteness of data.
2. To cope with the imbalance of data that results in favoring a large proportion of data.
3. To avoid data piling due to the results of High-dimension, low-sample data sets.

## Resources
[Dataset Used](https://zenodo.org/records/2667859#.YCwdTR1OlQI) <br>
[MentalBERT](https://arxiv.org/abs/2110.15621)<br>
[SVM Algorithm from scratch - Video](https://www.youtube.com/watch?v=UX0f9BNBcsY)<br>
[SVM Algorithm from scratch - Code](https://github.com/patrickloeber/MLfromscratch/blob/master/mlfromscratch/svm.py)<br>