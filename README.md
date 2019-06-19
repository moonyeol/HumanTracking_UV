# Human Tracking Unmanned Vehicle
HTD (Human Tracking Drone) is a parallel project for Hanium exhibition and graduation project for the Department of Electronic Engineering, Kyonggi University.

HTUV(Human Tracking Unmanned Vehicle; 인명추적무인차량)은 경기대학교 전자공학과 졸업작품을 진행하고 있는 프로젝트이다.

## 라이브러리 및 패키지

아래는 본 프로젝트를 진행하기 위해 선정한 라이브러리 및 패키지 버전이며, 일부는 호환성을 고려하여 선택되었다.

| 라이브러리/패키지 | 버전      |
| ----------------- | --------- |
| CUDA              | 10.0      |
| cuDNN             | 7.4.1     |
| OpenCV            | 4.0.1     |
| TensorFlow (GPU)  | 1.13.1    |
| Python            | 3.6.x     |
| Ubuntu            | 18.04 LTS |

Ubuntu 18.04 LTS 및 OpenCV 4.0.1은 CUDA 10.0을 현재 지원하며, CUDA 10.0을 지원하기 위해서는 cuDNN SDK는 7.4.1 이상 그리고 TensorFlow 1.13.0 이상의 버전을 사용해야 한다. 비록 TensorFlow 1.13.1이 파이썬 3.7을 지원하지만 이미 작업한 경험이 있는 파이썬 3.6으로 작업하기로 선정하였다.
