# End-to-End (E2E) Diffusion
[![GitHub Project](https://img.shields.io/badge/GitHub--blue?style=social&logo=GitHub)](https://github.com/ameya1101/e2e-diffusion)

[![PyPI version](https://img.shields.io/badge/python-3.9-blue)](https://img.shields.io/badge/python-3.9-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code written for my bachelor's thesis project on **[Score-based Generative Models for Detector Reconstruction and Fast Simulations in High-Energy Physics]()**.

---
## **Project Description**

In recent years there has been considerable progress in developing machine learning models suitable for applications in high-energy physics (HEP) for tasks such as event simulation, jet classification, and anomaly detection. In particular, there is a pressing need to develop faster and more accurate techniques for simulating particle physics processes. Currently, such simulations are both time-intensive and require heavy computational resources. Moreover, the High-Luminosity LHC (HL-LHC) upgrades are expected to place the existing computational infrastructure under unprecedented strain due to increased event rates and pileups. Simulations of particle physics events need to be faster without negatively affecting the accuracy and fidelity of the results. Recently, score-based generative models have been shown to produce realistic samples even in large dimensions, surpassing current state-of-the-art models on different benchmarks and categories. To this end, we introduce a score-based generative model in collider physics based on thermodynamic diffusion principles that provides effective reconstruction of LHC events on the level of calorimeter deposits and tracks, which offers the potential for a full detector-level fast simulation of physics events. We work with denoising diffusion probabilistic models (DDPMs) and adapt them to a point-cloud based representation of low-level detector data to faithfully model the distribution of hits in the barrel region of the electromagnetic calorimeter (ECAL) of the Compact Muon Solenoid (CMS) detector array. While this work is limited to the CMS detector suite, the point cloud formulation allows the method to readily be extended to alternative detector geometries. 

---
## **Authors**

* [Ameya Thete](mailto:f20180885@goa.bits-pilani.ac.in)
