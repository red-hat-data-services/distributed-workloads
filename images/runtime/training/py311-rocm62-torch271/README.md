# Training Runtime Container Image

ROCm enabled container image for Training in OpenShift AI.

It includes the following layers:
* UBI 9
* Python 3.11
* ROCm 6.2.4
* PyTorch 2.7.1

Replaces the legacy `py311-rocm62-torch241` and `py311-rocm62-torch251` images
(RHOAIENG-80821 / RHOAIENG-80822).
