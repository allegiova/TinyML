# TinyML Gesture Recognition for Arduino Nano 33 BLE
Done by students:
- Alessandro Rossi
- Alessaro Giovanardi
- Norbi Varga
## Overview

This repository contains a **TinyML-based gesture recognition system** designed to run on the **Arduino Nano 33 BLE**.  
The model processes motion data from the onboard **IMU sensors (accelerometer and gyroscope)** and performs gesture classification directly on the microcontroller.

The trained machine learning model is embedded in the firmware as a header file (`model.h`) and executed on-device using TinyML inference.

The system works by:

1. Collecting motion data from the IMU sensors
2. Extracting statistical and spectral features
3. Feeding the features to a trained ML model
4. Running inference directly on the Arduino board

Running inference on-device allows **real-time gesture recognition with very low power consumption**, which is one of the main advantages of TinyML systems.

---

# Repository Structure

```
TinyML/
│
├── gesture_recognition/
│   ├── gesture_recognition.ino   # Main Arduino sketch to run the model
│   ├── model.h                   # Pre-trained TinyML model embedded as C array
│
└── README.md
```

### Main components

#### `gesture_recognition/gesture_recognition.ino`

This is the **main Arduino sketch** used to run the gesture recognition system.

Responsibilities:

- Reads sensor data from the **accelerometer and gyroscope**
- Computes the feature vector
- Loads the trained TinyML model from `model.h`
- Performs inference
- Outputs predictions via **Serial Monitor**

#### `gesture_recognition/model.h`

This file contains the **pre-trained model** converted into a C header format.

The model is already trained and embedded as a byte array so that it can be compiled directly into the Arduino firmware.

No additional training is required to run the system.

---

# Features Extracted

To reduce the dimensionality of the raw IMU signals and make the model suitable for TinyML deployment, several **handcrafted features** are extracted from the sensor data.

Features are computed for both:

- **Accelerometer**
- **Gyroscope**

### 1. RMS (Root Mean Square)

The RMS measures the magnitude of the signal and captures the **overall energy of the motion**.

```
RMS = sqrt((1/N) * Σ(x_i²))
```

This feature is useful to detect **intensity of movements**.

---

### 2. Mean

The mean value captures the **average level of the signal** during the time window.

```
μ = (1/N) * Σ(x_i)
```

It provides information about **signal bias and orientation trends**.

---

### 3. PSD Peaks (Power Spectral Density)

Peaks of the **Power Spectral Density (PSD)** are extracted from the frequency spectrum of the signal.

This allows the model to capture:

- dominant motion frequencies
- periodic patterns in gestures
- dynamic characteristics of movements

Frequency-domain features are particularly useful for distinguishing gestures that have **similar amplitudes but different temporal dynamics**.

---

# Sensors Used

The system uses the **6-axis IMU** integrated in the Arduino Nano 33 BLE.

### Accelerometer

Measures **linear acceleration** along three axes:

```
Ax, Ay, Az
```

### Gyroscope

Measures **angular velocity** along three axes:

```
Gx, Gy, Gz
```

Combining both sensors improves gesture recognition performance since:

- accelerometers capture **translational movement**
- gyroscopes capture **rotational movement**

