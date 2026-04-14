#include "Arduino_BMI270_BMM150.h"
#include "arduinoFFT.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model.h"

// Configuration
const float accelerationThreshold = 2.5; 
const int numSamples = 128;
const double samplingFrequency = 100.0;

// Accelerometer Buffers
double aRealX[numSamples], aImagX[numSamples];
double aRealY[numSamples], aImagY[numSamples];
double aRealZ[numSamples], aImagZ[numSamples];

// Gyroscope Buffers
double gRealX[numSamples], gImagX[numSamples];
double gRealY[numSamples], gImagY[numSamples];
double gRealZ[numSamples], gImagZ[numSamples];

int samplesRead = numSamples;
bool isCapturing = false;

ArduinoFFT<double> FFT = ArduinoFFT<double>();

// TensorFlow Lite globals
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* GESTURES[] = {
   "wave",
   "circle",
   "up_down",
   "punch"
};
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

// Function prototypes
double calculateMean(double vReal[]);
double calculateRMS(double vReal[]);
double calculatePSDPeak(double vReal[], double vImag[], double mean);

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Load TFLite model
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  static tflite::MicroInterpreter static_interpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter = &static_interpreter;

  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ;
  float gX, gY, gZ;

  // 1. Wait for significant motion
  if (!isCapturing) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      if ((fabs(aX) + fabs(aY) + fabs(aZ)) >= accelerationThreshold) {
        isCapturing = true;
        samplesRead = 0;
      }
    }
  }

  // 2. Read 128 samples into accel and gyro buffers
  if (isCapturing && samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // Store Accel
      aRealX[samplesRead] = (double)aX;  aImagX[samplesRead] = 0.0;
      aRealY[samplesRead] = (double)aY;  aImagY[samplesRead] = 0.0;
      aRealZ[samplesRead] = (double)aZ;  aImagZ[samplesRead] = 0.0;

      // Store Gyro
      gRealX[samplesRead] = (double)gX;  gImagX[samplesRead] = 0.0;
      gRealY[samplesRead] = (double)gY;  gImagY[samplesRead] = 0.0;
      gRealZ[samplesRead] = (double)gZ;  gImagZ[samplesRead] = 0.0;

      samplesRead++;
    }
  }

  // 3. Process 18 features and run inference
  if (isCapturing && samplesRead == numSamples) {
    
    // --- ACCEL FEATURES ---
    double aMNx = calculateMean(aRealX); double aRMSx = calculateRMS(aRealX); double aPSDx = calculatePSDPeak(aRealX, aImagX, aMNx);
    double aMNy = calculateMean(aRealY); double aRMSy = calculateRMS(aRealY); double aPSDy = calculatePSDPeak(aRealY, aImagY, aMNy);
    double aMNz = calculateMean(aRealZ); double aRMSz = calculateRMS(aRealZ); double aPSDz = calculatePSDPeak(aRealZ, aImagZ, aMNz);

    // --- GYRO FEATURES ---
    double gMNx = calculateMean(gRealX); double gRMSx = calculateRMS(gRealX); double gPSDx = calculatePSDPeak(gRealX, gImagX, gMNx);
    double gMNy = calculateMean(gRealY); double gRMSy = calculateRMS(gRealY); double gPSDy = calculatePSDPeak(gRealY, gImagY, gMNy);
    double gMNz = calculateMean(gRealZ); double gRMSz = calculateRMS(gRealZ); double gPSDz = calculatePSDPeak(gRealZ, gImagZ, gMNz);

    // --- POPULATE TENSOR (0 to 17) ---
    // Accel X
    tflInputTensor->data.f[0] = aRMSx / 4.0;
    tflInputTensor->data.f[1] = (aMNx + 2.0) / 4.0;
    tflInputTensor->data.f[2] = aPSDx / 15.0;
    // Accel Y
    tflInputTensor->data.f[3] = aRMSy / 4.0;
    tflInputTensor->data.f[4] = (aMNy + 4.0) / 8.0;
    tflInputTensor->data.f[5] = aPSDy / 15.0;
    // Accel Z
    tflInputTensor->data.f[6] = aRMSz / 4.0;
    tflInputTensor->data.f[7] = (aMNz + 2.0) / 4.0;
    tflInputTensor->data.f[8] = aPSDz / 15.0;

    // Gyro X
    tflInputTensor->data.f[9]  = gRMSx / 2000.0;
    tflInputTensor->data.f[10] = (gMNx + 2000.0) / 4000.0;
    tflInputTensor->data.f[11] = gPSDx / 15.0;
    // Gyro Y
    tflInputTensor->data.f[12] = gRMSy / 2000.0;
    tflInputTensor->data.f[13] = (gMNy + 2000.0) / 4000.0;
    tflInputTensor->data.f[14] = gPSDy / 15.0;
    // Gyro Z
    tflInputTensor->data.f[15] = gRMSz / 2000.0;
    tflInputTensor->data.f[16] = (gMNz + 2000.0) / 4000.0;
    tflInputTensor->data.f[17] = gPSDz / 15.0;

    // Run inference
    if (tflInterpreter->Invoke() != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Print classification results
    Serial.println("--- GESTURE DETECTED ---");
    for (int i = 0; i < NUM_GESTURES; i++) {
      Serial.print(GESTURES[i]);
      Serial.print(": ");
      Serial.println(tflOutputTensor->data.f[i], 3);
    }
    Serial.println();

    // Reset capture state
    isCapturing = false; 
    delay(500); 
  }
}

// Calculate Mean
double calculateMean(double vReal[]) {
  double sum = 0;
  for(int i = 0; i < numSamples; i++) {
    sum += vReal[i];
  }
  return sum / numSamples;
}

// Calculate RMS
double calculateRMS(double vReal[]) {
  double sqSum = 0;
  for(int i = 0; i < numSamples; i++) {
    sqSum += (vReal[i] * vReal[i]);
  }
  return sqrt(sqSum / numSamples);
}

// Calculate PSD Peak (Removes DC offset first)
double calculatePSDPeak(double vReal[], double vImag[], double mean) {
  for(int i = 0; i < numSamples; i++) {
    vReal[i] -= mean;
  }

  FFT.windowing(vReal, numSamples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, numSamples, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, numSamples);
  
  return FFT.majorPeak(vReal, numSamples, samplingFrequency);
}