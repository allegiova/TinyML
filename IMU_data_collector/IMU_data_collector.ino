#include "Arduino_BMI270_BMM150.h"
#include "arduinoFFT.h"

// Configuration Constants 
const float accelerationThreshold = 1.5; // Trigger threshold in G's
const int numSamples = 128;              // Must be a power of 2 for FFT
const double samplingFrequency = 100.0;  // Approximate sampling rate in Hz

// Accelerometer Buffers
double aRealX[numSamples], aImagX[numSamples];
double aRealY[numSamples], aImagY[numSamples];
double aRealZ[numSamples], aImagZ[numSamples];

// Gyroscope Buffers
double gRealX[numSamples], gImagX[numSamples];
double gRealY[numSamples], gImagY[numSamples];
double gRealZ[numSamples], gImagZ[numSamples];

int samplesRead = 0;                         
bool isCapturing = false; // FLAG: True only when capturing data

ArduinoFFT<double> FFT = ArduinoFFT<double>();

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Print CSV Header (18 features total)
  Serial.println("RMS_aX,Mean_aX,PSDPeak_aX,RMS_aY,Mean_aY,PSDPeak_aY,RMS_aZ,Mean_aZ,PSDPeak_aZ,RMS_gX,Mean_gX,PSDPeak_gX,RMS_gY,Mean_gY,PSDPeak_gY,RMS_gZ,Mean_gZ,PSDPeak_gZ");
}

void loop() {
  float aX, aY, aZ;
  float gX, gY, gZ;

  // 1. STANDBY: Wait for significant motion
  if (!isCapturing) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      
      // Trigger capture if threshold is exceeded
      if ((fabs(aX) + fabs(aY) + fabs(aZ)) >= accelerationThreshold) {
        isCapturing = true; 
        samplesRead = 0;    
      }
    }
  }

  // 2. ACQUISITION: Read both Accel and Gyro data
  if (isCapturing && samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);
      
      // Store Accelerometer data
      aRealX[samplesRead] = (double)aX; aImagX[samplesRead] = 0.0;
      aRealY[samplesRead] = (double)aY; aImagY[samplesRead] = 0.0;
      aRealZ[samplesRead] = (double)aZ; aImagZ[samplesRead] = 0.0;

      // Store Gyroscope data
      gRealX[samplesRead] = (double)gX; gImagX[samplesRead] = 0.0;
      gRealY[samplesRead] = (double)gY; gImagY[samplesRead] = 0.0;
      gRealZ[samplesRead] = (double)gZ; gImagZ[samplesRead] = 0.0;

      samplesRead++;
    }
  }

  // 3. PROCESSING: Calculate and print features
  if (isCapturing && samplesRead == numSamples) {
    
    // Process Accelerometer axes
    processAndPrintAxis(aRealX, aImagX); Serial.print(",");
    processAndPrintAxis(aRealY, aImagY); Serial.print(",");
    processAndPrintAxis(aRealZ, aImagZ); Serial.print(",");

    // Process Gyroscope axes
    processAndPrintAxis(gRealX, gImagX); Serial.print(",");
    processAndPrintAxis(gRealY, gImagY); Serial.print(",");
    processAndPrintAxis(gRealZ, gImagZ); 

    Serial.println(); // End row
    
    // Go back to standby
    isCapturing = false; 
    delay(500); // Prevent double-triggering
  }
}

// Extract and print Mean, RMS, and PSD Peak
void processAndPrintAxis(double vReal[], double vImag[]) {
  // --- FEATURE 1: MEAN ---
  double sum = 0;
  for(int i = 0; i < numSamples; i++) sum += vReal[i];
  double mean = sum / numSamples;

  // --- FEATURE 2: RMS ---
  double sqSum = 0;
  for(int i = 0; i < numSamples; i++) sqSum += (vReal[i] * vReal[i]);
  double rms = sqrt(sqSum / numSamples);

  // --- FEATURE 3: PSD PEAK ---
  for(int i = 0; i < numSamples; i++) vReal[i] -= mean; // Remove DC offset

  FFT.windowing(vReal, numSamples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, numSamples, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, numSamples);
  
  double peakFreq = FFT.majorPeak(vReal, numSamples, samplingFrequency);

  // Print results
  Serial.print(rms, 4);      Serial.print(",");
  Serial.print(mean, 4);     Serial.print(",");
  Serial.print(peakFreq, 2); 
}