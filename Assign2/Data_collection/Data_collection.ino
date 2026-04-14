/**
 * @file MFCC_Data_Acquisition.ino
 * @brief High-speed MFCC extraction for TinyML Speech Recognition.
 * * This sketch captures audio from a PDM microphone, processes it through 
 * a DSP pipeline (Pre-emphasis, Hamming, FFT, Mel Filterbank, DCT), 
 * and streams 13 MFCC coefficients via Serial for dataset collection.
 */

#include <PDM.h>
#include <math.h>
#include <arm_math.h> // CMSIS-DSP library for optimized math

// ==========================================
// CONFIGURATION & CONSTANTS
// ==========================================
#define MFCC_FRAME_LEN    256      // Audio window size (16ms @ 16kHz)
#define MFCC_HOP_LEN      128      // Step size for 50% overlap (8ms)
#define MFCC_PRE_EMPHASIS 0.97f    // High-pass filter to balance spectrum
#define TWO_PI            6.28318530718f
#define MFCC_NUM_FILTERS  26       // Number of Mel bands
#define MFCC_MIN_FREQ     300      // Low frequency cutoff for speech
#define MFCC_MAX_FREQ     8000     // High frequency cutoff (Nyquist)
#define FRAMES_PER_SECOND 125      // 1000ms / 8ms hop = 125 frames

// ==========================================
// GLOBAL BUFFERS & STATE
// ==========================================
int frameCounter = 0;              // Tracks progress within a 1-second sample
volatile bool isRecording = true;  // Flag to manage PDM interrupt state
short sampleBuffer[MFCC_FRAME_LEN * 2]; // Circular buffer for raw PDM data
volatile int samplesRead = 0;      // Current number of samples ready for processing

// DSP Pipeline intermediate buffers
float _sig[MFCC_FRAME_LEN];        
float _window[MFCC_FRAME_LEN];     
float _windowed[MFCC_FRAME_LEN];   

// FFT & Spectral analysis objects
arm_rfft_fast_instance_f32 fft_instance; 
float _fft_out[MFCC_FRAME_LEN];    
float _power[MFCC_FRAME_LEN / 2];  

// Feature Extraction matrices
float _filterbank[MFCC_NUM_FILTERS][MFCC_FRAME_LEN / 2]; 
float _mel_energy[MFCC_NUM_FILTERS];                     
float _dct_matrix[13][MFCC_NUM_FILTERS]; 
float _mfcc_coeffs[13];                  

// Frequency conversion utilities
float hz2mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float mel2hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

// ==========================================
// SETUP: Hardware & DSP Pre-computations
// ==========================================
void setup() {
  // Use a very high Baud Rate (250000) to prevent Serial output bottlenecking
  Serial.begin(250000); 
  while (!Serial);

  // 1. Initialize the Hamming Window to reduce spectral leakage
  for (int i = 0; i < MFCC_FRAME_LEN; i++) {
    _window[i] = 0.54f - 0.46f * cos(TWO_PI * i / (MFCC_FRAME_LEN - 1));
  }

  // 2. Configure PDM Microphone (Mono, 16kHz)
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    while (1); // Halt if microphone fails
  }

  // 3. Initialize Fast Fourier Transform
  arm_rfft_fast_init_f32(&fft_instance, MFCC_FRAME_LEN);

  // 4. Pre-compute Mel Filterbank Weights (Saves CPU during loop)
  float mel_low = hz2mel(MFCC_MIN_FREQ), mel_high = hz2mel(MFCC_MAX_FREQ);
  int bin_pts[28];
  for (int i = 0; i < 28; i++) {
    bin_pts[i] = floor((mel2hz(mel_low + i * ((mel_high - mel_low) / 27.0f)) / MFCC_MAX_FREQ) * (MFCC_FRAME_LEN / 2));
  }
  for (int m = 0; m < MFCC_NUM_FILTERS; m++) {
    for (int k = 0; k < MFCC_FRAME_LEN / 2; k++) _filterbank[m][k] = 0.0f;
    for (int k = bin_pts[m]; k < bin_pts[m+1]; k++) _filterbank[m][k] = (float)(k - bin_pts[m]) / (bin_pts[m+1] - bin_pts[m]);
    for (int k = bin_pts[m+1]; k <= bin_pts[m+2]; k++) _filterbank[m][k] = (float)(bin_pts[m+2] - k) / (bin_pts[m+2] - bin_pts[m+1]);
  }

  // 5. Pre-compute Discrete Cosine Transform Matrix
  for (int k = 0; k < 13; k++) {
    for (int m = 0; m < MFCC_NUM_FILTERS; m++) {
      _dct_matrix[k][m] = cosf(PI * (k + 1) * (m + 0.5f) / MFCC_NUM_FILTERS);
    }
  }

  Serial.println("System Initialized. Set Serial Monitor to 250000 baud.");
  delay(1000);
}

// ==========================================
// MAIN LOOP: Coordination & Synchronization
// ==========================================
void loop() {
  // Trigger processing only when enough samples are available (Frame Length)
  if (samplesRead >= MFCC_FRAME_LEN) {
    
    processAudioFrame();
    frameCounter++; 

    // --- CRITICAL SECTION: Buffer Management ---
    // Protect variables from interrupt corruption while shifting memory
  int samplesToKeep = samplesRead - MFCC_HOP_LEN;
  for (int i = 0; i < samplesToKeep; i++) {
    sampleBuffer[i] = sampleBuffer[i + MFCC_HOP_LEN];
  }
  samplesRead = samplesToKeep;
  interrupts();
    // -------------------------------------------

    // Handle the end of a 1-second acquisition block
    if (frameCounter >= FRAMES_PER_SECOND) {
      isRecording = false; // Pause microphone to avoid buffer overflow during delay
      Serial.println("=== END OF SAMPLE ===");
      
      Serial.print("Next sample in: ");
      for (int i = 3; i > 0; i--) { 
        Serial.print(i); Serial.print("... "); 
        delay(600); 
      }
      Serial.println("GO!");
      
      noInterrupts();
      samplesRead = 0;
      frameCounter = 0;
      interrupts();
      
      isRecording = true; // Resume acquisition
    }
  }
}

// ==========================================
// INTERRUPT HANDLER: Audio Acquisition
// ==========================================
void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (!isRecording || (samplesRead + (bytesAvailable / 2) > MFCC_FRAME_LEN * 2)) {
    PDM.read(sampleBuffer, bytesAvailable); // Flush data if system is busy
    return;
  }
  int bytesRead = PDM.read(&sampleBuffer[samplesRead], bytesAvailable);
  samplesRead += bytesRead / 2;
}

// ==========================================
// DSP PIPELINE: MFCC Feature Extraction
// ==========================================
void processAudioFrame() {
  // Step 1: Apply Pre-emphasis filter to boost high-frequency components
  _sig[0] = (float)sampleBuffer[0];
  for (int i = 1; i < MFCC_FRAME_LEN; i++) {
    _sig[i] = (float)sampleBuffer[i] - MFCC_PRE_EMPHASIS * (float)sampleBuffer[i-1];
  }

  // Step 2: Apply Hamming Window
  arm_mult_f32(_sig, _window, _windowed, MFCC_FRAME_LEN);

  // Step 3: Compute Real FFT
  arm_rfft_fast_f32(&fft_instance, _windowed, _fft_out, 0);

  // Step 4: Calculate Power Spectrum
  _power[0] = _fft_out[0] * _fft_out[0];   
  _power[127] = _fft_out[1] * _fft_out[1];   
  arm_cmplx_mag_squared_f32(&_fft_out[2], &_power[1], MFCC_FRAME_LEN/2 - 1);

  // Step 5: Mel Filterbank Integration & Log Transform
  for (int m = 0; m < MFCC_NUM_FILTERS; m++) {
    float energy;
    arm_dot_prod_f32(_filterbank[m], _power, MFCC_FRAME_LEN / 2, &energy);
    _mel_energy[m] = logf(energy + 1e-10f); // Add epsilon to prevent log(0)
  }

  // Step 6: Discrete Cosine Transform (DCT) to get 13 MFCC coefficients
  for (int k = 0; k < 13; k++) {
    arm_dot_prod_f32(_dct_matrix[k], _mel_energy, MFCC_NUM_FILTERS, &_mfcc_coeffs[k]);
    Serial.print(_mfcc_coeffs[k]);
    Serial.print(k < 12 ? "\t" : "");
  }
  Serial.println(); // New line for each frame
}