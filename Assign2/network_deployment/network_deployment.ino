#include <PDM.h>
#include <math.h>
#include <arm_math.h> 

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h" // COMMENTATO PER EVITARE ERRORI DI VERSIONE

#include "kws_model.h"

// ==========================================
// CONFIGURATION & CONSTANTS
// ==========================================
#define MFCC_FRAME_LEN    256      
#define MFCC_HOP_LEN      128      
#define MFCC_PRE_EMPHASIS 0.97f    
#define TWO_PI            6.28318530718f
#define MFCC_NUM_FILTERS  26       
#define MFCC_MIN_FREQ     300      
#define MFCC_MAX_FREQ     8000     
#define FRAMES_PER_SECOND 125      
#define NUM_COEFFS        13

// ==========================================
// GLOBALS: Audio & DSP Buffers
// ==========================================
int frameCounter = 0;              
volatile bool isRecording = true;  

// RIPRISTINATO IL TUO BUFFER SICURO
short sampleBuffer[MFCC_FRAME_LEN + 1024]; 
volatile int samplesRead = 0;   

float _sig[MFCC_FRAME_LEN];
float _window[MFCC_FRAME_LEN];
float _windowed[MFCC_FRAME_LEN];  
arm_rfft_fast_instance_f32 fft_instance; 
float _fft_out[MFCC_FRAME_LEN];
float _power[MFCC_FRAME_LEN / 2];        
float _filterbank[MFCC_NUM_FILTERS][MFCC_FRAME_LEN / 2];
float _mel_energy[MFCC_NUM_FILTERS];                     
float _dct_matrix[NUM_COEFFS][MFCC_NUM_FILTERS];
float _mfcc_coeffs[NUM_COEFFS];                  

float hz2mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
float mel2hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

// ==========================================
// GLOBALS: TensorFlow Lite
// ==========================================
const tflite::Model* tflModel = nullptr;             
tflite::MicroInterpreter* tflInterpreter = nullptr;  
TfLiteTensor* tflInputTensor = nullptr;              
TfLiteTensor* tflOutputTensor = nullptr;             

constexpr int tensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[tensorArenaSize];

const char* CLASSES[] = {"blue", "green", "red", "yellow"};

// ==========================================
// SETUP ROUTINE
// ==========================================
void setup() {
  Serial.begin(115200); 
  while (!Serial);

  for (int i = 0; i < MFCC_FRAME_LEN; i++) {
    _window[i] = 0.54f - 0.46f * cos(TWO_PI * i / (MFCC_FRAME_LEN - 1));
  }
  arm_rfft_fast_init_f32(&fft_instance, MFCC_FRAME_LEN);

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
  for (int k = 0; k < NUM_COEFFS; k++) {
    for (int m = 0; m < MFCC_NUM_FILTERS; m++) {
      _dct_matrix[k][m] = cosf(PI * (k + 1) * (m + 0.5f) / MFCC_NUM_FILTERS);
    }
  }

  tflModel = tflite::GetModel(model);
  // Controllo versione rimosso per compatibilità
  
  static tflite::AllOpsResolver tflOpsResolver;
  static tflite::MicroInterpreter static_interpreter(
    tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter = &static_interpreter;

  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    Serial.println("Error: Failed to start PDM Microphone!");
    while (1); 
  }

  Serial.println("\nSystem Ready! Say a color (Blue, Green, Red, Yellow)...");
  delay(1000); // Piccola pausa prima di iniziare a leggere
}

// ==========================================
// MAIN LOOP
// ==========================================
void loop() {
  // Aspettiamo di avere un frame completo (256 campioni)
  if (samplesRead >= MFCC_FRAME_LEN) {
    
    processAudioFrame();

    for (int i = 0; i < NUM_COEFFS; i++) {
      tflInputTensor->data.f[frameCounter * NUM_COEFFS + i] = _mfcc_coeffs[i];
    }
    
    frameCounter++;

    // Spostamento del buffer sicuro nel loop (come facevi nel Data_collection)
    noInterrupts();
    memmove(sampleBuffer, &sampleBuffer[MFCC_HOP_LEN], (samplesRead - MFCC_HOP_LEN) * sizeof(short));
    samplesRead -= MFCC_HOP_LEN;
    interrupts();

    if (frameCounter >= FRAMES_PER_SECOND) {
      isRecording = false; 

      unsigned long startTime = millis();
      TfLiteStatus invoke_status = tflInterpreter->Invoke();
      unsigned long duration = millis() - startTime;

      if (invoke_status != kTfLiteOk) {
        Serial.println("Error: Model invoke failed!");
        return;
      }

      float max_prob = 0.0;
      int best_class = -1;
      
      Serial.println("\n=== PREDICTION RESULTS ===");
      for (int i = 0; i < 4; i++) {
        float prob = tflOutputTensor->data.f[i]; 
        
        Serial.print(CLASSES[i]);
        Serial.print(": ");
        Serial.print(prob * 100.0f); 
        Serial.println("%");
        
        if (prob > max_prob) {
          max_prob = prob;
          best_class = i;
        }
      }

      Serial.print(">> PREDICTED WORD: ");
      Serial.print(CLASSES[best_class]);
      Serial.print(" (Inference time: ");
      Serial.print(duration);
      Serial.println(" ms) <<\n");
      
      Serial.print("Next sample in: ");
      for (int i = 3; i > 0; i--) { Serial.print(i); Serial.print("... "); delay(600); }
      Serial.println("GO!");
      
      noInterrupts();
      samplesRead = 0;
      frameCounter = 0;
      memset(sampleBuffer, 0, sizeof(sampleBuffer)); 
      interrupts();
      
      isRecording = true; 
    }
  }
}

// ==========================================
// INTERRUPT HANDLER: Microphone Data
// ==========================================
void onPDMdata() {
  int bytesAvailable = PDM.available();

  // Se non stiamo registrando, svuotiamo il buffer PDM a vuoto
  if (!isRecording) {
    PDM.read(sampleBuffer, bytesAvailable); 
    return;
  }

  // Se c'è spazio nel nostro array, copiamo l'audio
  if (samplesRead + (bytesAvailable / 2) < (MFCC_FRAME_LEN + 1024)) {
    PDM.read(&sampleBuffer[samplesRead], bytesAvailable);
    samplesRead += bytesAvailable / 2;
  } else {
    // Se siamo pieni, scarichiamo i dati all'inizio dell'array per non esplodere (evita il crash)
    PDM.read(sampleBuffer, bytesAvailable);
  }
}

// ==========================================
// DSP PIPELINE: MFCC Feature Extraction
// ==========================================
void processAudioFrame() {
  _sig[0] = (float)sampleBuffer[0];
  for (int i = 1; i < MFCC_FRAME_LEN; i++) {
    _sig[i] = (float)sampleBuffer[i] - MFCC_PRE_EMPHASIS * (float)sampleBuffer[i-1];
  }

  arm_mult_f32(_sig, _window, _windowed, MFCC_FRAME_LEN);
  arm_rfft_fast_f32(&fft_instance, _windowed, _fft_out, 0);

  _power[0] = _fft_out[0] * _fft_out[0];   
  _power[127] = _fft_out[1] * _fft_out[1];   
  arm_cmplx_mag_squared_f32(&_fft_out[2], &_power[1], MFCC_FRAME_LEN/2 - 1);

  for (int m = 0; m < MFCC_NUM_FILTERS; m++) {
    float energy;
    arm_dot_prod_f32(_filterbank[m], _power, MFCC_FRAME_LEN / 2, &energy);
    _mel_energy[m] = logf(energy + 1e-10f); 
  }

  for (int k = 0; k < NUM_COEFFS; k++) {
    arm_dot_prod_f32(_dct_matrix[k], _mel_energy, MFCC_NUM_FILTERS, &_mfcc_coeffs[k]);
  }
}