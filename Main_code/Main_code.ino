/*
  TMR heartbeat sensor -> ESP32-S3 (N16R8) + ADS1115
  Streams the SAME serial format as the old STM32 board, so the Python viewer
  works unchanged.  Viewer baud = 921600.

  Wiring:
    ADS1115 VDD -> 3V3     GND -> GND
    ADS1115 SDA -> GPIO8   SCL -> GPIO9
    ADS1115 ADDR -> GND    (I2C address 0x48)
    ADS1115 A0 -> TMR V-   A1 -> TMR V+
  Avoid GPIO26-37 (flash/PSRAM) and GPIO19/20 (USB) on the N16R8.

  BOOT button (GPIO0): stop / start a new recording.
*/
#include <Wire.h>
#include <math.h>
#include <stdio.h>
#include <Adafruit_ADS1X15.h>

// ---- settings -------------------------------------------------------------
static const int       PIN_SDA    = 8;
static const int       PIN_SCL    = 9;
static const int       PIN_BTN    = 0;           // BOOT button
static const uint32_t  PERIOD_US  = 20000;       // 50 Hz output, same as old board
static const adsGain_t GAIN       = GAIN_EIGHT;  // +-0.512 V, 15.6 uV/step
                                                 // GAIN_SIXTEEN = +-0.256 V, 7.8 uV/step
static const bool      AUTO_START = true;        // stream as soon as it boots
// ---------------------------------------------------------------------------

Adafruit_ADS1115 ads;
bool     recording = false;
uint32_t t0_us = 0, next_us = 0;
bool     btn_prev = true;
uint32_t btn_ms = 0;

// Record format: \r\n S d.dddddd S d.dddddd ss.sss \n\r   (28 bytes, no separators)
// S = '0' positive, '1' negative. Seconds wrap at 100.
static void fmtVolts(char *out, float v) {
  long micro = lroundf(fabsf(v) * 1e6f);
  if (micro > 9999999L) micro = 9999999L;
  snprintf(out, 12, "%c%ld.%06ld", v < 0 ? '1' : '0', micro / 1000000L, micro % 1000000L);
}

static int fmtRecord(char *out, float ch0, float ch1, double t_s) {
  char a[12], b[12];
  fmtVolts(a, ch0);
  fmtVolts(b, ch1);
  return snprintf(out, 40, "\r\n%s%s%06.3f\n\r", a, b, fmod(t_s, 100.0));
}

void startRecording() {
  recording = true;
  t0_us = micros();
  next_us = t0_us;
}

void setup() {
  Serial.begin(921600);
  delay(300);
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  pinMode(PIN_BTN, INPUT_PULLUP);

  if (!ads.begin(0x48, &Wire)) {
    while (true) {
      Serial.println("ADS1115 not found - check SDA=GPIO8 SCL=GPIO9 VDD GND ADDR");
      delay(1000);
    }
  }
  ads.setGain(GAIN);
  ads.setDataRate(RATE_ADS1115_128SPS);
  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, true);   // continuous, A0-A1
  Serial.println("Ok, ADS1115 ready");
  if (AUTO_START) startRecording();
}

void loop() {
  bool b = digitalRead(PIN_BTN);
  if (!b && btn_prev && millis() - btn_ms > 200) {
    btn_ms = millis();
    if (recording) { recording = false; Serial.println("stopped"); }
    else           { startRecording(); }
  }
  btn_prev = b;

  if (!recording) return;
  uint32_t now = micros();
  if ((int32_t)(now - next_us) < 0) return;
  next_us += PERIOD_US;

  float ch0 = ads.computeVolts(ads.getLastConversionResults());
  float ch1 = 0.0f;                              // unused, viewer still expects it
  char rec[40];
  int n = fmtRecord(rec, ch0, ch1, (now - t0_us) * 1e-6);
  Serial.write((const uint8_t *)rec, n);
}