#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>

/*
  TMR + ADS1115 + MPU6050
  ESP32-S3

  SERIAL FORMAT:

  D,time,ch0,ch1,ax,ay,az

  Example:

  D,3.500,-0.053938,0.000000,0.014221,-0.021005,1.018340

  Baud = 921600

  Wiring
  ------------------------------------------------

  ADS1115:
    VDD  -> 3.3V
    GND  -> GND
    SDA  -> GPIO8
    SCL  -> GPIO9
    ADDR -> GND
    A0   -> TMR V-
    A1   -> TMR V+

  MPU6050:
    VCC -> 3.3V
    GND -> GND
    SDA -> GPIO8
    SCL -> GPIO9
    AD0 -> GND

  Addresses:
    ADS1115 = 0x48
    MPU6050 = 0x68
*/


// ==========================================================
// SETTINGS
// ==========================================================

static const int PIN_SDA = 8;
static const int PIN_SCL = 9;
static const int PIN_BTN = 0;

static const uint32_t PERIOD_US = 20000;   // 50 Hz

static const adsGain_t GAIN = GAIN_FOUR;

static const bool AUTO_START = true;


// ==========================================================
// I2C ADDRESSES
// ==========================================================

#define ADS1115_ADDR 0x48
#define MPU6050_ADDR 0x68


// ==========================================================
// ADS1115
// ==========================================================

Adafruit_ADS1115 ads;


// ==========================================================
// RECORDING
// ==========================================================

bool recording = false;

uint32_t t0_us = 0;
uint32_t next_us = 0;

bool btn_prev = true;
uint32_t btn_ms = 0;


// ==========================================================
// MPU WRITE
// ==========================================================

bool mpuWrite(
    uint8_t reg,
    uint8_t value
)
{
    Wire.beginTransmission(
        MPU6050_ADDR
    );

    Wire.write(
        reg
    );

    Wire.write(
        value
    );

    return (
        Wire.endTransmission()
        ==
        0
    );
}


// ==========================================================
// MPU READ X/Y/Z
// ==========================================================

bool readMPU(
    float &ax,
    float &ay,
    float &az
)
{
    Wire.beginTransmission(
        MPU6050_ADDR
    );

    // ACCEL_XOUT_H
    Wire.write(
        0x3B
    );


    if (
        Wire.endTransmission(
            false
        )
        != 0
    )
    {
        return false;
    }


    uint8_t received =
        Wire.requestFrom(
            (uint8_t)MPU6050_ADDR,
            (uint8_t)6,
            (uint8_t)true
        );


    if (
        received != 6
    )
    {
        return false;
    }


    int16_t rawAx =
        ((int16_t)Wire.read() << 8)
        |
        Wire.read();


    int16_t rawAy =
        ((int16_t)Wire.read() << 8)
        |
        Wire.read();


    int16_t rawAz =
        ((int16_t)Wire.read() << 8)
        |
        Wire.read();


    // MPU6050 ±2 g
    // 16384 LSB / g

    ax =
        rawAx / 16384.0f;

    ay =
        rawAy / 16384.0f;

    az =
        rawAz / 16384.0f;


    return true;
}


// ==========================================================
// DEVICE CHECK
// ==========================================================

bool deviceExists(
    uint8_t address
)
{
    Wire.beginTransmission(
        address
    );

    return (
        Wire.endTransmission()
        ==
        0
    );
}


// ==========================================================
// MPU SETUP
// ==========================================================

void setupMPU()
{
    // Wake MPU6050
    mpuWrite(
        0x6B,
        0x00
    );

    delay(
        100
    );


    // Accelerometer ±2 g
    mpuWrite(
        0x1C,
        0x00
    );


    // Digital low-pass filter
    // DLPF_CFG = 3
    mpuWrite(
        0x1A,
        0x03
    );


    // Sample rate:
    // 1000 / (1 + 19)
    // = 50 Hz
    mpuWrite(
        0x19,
        19
    );
}


// ==========================================================
// START RECORDING
// ==========================================================

void startRecording()
{
    recording = true;

    t0_us =
        micros();

    next_us =
        t0_us;
}


// ==========================================================
// SETUP
// ==========================================================

void setup()
{
    // ------------------------------------------------------
    // SERIAL
    // ------------------------------------------------------

    Serial.begin(
        921600
    );

    delay(
        500
    );


    // ------------------------------------------------------
    // I2C
    // ------------------------------------------------------

    Wire.begin(
        PIN_SDA,
        PIN_SCL
    );


    // 100 kHz worked reliably
    Wire.setClock(
        100000
    );


    delay(
        100
    );


    // ------------------------------------------------------
    // BUTTON
    // ------------------------------------------------------

    pinMode(
        PIN_BTN,
        INPUT_PULLUP
    );


    // ------------------------------------------------------
    // ADS1115
    // ------------------------------------------------------

    if (
        !ads.begin(
            ADS1115_ADDR,
            &Wire
        )
    )
    {
        while (
            true
        )
        {
            Serial.println(
                "ADS1115 not found"
            );

            delay(
                1000
            );
        }
    }


    ads.setGain(
        GAIN
    );


    ads.setDataRate(
        RATE_ADS1115_128SPS
    );


    // Continuous differential A0 - A1
    ads.startADCReading(
        ADS1X15_REG_CONFIG_MUX_DIFF_0_1,
        true
    );


    // ------------------------------------------------------
    // MPU6050
    // ------------------------------------------------------

    if (
        deviceExists(
            MPU6050_ADDR
        )
    )
    {
        setupMPU();

        Serial.println(
            "MPU6050 found"
        );
    }
    else
    {
        Serial.println(
            "WARNING: MPU6050 not found"
        );
    }


    Serial.println(
        "ADS1115 + MPU6050 XYZ ready"
    );


    // ------------------------------------------------------
    // AUTO START
    // ------------------------------------------------------

    if (
        AUTO_START
    )
    {
        startRecording();
    }
}


// ==========================================================
// LOOP
// ==========================================================

void loop()
{
    // ------------------------------------------------------
    // BUTTON
    // ------------------------------------------------------

    bool b =
        digitalRead(
            PIN_BTN
        );


    if (
        !b
        &&
        btn_prev
        &&
        millis() - btn_ms > 200
    )
    {
        btn_ms =
            millis();


        if (
            recording
        )
        {
            recording =
                false;

            Serial.println(
                "stopped"
            );
        }
        else
        {
            startRecording();
        }
    }


    btn_prev = b;


    // ------------------------------------------------------
    // NOT RECORDING
    // ------------------------------------------------------

    if (
        !recording
    )
    {
        return;
    }


    // ------------------------------------------------------
    // 50 Hz
    // ------------------------------------------------------

    uint32_t now =
        micros();


    if (
        (int32_t)(
            now - next_us
        )
        <
        0
    )
    {
        return;
    }


    next_us +=
        PERIOD_US;


    // ======================================================
    // TMR
    // ======================================================

    int16_t adcRaw =
        ads.getLastConversionResults();


    float ch0 =
        ads.computeVolts(
            adcRaw
        );


    // Ch1 unused
    float ch1 =
        0.0f;


    // ======================================================
    // MPU X/Y/Z
    // ======================================================

    float ax =
        0.0f;

    float ay =
        0.0f;

    float az =
        1.0f;


    bool mpuOK =
        readMPU(
            ax,
            ay,
            az
        );


    // Do NOT stop serial output if MPU read fails
    if (
        !mpuOK
    )
    {
        ax =
            0.0f;

        ay =
            0.0f;

        az =
            1.0f;
    }


    // ======================================================
    // TIME
    // ======================================================

    float t_s =
        (
            now
            -
            t0_us
        )
        *
        1e-6f;


    // ======================================================
    // SERIAL OUTPUT
    //
    // D,time,ch0,ch1,ax,ay,az
    // ======================================================

    Serial.print(
        "D,"
    );


    Serial.print(
        t_s,
        3
    );


    Serial.print(
        ","
    );


    Serial.print(
        ch0,
        6
    );


    Serial.print(
        ","
    );


    Serial.print(
        ch1,
        6
    );


    Serial.print(
        ","
    );


    Serial.print(
        ax,
        6
    );


    Serial.print(
        ","
    );


    Serial.print(
        ay,
        6
    );


    Serial.print(
        ","
    );


    Serial.println(
        az,
        6
    );
}