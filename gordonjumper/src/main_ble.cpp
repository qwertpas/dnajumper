
// BLE command bridge for Vertiq motor control.
#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <esp_heap_caps.h>
#include <iq_module_communication.hpp>

// ---- Base64 encoding ----
static const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t base64_encode(const uint8_t *input, size_t inputLen, char *output) {
    size_t i = 0, j = 0;
    uint8_t arr3[3], arr4[4];
    size_t outputLen = 0;
    
    while (inputLen--) {
        arr3[i++] = *(input++);
        if (i == 3) {
            arr4[0] = (arr3[0] & 0xfc) >> 2;
            arr4[1] = ((arr3[0] & 0x03) << 4) + ((arr3[1] & 0xf0) >> 4);
            arr4[2] = ((arr3[1] & 0x0f) << 2) + ((arr3[2] & 0xc0) >> 6);
            arr4[3] = arr3[2] & 0x3f;
            for (i = 0; i < 4; i++) {
                output[outputLen++] = base64_chars[arr4[i]];
            }
            i = 0;
        }
    }
    
    if (i) {
        for (j = i; j < 3; j++) arr3[j] = '\0';
        arr4[0] = (arr3[0] & 0xfc) >> 2;
        arr4[1] = ((arr3[0] & 0x03) << 4) + ((arr3[1] & 0xf0) >> 4);
        arr4[2] = ((arr3[1] & 0x0f) << 2) + ((arr3[2] & 0xc0) >> 6);
        for (j = 0; j < i + 1; j++) {
            output[outputLen++] = base64_chars[arr4[j]];
        }
        while (i++ < 3) {
            output[outputLen++] = '=';
        }
    }
    
    output[outputLen] = '\0';
    return outputLen;
}

size_t base64_encoded_length(size_t inputLen) {
    return 4 * ((inputLen + 2) / 3);
}

// ---- PRBS RNG ----
static uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// ---- PRBS state ----
bool prbs_running = false;
uint32_t prbs_rng = 0x12345678u;
uint32_t prbs_start_time = 0;
uint32_t prbs_next_bit_time = 0;
float prbs_amplitude = 5.0f;
uint32_t prbs_bit_us = 20000;
uint32_t prbs_duration_us = 10000000;
float prbs_v_cmd = 0.0f;

// ---- Loop timing ----
const uint32_t LOOP_FREQ_HZ = 2000;
const uint32_t LOOP_PERIOD_US = 1000000 / LOOP_FREQ_HZ;  // 500µs

// ---- Logging buffer ----
const uint32_t LOG_DURATION_S = 5;
// const uint32_t LOG_BUFFER_SIZE = LOOP_FREQ_HZ * LOG_DURATION_S;  // 10000 samples
const uint32_t LOG_BUFFER_SIZE = 12000;  // a bit more

struct LogSample {
    float angle;
    float vel;
    float vbat;
    float set_volts;
    uint32_t time_us;
};

LogSample *logBuffer = nullptr;
uint32_t logHead = 0;      // Next write position (circular)
uint32_t logCount = 0;     // Total samples written (for knowing if wrapped)
uint32_t logStartTime = 0;
uint32_t logStopTime = 0;  // When to actually stop logging (for post-target delay)
bool logging = false;
bool logReady = false;  // true when a complete log is ready to download

void setupLogBuffer() {
    size_t bytes = LOG_BUFFER_SIZE * sizeof(LogSample);
    logBuffer = (LogSample *)heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!logBuffer) {
        Serial.printf("Log buffer allocation failed (%u bytes)\n", (unsigned)bytes);
        while (true) {
            delay(1000);
        }
    }
    memset(logBuffer, 0, bytes);
    Serial.printf("Log buffer: %lu samples (%u bytes PSRAM)\n", LOG_BUFFER_SIZE, (unsigned)bytes);
}

// ---- Motor setup ----
const int TX_PIN = 13; // ESP32 TX → Vertiq RX
const int RX_PIN = 12; // ESP32 RX → Vertiq TX
const int MTR_GND = 11;
const int LED_PIN = 48;
const int MOTOR_ID_MIN = 0;
const int MOTOR_ID_MAX = 16;
const uint32_t MOTOR_STARTUP_DELAY_MS = 1000;
const uint32_t MOTOR_RESCAN_MS = 1000;

GenericInterface com; // Use GenericInterface directly for batching
PowerMonitorClient *power = nullptr;
BrushlessDriveClient *mot = nullptr;
MultiTurnAngleControlClient *multi = nullptr;
int motor_id = -1;
bool motor_found = false;
uint32_t last_motor_scan_ms = 0;

// ---- BLE UART service ----
const char *BLE_NAME = "dnajumper";
static BLEUUID BLE_SERVICE_UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
static BLEUUID BLE_RX_UUID("6E400002-B5A3-F393-E0A9-E50E24DCCA9E");  // central writes commands
static BLEUUID BLE_TX_UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E");  // firmware notifies responses

BLEServer *bleServer = nullptr;
BLECharacteristic *bleTx = nullptr;
bool bleClientConnected = false;
uint16_t bleMtu = 23;

portMUX_TYPE bleCommandMux = portMUX_INITIALIZER_UNLOCKED;
char bleCommand[128];
volatile bool bleCommandReady = false;

size_t blePayloadSize() {
    uint16_t payload = bleMtu > 23 ? bleMtu - 3 : 20;
    return constrain(payload, (uint16_t)20, (uint16_t)180);
}

void commandWrite(const uint8_t *data, size_t len) {
    if (!bleClientConnected || !bleTx || len == 0) {
        return;
    }

    size_t maxChunk = blePayloadSize();
    for (size_t offset = 0; offset < len; offset += maxChunk) {
        size_t chunkLen = min(maxChunk, len - offset);
        bleTx->setValue((uint8_t *)(data + offset), chunkLen);
        bleTx->notify();
        delay(2);
    }
}

void commandReply(const String &message) {
    commandWrite((const uint8_t *)message.c_str(), message.length());
}

bool readBleCommand(String &cmd) {
    char local[sizeof(bleCommand)];
    bool ready = false;

    portENTER_CRITICAL(&bleCommandMux);
    if (bleCommandReady) {
        strncpy(local, bleCommand, sizeof(local));
        local[sizeof(local) - 1] = '\0';
        bleCommandReady = false;
        ready = true;
    }
    portEXIT_CRITICAL(&bleCommandMux);

    if (!ready) {
        return false;
    }

    cmd = local;
    cmd.trim();
    cmd.toUpperCase();
    return cmd.length() > 0;
}

class MotorBleServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
        bleClientConnected = true;
        bleMtu = server->getPeerMTU(server->getConnId());
        Serial.println("BLE client connected");
    }

    void onDisconnect(BLEServer *server) override {
        bleClientConnected = false;
        bleMtu = 23;
        Serial.println("BLE client disconnected");
        server->startAdvertising();
    }

    void onMtuChanged(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
        bleMtu = param->mtu.mtu;
        Serial.printf("BLE MTU %u\n", bleMtu);
    }
};

class MotorBleCommandCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *characteristic) override {
        size_t len = min(characteristic->getLength(), sizeof(bleCommand) - 1);
        if (len == 0) {
            return;
        }

        portENTER_CRITICAL(&bleCommandMux);
        memcpy(bleCommand, characteristic->getData(), len);
        bleCommand[len] = '\0';
        bleCommandReady = true;
        portEXIT_CRITICAL(&bleCommandMux);
    }
};

void setupBle() {
    BLEDevice::init(BLE_NAME);
    BLEDevice::setMTU(185);

    bleServer = BLEDevice::createServer();
    bleServer->setCallbacks(new MotorBleServerCallbacks());

    BLEService *service = bleServer->createService(BLE_SERVICE_UUID);
    bleTx = service->createCharacteristic(
        BLE_TX_UUID,
        BLECharacteristic::PROPERTY_NOTIFY | BLECharacteristic::PROPERTY_READ
    );
    bleTx->addDescriptor(new BLE2902());

    BLECharacteristic *bleRx = service->createCharacteristic(
        BLE_RX_UUID,
        BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
    );
    bleRx->setCallbacks(new MotorBleCommandCallbacks());

    service->start();
    BLEAdvertising *advertising = BLEDevice::getAdvertising();
    advertising->addServiceUUID(BLE_SERVICE_UUID);
    advertising->setScanResponse(true);
    advertising->setMinPreferred(0x06);
    advertising->setMinPreferred(0x12);
    BLEDevice::startAdvertising();
}

// Helper to send all bytes in the com TX queue
void comSend(GenericInterface &iface) {
    uint8_t buf[128];
    uint8_t len;
    if (iface.GetTxBytes(buf, len)) {
        Serial1.write(buf, len);
    }
}

void comSend() {
    comSend(com);
}

void comReadClients(GenericInterface &iface, PowerMonitorClient &readPower, MultiTurnAngleControlClient &readMulti) {
    uint8_t buf[128];
    uint8_t len;
    while (Serial1.available()) {
        len = Serial1.readBytes(buf, min((int)sizeof(buf), Serial1.available()));
        iface.SetRxBytes(buf, len);
        
        uint8_t *packet;
        uint8_t pLen;
        while (iface.PeekPacket(&packet, &pLen)) {
            readMulti.ReadMsg(packet, pLen);
            readPower.ReadMsg(packet, pLen);
            iface.DropPacket();
        }
    }
}

// Helper to read and parse packets
void comRead() {
    if (power && multi) {
        comReadClients(com, *power, *multi);
    }
}

bool probeMotorAddress(int id) {
    GenericInterface probeCom;
    PowerMonitorClient probePower(id);
    MultiTurnAngleControlClient probeMulti(id);

    while (Serial1.available()) {
        Serial1.read();
    }

    for (int attempt = 0; attempt < 2; attempt++) {
        probePower.volts_.get(probeCom);
        probeMulti.obs_angular_displacement_.get(probeCom);
        comSend(probeCom);

        uint32_t start = micros();
        while (micros() - start < 3000) {
            comReadClients(probeCom, probePower, probeMulti);
            if (probePower.volts_.IsFresh() || probeMulti.obs_angular_displacement_.IsFresh()) {
                return true;
            }
        }
        delay(2);
    }

    return false;
}

void useMotorAddress(int id, bool found) {
    delete power;
    delete mot;
    delete multi;

    motor_id = found ? id : -1;
    motor_found = found;
    power = new PowerMonitorClient(id);
    mot = new BrushlessDriveClient(id);
    multi = new MultiTurnAngleControlClient(id);
}

void scanMotorAddress() {
    last_motor_scan_ms = millis();
    Serial.printf("Scanning Vertiq addresses %d..%d\n", MOTOR_ID_MIN, MOTOR_ID_MAX);
    for (int id = MOTOR_ID_MIN; id <= MOTOR_ID_MAX; id++) {
        Serial.printf("  probe %d\n", id);
        if (probeMotorAddress(id)) {
            useMotorAddress(id, true);
            Serial.printf("Vertiq found at address %d\n", id);
            return;
        }
    }

    useMotorAddress(MOTOR_ID_MIN, false);
    Serial.println("No Vertiq found");
}

void stopMotor() {
    if (!motor_found) {
        return;
    }
    multi->ctrl_coast_.set(com);
    comSend();
}

// State for LOG command flow
bool awaitingLogConfirm = false;

void setup() {
    // setCpuFrequencyMhz(80);
    setCpuFrequencyMhz(240);

    Serial.begin(115200);
    setupLogBuffer();
    pinMode(MTR_GND, OUTPUT);
    digitalWrite(MTR_GND, LOW);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    delay(1000);
    Serial.println("\nStarting Vertiq BLE Control");

    // ---- BLE listener ----
    setupBle();
    Serial.printf("Advertising BLE device '%s'\n", BLE_NAME);

    // ---- Motor UART ----
    Serial1.begin(921600, SERIAL_8N1, RX_PIN, TX_PIN);
    delay(MOTOR_STARTUP_DELAY_MS);
    scanMotorAddress();
    stopMotor();
    Serial.println("Motor interface initialized at 921600 baud");
}

float raw_angle = 0;
float zero_angle = 0;
float angle = 0;
float target = 0;
bool spinning = false;
float start_err = 1;
float set_voltage = 0.2f;

// ---- Rebound action ----
float reboundDropRad = 2.0f;
const uint32_t REBOUND_DELAY_US = 500000;
bool reboundLoaded = false;
bool reboundWaiting = false;
bool haveReachedTarget = false;
float reboundTarget = 0;
float reachedTarget = 0;
int reachedDirection = 0;
uint32_t reachedTime = 0;

// ---- Chained command queue ----
enum ChainCmdType { CHAIN_VOLTAGE, CHAIN_TARGET, CHAIN_REBOUND };
struct ChainCmd { ChainCmdType type; float value; };
const int MAX_CHAIN_LEN = 10;
ChainCmd chainQueue[MAX_CHAIN_LEN];
int chainLen = 0;   // Total commands in queue
int chainIdx = 0;   // Next command to execute

// Parse chained command string like "v4t5v6t15" into chainQueue
// Returns true if valid chain with at least one command parsed
bool parseChain(const String &cmd) {
    chainLen = 0;
    chainIdx = 0;
    int pos = 0;
    int len = cmd.length();
    
    while (pos < len && chainLen < MAX_CHAIN_LEN) {
        char c = cmd.charAt(pos);
        if (c == 'V' || c == 'v') {
            // Find end of number
            int numStart = pos + 1;
            int numEnd = numStart;
            while (numEnd < len) {
                char nc = cmd.charAt(numEnd);
                if ((nc >= '0' && nc <= '9') || nc == '.' || nc == '-') {
                    numEnd++;
                } else {
                    break;
                }
            }
            if (numEnd > numStart) {
                chainQueue[chainLen].type = CHAIN_VOLTAGE;
                chainQueue[chainLen].value = cmd.substring(numStart, numEnd).toFloat();
                chainLen++;
            }
            pos = numEnd;
        } else if (c == 'T' || c == 't') {
            int numStart = pos + 1;
            int numEnd = numStart;
            while (numEnd < len) {
                char nc = cmd.charAt(numEnd);
                if ((nc >= '0' && nc <= '9') || nc == '.' || nc == '-') {
                    numEnd++;
                } else {
                    break;
                }
            }
            if (numEnd > numStart) {
                chainQueue[chainLen].type = CHAIN_TARGET;
                chainQueue[chainLen].value = cmd.substring(numStart, numEnd).toFloat();
                chainLen++;
            }
            pos = numEnd;
        } else if (c == 'R' || c == 'r') {
            chainQueue[chainLen].type = CHAIN_REBOUND;
            chainQueue[chainLen].value = 0;
            chainLen++;
            pos++;
        } else {
            pos++;  // Skip unknown characters
        }
    }
    return chainLen > 0;
}

// ---- Control mode ----
enum ControlMode { MODE_VOLTAGE, MODE_VELOCITY };
ControlMode control_mode = MODE_VOLTAGE;
float set_velocity = 20.0f;  // rad/s for velocity mode

void clearRebound() {
    reboundLoaded = false;
    reboundWaiting = false;
}

void loadRebound(uint32_t now) {
    reboundTarget = spinning ? target : reachedTarget;
    reboundLoaded = true;
    reboundWaiting = !spinning;
    if (!spinning && !haveReachedTarget) {
        reachedTarget = target;
        reachedDirection = (start_err >= 0) ? 1 : -1;
        reachedTime = now;
        haveReachedTarget = true;
    }
}

void startRebound(float currentAngle) {
    target = constrain(-reboundTarget, -188, 188);
    start_err = target - currentAngle;
    spinning = true;
    control_mode = MODE_VOLTAGE;
    chainLen = 0;
    chainIdx = 0;
    clearRebound();
    logStopTime = 0;
    logging = true;
    logReady = false;
    Serial.printf("REBOUND: T%.1f (err %.1f)\n", target, start_err);
}

// ---- Homing state ----
bool homing = false;
ControlMode pre_home_mode = MODE_VOLTAGE;
float pre_home_velocity = 20.0f;

float vbat = 0;
float vel = 0;

uint32_t lastLoopTime = 0;
uint32_t lastUartTimeUs = 0;

void sendLogData() {
    // Each LogSample is 20 bytes. 
    // We will send in chunks of 30 samples = 600 bytes raw.
    // 600 bytes raw becomes 800 bytes in Base64, then BLE notification chunking splits it.
    const uint32_t SAMPLES_PER_CHUNK = 30;
    char base64Chunk[801]; // 4 * (600/3) + 1 for null
    LogSample tempChunk[SAMPLES_PER_CHUNK];
    
    uint32_t totalSamples = min(logCount, LOG_BUFFER_SIZE);
    // Start index: if wrapped, start from logHead (oldest), else start from 0
    uint32_t startIdx = (logCount >= LOG_BUFFER_SIZE) ? logHead : 0;
    
    uint32_t samplesSent = 0;
    int chunkNum = 0;
    
    while (samplesSent < totalSamples) {
        uint32_t samplesToProcess = min(SAMPLES_PER_CHUNK, totalSamples - samplesSent);
        
        // Copy samples to temp buffer in order (handling circular wrap)
        for (uint32_t i = 0; i < samplesToProcess; i++) {
            uint32_t idx = (startIdx + samplesSent + i) % LOG_BUFFER_SIZE;
            tempChunk[i] = logBuffer[idx];
        }
        
        size_t bytesToProcess = samplesToProcess * sizeof(LogSample);
        base64_encode((uint8_t *)tempChunk, bytesToProcess, base64Chunk);
        
        commandWrite((uint8_t *)base64Chunk, strlen(base64Chunk));
        
        samplesSent += samplesToProcess;
        chunkNum++;
        delay(2);
    }
    
    // Send end marker
    delay(20);
    commandReply("END\n");
    
    Serial.printf("Sent %lu samples in %d chunks\n", samplesSent, chunkNum);
}

long loopcount = 0;
void loop() {
    // ---- Precise loop timing ----
    uint32_t now = micros();
    if (now - lastLoopTime < LOOP_PERIOD_US) {
        return;  // Not time yet
    }
    lastLoopTime = now;
    loopcount++;

    if (!motor_found && millis() - last_motor_scan_ms >= MOTOR_RESCAN_MS) {
        scanMotorAddress();
        stopMotor();
    }
    
    if (motor_found) {
    // ---- Batch read sensors and set control ----
    power->volts_.get(com);
    multi->obs_angular_velocity_.get(com);
    multi->obs_angular_displacement_.get(com);

    if (reboundLoaded && reboundWaiting && !spinning && haveReachedTarget &&
        (int32_t)(now - (reachedTime + REBOUND_DELAY_US)) >= 0) {
        bool reversedAfterPositiveMove = reachedDirection > 0 && angle <= reachedTarget - reboundDropRad;
        bool reversedAfterNegativeMove = reachedDirection < 0 && angle >= reachedTarget + reboundDropRad;
        if (reversedAfterPositiveMove || reversedAfterNegativeMove) {
            startRebound(angle);
        }
    }
    
    float error = target - angle;
    float current_set_volts = 0;

    // ---- PRBS mode ----
    if (prbs_running) {
        uint32_t now_prbs = micros();
        // Check duration
        if ((now_prbs - prbs_start_time) >= prbs_duration_us) {
            prbs_running = false;
            prbs_v_cmd = 0.0f;
            multi->ctrl_coast_.set(com);
            logging = false;
            logReady = true;
            digitalWrite(LED_PIN, LOW);
        } else {
            // Update bit
            if ((int32_t)(now_prbs - prbs_next_bit_time) >= 0) {
                prbs_next_bit_time += prbs_bit_us;
                uint32_t r = xorshift32(prbs_rng);
                prbs_v_cmd = (r & 1u) ? prbs_amplitude : -prbs_amplitude;
            }
            current_set_volts = prbs_v_cmd;
            multi->ctrl_volts_.set(com, current_set_volts);
            digitalWrite(LED_PIN, HIGH);
        }
    }
    // ---- Normal spinning mode ----
    else if(spinning && start_err*error>0) {
        if (control_mode == MODE_VOLTAGE) {
            current_set_volts = (error > 0) ? set_voltage : -set_voltage;
            multi->ctrl_volts_.set(com, current_set_volts);
        } else {
            // Velocity control mode
            float vel_cmd = (error > 0) ? set_velocity : -set_velocity;
            multi->ctrl_velocity_.set(com, vel_cmd);
        }
        digitalWrite(LED_PIN, HIGH);
    } else {
        // Target reached (or not spinning) - check for chained commands
        bool chainContinued = false;
        if (spinning && chainIdx < chainLen) {
            // Execute pending chain commands
            while (chainIdx < chainLen) {
                ChainCmd &cmd = chainQueue[chainIdx];
                chainIdx++;
                if (cmd.type == CHAIN_VOLTAGE) {
                    set_voltage = constrain(cmd.value, 0, vbat);
                    Serial.printf("CHAIN: V%.2f\n", set_voltage);
                } else if (cmd.type == CHAIN_TARGET) {
                    target = constrain(cmd.value, -188, 188);
                    start_err = target - angle;
                    haveReachedTarget = false;
                    Serial.printf("CHAIN: T%.1f (err %.1f)\n", target, start_err);
                    chainContinued = true;
                    break;  // Continue spinning to new target
                } else if (cmd.type == CHAIN_REBOUND) {
                    loadRebound(now);
                    Serial.printf("CHAIN: R T%.1f\n", reboundTarget);
                }
            }
        }
        
        if (!chainContinued) {
            if (spinning) {
                reachedTarget = target;
                reachedDirection = (start_err >= 0) ? 1 : -1;
                reachedTime = now;
                haveReachedTarget = true;
                if (reboundLoaded) {
                    reboundWaiting = true;
                }
            }

            // Handle homing completion and always coast at rest.
            if (homing) {
                control_mode = pre_home_mode;
                set_velocity = pre_home_velocity;
                homing = false;
                Serial.println("HOME COMPLETE");
            }
            multi->ctrl_coast_.set(com);

            // When movement ends, schedule logging to stop after 0.1s
            if (spinning && logging && logStopTime == 0 && !reboundLoaded) {
                logStopTime = now + 100000;  // 0.1 seconds after target reached
            }
            spinning = false;
            chainLen = 0;  // Clear chain
            chainIdx = 0;
            digitalWrite(LED_PIN, LOW);
        }
    }
    
    // Stop logging after post-target delay
    if (logStopTime != 0 && (int32_t)(now - logStopTime) >= 0) {
        logging = false;
        logReady = true;
        logStopTime = 0;
        Serial.printf("Log stopped: %lu samples\n", min(logCount, LOG_BUFFER_SIZE));
    }
    
    // Send all requests in one burst
    uint32_t uartStart = micros();
    comSend();
    
    // Wait for replies (with 1.5ms timeout)
    while (micros() - uartStart < 1500) {
        comRead();
        if (power->volts_.IsFresh() && 
            multi->obs_angular_velocity_.IsFresh() && 
            multi->obs_angular_displacement_.IsFresh()) {
            break;
        }
    }
    lastUartTimeUs = micros() - uartStart;
    
    // Get values from entries
    if (power->volts_.IsFresh()) vbat = power->volts_.get_reply();
    if (multi->obs_angular_velocity_.IsFresh()) vel = multi->obs_angular_velocity_.get_reply();
    if (multi->obs_angular_displacement_.IsFresh()) {
        raw_angle = multi->obs_angular_displacement_.get_reply();
        angle = raw_angle - zero_angle;
    }

    // ---- Logging (circular buffer) ----
    if (logging) {
        logBuffer[logHead].angle = angle;
        logBuffer[logHead].vel = vel;
        logBuffer[logHead].vbat = vbat;
        logBuffer[logHead].set_volts = current_set_volts;
        logBuffer[logHead].time_us = now - logStartTime;
        logHead = (logHead + 1) % LOG_BUFFER_SIZE;
        logCount++;
    }
    }

    // ---- Handle BLE commands ----
    if (loopcount % 10 == 0) {
        String cmd;
        if (readBleCommand(cmd)) {

            // Handle Y/N/C confirmation for log download
            if (awaitingLogConfirm) {
                if (cmd == "Y" || cmd == "YES" || cmd.startsWith("Y ")) {
                    awaitingLogConfirm = false;
                    commandReply("SENDING...\n");
                    delay(50);
                    sendLogData();
                    // Clear log after successful download
                    logHead = 0;
                    logCount = 0;
                    logStartTime = 0;
                } else if (cmd == "C" || cmd == "CLEAR") {
                    awaitingLogConfirm = false;
                    logHead = 0;
                    logCount = 0;
                    logStartTime = 0;
                    commandReply("LOG CLEARED\n");
                } else {
                    awaitingLogConfirm = false;
                    commandReply("CANCELLED\n");
                }
            } else if (!motor_found && !(cmd.startsWith("STATUS") || cmd == "S")) {
                commandReply("NO MOTOR\n");
            }
            // ---- Chained command detection (e.g., V4T5V6T15, T29R) ----
            else if ((cmd.startsWith("V") || cmd.startsWith("T")) && 
                     ((cmd.indexOf('V') >= 0 && cmd.indexOf('T') >= 0) || cmd.indexOf('R') >= 0) &&
                     !cmd.startsWith("VOLT") && !cmd.startsWith("VEL")) {
                // Parse as chained command
                if (parseChain(cmd)) {
                    clearRebound();
                    haveReachedTarget = false;
                    // Start logging
                    if (logStartTime == 0) {
                        logStartTime = micros();
                    }
                    logging = true;
                    logReady = false;
                    logStopTime = 0;
                    
                    // Execute commands until we hit a target
                    String response = "CHAIN[" + String(chainLen) + "]: ";
                    while (chainIdx < chainLen) {
                        ChainCmd &c = chainQueue[chainIdx];
                        chainIdx++;
                        if (c.type == CHAIN_VOLTAGE) {
                            set_voltage = constrain(c.value, 0, vbat);
                            response += "V" + String(c.value, 1) + " ";
                        } else if (c.type == CHAIN_TARGET) {
                            target = constrain(c.value, -188, 188);
                            start_err = target - angle;
                            spinning = true;
                            haveReachedTarget = false;
                            response += "T" + String(c.value, 1) + " ";
                            break;  // Start spinning, remaining commands execute on target reach
                        } else if (c.type == CHAIN_REBOUND) {
                            loadRebound(now);
                            response += "R ";
                        }
                    }
                    response += "(logging)\n";
                    Serial.print(response);
                    commandReply(response);
                } else {
                    commandReply("INVALID CHAIN\n");
                }

            } else if (cmd.startsWith("T")) {
                float val = cmd.substring(1).toFloat();
                target = constrain(val, -188, 188); //30 rotations
                start_err = target - angle;
                spinning = true;
                chainLen = 0;  // Clear any pending chain
                chainIdx = 0;
                clearRebound();
                haveReachedTarget = false;
                
                // Start logging (only set start time if buffer was cleared)
                if (logStartTime == 0) {
                    logStartTime = micros();
                }
                logging = true;
                logReady = false;
                logStopTime = 0;  // Cancel any pending log stop
                
                Serial.printf("TARGET SET %.1f, ERR %.1f (logging)\n", target, start_err);
                commandReply("TARGET SET " + String(target, 1) + ", ERR " + String(start_err, 1) + " (logging)\n");

            } else if (cmd.startsWith("RTH")) {
                float val = cmd.substring(3).toFloat();
                reboundDropRad = constrain(val, 0.1f, 20.0f);
                Serial.printf("REBOUND THRESHOLD %.2f rad\n", reboundDropRad);
                commandReply("REBOUND THRESHOLD " + String(reboundDropRad, 2) + " rad\n");

            } else if (cmd == "R" || cmd == "REBOUND") {
                if (spinning || haveReachedTarget) {
                    loadRebound(now);
                    String state = reboundWaiting ? "WAITING" : "LOADED";
                    Serial.printf("REBOUND %s T%.1f\n", state.c_str(), reboundTarget);
                    commandReply("REBOUND " + state + " T" + String(reboundTarget, 1) + "\n");
                } else {
                    commandReply("NO TARGET FOR REBOUND\n");
                }

            } else if(cmd.startsWith("VOLT")) {
                float val = cmd.substring(4).toFloat();
                set_voltage = constrain(val, 0, vbat);
                control_mode = MODE_VOLTAGE;
                Serial.printf("MODE: VOLTAGE %.2fV\n", set_voltage);
                commandReply("MODE: VOLTAGE " + String(set_voltage, 2) + "V\n");

            } else if(cmd.startsWith("VEL")) {
                float val = cmd.substring(3).toFloat();
                set_velocity = constrain(val, 0.1f, 100.0f);
                control_mode = MODE_VELOCITY;
                Serial.printf("MODE: VELOCITY %.1f rad/s\n", set_velocity);
                commandReply("MODE: VELOCITY " + String(set_velocity, 1) + " rad/s\n");

            } else if(cmd.startsWith("V")) {
                float val = cmd.substring(1).toFloat();
                set_voltage = constrain(val, 0, vbat);
                chainLen = 0;  // Clear any pending chain
                chainIdx = 0;
                Serial.printf("VOLTAGE SET %.2f / %.2f\n", set_voltage, vbat);
                commandReply("VOLTAGE SET " + String(set_voltage, 2) + " / " + String(vbat, 2) + "\n");

            } else if(cmd.startsWith("STOP") || cmd.startsWith("OFF")){
                spinning = false;
                prbs_running = false;
                prbs_v_cmd = 0.0f;
                logging = false;
                homing = false;
                chainLen = 0;  // Clear chain
                chainIdx = 0;
                clearRebound();
                multi->ctrl_coast_.set(com);
                comSend();
                commandReply("STOPPED\n");

            }else if (cmd.startsWith("ZERO") || cmd == "Z") {
                zero_angle = raw_angle;
                target = 0;
                clearRebound();
                haveReachedTarget = false;
                Serial.println("ZEROED");
                commandReply("ZEROED\n");

            } else if (cmd == "H" || cmd == "HOME") {
                // Home: velocity control at 5 rad/s to 0, then restore previous mode
                pre_home_mode = control_mode;
                pre_home_velocity = set_velocity;
                control_mode = MODE_VELOCITY;
                set_velocity = 5.0f;
                target = 0;
                start_err = target - angle;
                spinning = true;
                homing = true;
                chainLen = 0;
                chainIdx = 0;
                clearRebound();
                haveReachedTarget = false;
                Serial.printf("HOMING from %.1f rad\n", angle);
                commandReply("HOMING from " + String(angle, 1) + " rad\n");

            } else if (cmd.startsWith("STATUS") || cmd == "S") {
                String modeStr = (control_mode == MODE_VOLTAGE) ? "VOLT" : "VEL";
                commandReply(
                    "STATUS " +
                    String("SPINNING:") + String(spinning ? 1 : 0) +
                    " " + String("MOTOR:") + String(motor_found ? "FOUND" : "NONE") +
                    " " + String("MOTOR_ID:") + String(motor_id) +
                    " " + String("TARGET:") + String(target, 1) +
                    " " + String("ANGLE:") + String(angle, 1) +
                    " " + String("MODE:") + modeStr +
                    " " + String("VSET:") + String(set_voltage, 2) +
                    " " + String("VELSET:") + String(set_velocity, 1) +
                    " " + String("REBOUND:") + String(reboundLoaded ? (reboundWaiting ? "WAITING" : "LOADED") : "0") +
                    " " + String("RTH:") + String(reboundDropRad, 2) +
                    " " + String("VBAT:") + String(vbat, 2) +
                    " " + String("VEL:") + String(vel, 1) +
                    " " + String("UART_US:") + String(lastUartTimeUs) +
                    "\n"
                );

            } else if (cmd == "LOG") {
                uint32_t totalSamples = min(logCount, LOG_BUFFER_SIZE);
                if (totalSamples == 0) {
                    commandReply("NO LOG DATA\n");
                } else {
                    // Calculate duration from newest sample
                    uint32_t newestIdx = (logHead + LOG_BUFFER_SIZE - 1) % LOG_BUFFER_SIZE;
                    uint32_t duration_ms = logBuffer[newestIdx].time_us / 1000;

                    String info = "LOG: " + String(totalSamples) + " samples, " +
                                String(duration_ms) + "ms duration\n" +
                                "Download? (Y/N/C)\n";
                    commandReply(info);

                    awaitingLogConfirm = true;
                }

            } else if (cmd.startsWith("PRBS")) {
                // Parse: PRBS [amplitude] [bit_ms] [duration_s]
                String args = cmd.substring(4);
                args.trim();
                if (args.length() > 0) {
                    int sp1 = args.indexOf(' ');
                    if (sp1 > 0) {
                        prbs_amplitude = constrain(args.substring(0, sp1).toFloat(), 0.1f, 7.0f);
                        String rest = args.substring(sp1 + 1);
                        rest.trim();
                        int sp2 = rest.indexOf(' ');
                        if (sp2 > 0) {
                            prbs_bit_us = rest.substring(0, sp2).toInt() * 1000;
                            prbs_duration_us = rest.substring(sp2 + 1).toInt() * 1000000;
                        } else {
                            prbs_bit_us = rest.toInt() * 1000;
                        }
                    } else {
                        prbs_amplitude = constrain(args.toFloat(), 0.1f, 7.0f);
                    }
                }
                // Start PRBS
                prbs_rng = 0x12345678u;
                prbs_start_time = micros();
                prbs_next_bit_time = prbs_start_time;
                prbs_v_cmd = 0.0f;
                prbs_running = true;
                spinning = false;
                clearRebound();
                haveReachedTarget = false;
                // Start logging (reset buffer for PRBS sysid)
                logHead = 0;
                logCount = 0;
                logStartTime = prbs_start_time;
                logging = true;
                logReady = false;

                commandReply("PRBS: " + String(prbs_amplitude, 1) + "V, " +
                    String(prbs_bit_us/1000) + "ms, " + String(prbs_duration_us/1000000) + "s\n");

            } else {
                Serial.printf("Unknown cmd: '%s'\n", cmd.c_str());
                commandReply("UNKNOWN CMD\n");
            }
        }
    }
}
