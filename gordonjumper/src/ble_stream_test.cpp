#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <iq_module_communication.hpp>

constexpr int TX_PIN = 13;
constexpr int RX_PIN = 12;
constexpr int MOTOR_GND_PIN = 11;
constexpr uint32_t MOTOR_BAUD = 921600;
constexpr uint32_t MOTOR_TIMEOUT_US = 3000;
constexpr uint32_t WAVE_HALF_PERIOD_US = 500000;
constexpr uint32_t WAVE_DURATION_US = 10000000;
constexpr float WAVE_VOLTS = 1.0f;
constexpr uint16_t PACKET_MAGIC = 0xD14A;
constexpr uint8_t SAMPLES_PER_PACKET = 15;
constexpr uint8_t PACKET_QUEUE_SIZE = 32;

constexpr char BLE_NAME[] = "dnajumper-test";
constexpr char SERVICE_UUID[] = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
constexpr char WRITE_UUID[] = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";
constexpr char NOTIFY_UUID[] = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";

struct __attribute__((packed)) Sample {
    uint16_t offset_us;
    float angle;
    float velocity;
};

struct __attribute__((packed)) PacketHeader {
    uint16_t magic;
    uint16_t packet_seq;
    uint32_t first_sample_seq;
    uint32_t base_time_us;
    float battery;
    float command;
    float target;
    uint16_t uart_timeouts;
    uint16_t queue_drops;
    uint16_t notify_errors;
    uint8_t count;
    int8_t motor_id;
};

struct __attribute__((packed)) StreamPacket {
    PacketHeader header;
    Sample samples[SAMPLES_PER_PACKET];
};

static_assert(sizeof(Sample) == 10);
static_assert(sizeof(StreamPacket) == 182);

GenericInterface com;
PowerMonitorClient *power = nullptr;
MultiTurnAngleControlClient *motor = nullptr;
int motor_id = -1;

BLEServer *ble_server = nullptr;
BLECharacteristic *ble_notify = nullptr;
QueueHandle_t packet_queue = nullptr;
volatile bool ble_connected = false;
volatile bool stream_enabled = false;
volatile uint32_t notify_errors = 0;
volatile uint32_t packets_sent = 0;

uint32_t sample_seq = 0;
uint32_t packet_seq = 0;
uint32_t uart_timeouts = 0;
uint32_t queue_drops = 0;
uint32_t request_start_us = 0;
bool request_pending = false;
float battery = 0.0f;
uint32_t battery_counter = 0;
float motor_command = 0.0f;
bool control_dirty = true;
bool wave_running = false;
uint32_t wave_start_us = 0;
uint32_t wave_next_us = 0;

enum Action : uint8_t {
    ACTION_NONE,
    ACTION_WAVE,
    ACTION_OFF,
};
volatile Action pending_action = ACTION_NONE;

StreamPacket current_packet{};
uint8_t current_count = 0;

uint32_t stats_start_ms = 0;
uint32_t stats_samples = 0;
uint64_t stats_uart_total_us = 0;
uint32_t stats_uart_max_us = 0;

void sendCom(GenericInterface &iface) {
    uint8_t data[128];
    uint8_t length = 0;
    if (iface.GetTxBytes(data, length)) {
        Serial1.write(data, length);
    }
}

void readCom(GenericInterface &iface, PowerMonitorClient &read_power,
             MultiTurnAngleControlClient &read_motor) {
    uint8_t data[128];
    int available = Serial1.available();
    while (available > 0) {
        size_t length = Serial1.read(
            data, min((int)sizeof(data), available));
        if (length == 0) {
            break;
        }
        iface.SetRxBytes(data, length);

        uint8_t *packet = nullptr;
        uint8_t packet_length = 0;
        while (iface.PeekPacket(&packet, &packet_length)) {
            read_motor.ReadMsg(packet, packet_length);
            read_power.ReadMsg(packet, packet_length);
            iface.DropPacket();
        }
        available = Serial1.available();
    }
}

bool probeMotor(int id) {
    GenericInterface probe_com;
    PowerMonitorClient probe_power(id);
    MultiTurnAngleControlClient probe_motor(id);

    while (Serial1.available()) {
        Serial1.read();
    }

    for (int attempt = 0; attempt < 2; ++attempt) {
        probe_power.volts_.get(probe_com);
        probe_motor.obs_angular_displacement_.get(probe_com);
        sendCom(probe_com);

        uint32_t start = micros();
        while (micros() - start < 4000) {
            readCom(probe_com, probe_power, probe_motor);
            if (probe_power.volts_.IsFresh() ||
                probe_motor.obs_angular_displacement_.IsFresh()) {
                return true;
            }
        }
        delay(2);
    }
    return false;
}

bool findMotor() {
    for (int id = 0; id <= 16; ++id) {
        if (probeMotor(id)) {
            motor_id = id;
            power = new PowerMonitorClient(id);
            motor = new MultiTurnAngleControlClient(id);
            Serial.printf("motor=%d\n", id);
            return true;
        }
    }
    Serial.println("motor not found");
    return false;
}

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
        ble_connected = true;
        server->updateConnParams(param->connect.remote_bda, 6, 12, 0, 400);
        Serial.printf("ble connected mtu=%u\n",
                      server->getPeerMTU(server->getConnId()));
    }

    void onDisconnect(BLEServer *server) override {
        ble_connected = false;
        stream_enabled = false;
        pending_action = ACTION_OFF;
        xQueueReset(packet_queue);
        server->startAdvertising();
        Serial.println("ble disconnected");
    }

    void onMtuChanged(BLEServer *, esp_ble_gatts_cb_param_t *param) override {
        Serial.printf("ble mtu=%u\n", param->mtu.mtu);
    }
};

class NotifyCallbacks : public BLECharacteristicCallbacks {
    void onStatus(BLECharacteristic *, Status status, uint32_t) override {
        if (status != SUCCESS_NOTIFY) {
            ++notify_errors;
        }
    }
};

class WriteCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *characteristic) override {
        std::string value = characteristic->getValue();
        if (value == "START") {
            xQueueReset(packet_queue);
            notify_errors = 0;
            stream_enabled = true;
            Serial.println("stream started");
        } else if (value == "STOP") {
            stream_enabled = false;
            pending_action = ACTION_OFF;
            xQueueReset(packet_queue);
            Serial.println("stream stopped");
        } else if (value == "WAVE") {
            pending_action = ACTION_WAVE;
        } else if (value == "OFF") {
            pending_action = ACTION_OFF;
        }
    }
};

void bleSendTask(void *) {
    StreamPacket packet;
    while (true) {
        if (xQueueReceive(packet_queue, &packet, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        if (!ble_connected || !stream_enabled) {
            continue;
        }
        size_t length = sizeof(PacketHeader) +
                        packet.header.count * sizeof(Sample);
        ble_notify->setValue(reinterpret_cast<uint8_t *>(&packet), length);
        ble_notify->notify();
        ++packets_sent;
    }
}

void setupBle() {
    packet_queue = xQueueCreate(PACKET_QUEUE_SIZE, sizeof(StreamPacket));

    BLEDevice::init(BLE_NAME);
    BLEDevice::setMTU(185);
    ble_server = BLEDevice::createServer();
    ble_server->setCallbacks(new ServerCallbacks());

    BLEService *service = ble_server->createService(SERVICE_UUID);
    ble_notify = service->createCharacteristic(
        NOTIFY_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    ble_notify->addDescriptor(new BLE2902());
    ble_notify->setCallbacks(new NotifyCallbacks());

    BLECharacteristic *ble_write = service->createCharacteristic(
        WRITE_UUID,
        BLECharacteristic::PROPERTY_WRITE |
        BLECharacteristic::PROPERTY_WRITE_NR);
    ble_write->setCallbacks(new WriteCallbacks());
    service->start();

    BLEAdvertising *advertising = BLEDevice::getAdvertising();
    advertising->addServiceUUID(SERVICE_UUID);
    advertising->setScanResponse(true);
    advertising->setMinPreferred(0x06);
    advertising->setMinPreferred(0x0C);
    BLEDevice::startAdvertising();

    xTaskCreatePinnedToCore(bleSendTask, "ble-send", 4096, nullptr, 2, nullptr, 0);
}

void flushPacket() {
    if (current_count == 0) {
        return;
    }
    if (!ble_connected || !stream_enabled) {
        current_count = 0;
        return;
    }

    current_packet.header.uart_timeouts = uart_timeouts;
    current_packet.header.queue_drops = queue_drops;
    current_packet.header.count = current_count;
    current_packet.header.notify_errors = notify_errors;

    if (xQueueSend(packet_queue, &current_packet, 0) != pdTRUE) {
        ++queue_drops;
    }
    current_count = 0;
}

void setMotorCommand(float volts) {
    if (motor_command == volts) {
        return;
    }
    flushPacket();
    motor_command = volts;
    control_dirty = true;
    Serial.printf("command=%.1fV\n", motor_command);
}

void queueSample(float angle, float velocity, uint32_t now) {
    if (!ble_connected || !stream_enabled) {
        current_count = 0;
        return;
    }

    if (current_count == 0) {
        current_packet.header.magic = PACKET_MAGIC;
        current_packet.header.packet_seq = packet_seq++;
        current_packet.header.first_sample_seq = sample_seq;
        current_packet.header.base_time_us = now;
        current_packet.header.battery = battery;
        current_packet.header.command = motor_command;
        current_packet.header.target = 0.0f;
        current_packet.header.motor_id = motor_id;
    }

    current_packet.samples[current_count] = {
        static_cast<uint16_t>(now - current_packet.header.base_time_us),
        angle,
        velocity
    };
    ++current_count;

    if (current_count != SAMPLES_PER_PACKET) {
        return;
    }
    flushPacket();
}

void startMotorRequest() {
    motor->obs_angular_displacement_.get(com);
    motor->obs_angular_velocity_.get(com);
    if (control_dirty) {
        if (motor_command == 0.0f) {
            motor->ctrl_coast_.set(com);
        } else {
            motor->ctrl_volts_.set(com, motor_command);
        }
        control_dirty = false;
    }
    if (++battery_counter >= 100) {
        power->volts_.get(com);
        battery_counter = 0;
    }
    sendCom(com);
    request_start_us = micros();
    request_pending = true;
}

void finishMotorRequest() {
    uint32_t now = micros();
    uint32_t uart_us = now - request_start_us;
    float angle = motor->obs_angular_displacement_.get_reply();
    float velocity = motor->obs_angular_velocity_.get_reply();
    if (power->volts_.IsFresh()) {
        battery = power->volts_.get_reply();
    }

    queueSample(angle, velocity, now);
    ++sample_seq;
    ++stats_samples;
    stats_uart_total_us += uart_us;
    stats_uart_max_us = max(stats_uart_max_us, uart_us);
    request_pending = false;
}

void timeoutMotorRequest() {
    if (motor->obs_angular_displacement_.IsFresh()) {
        motor->obs_angular_displacement_.get_reply();
    }
    if (motor->obs_angular_velocity_.IsFresh()) {
        motor->obs_angular_velocity_.get_reply();
    }
    if (power->volts_.IsFresh()) {
        battery = power->volts_.get_reply();
    }
    ++uart_timeouts;
    request_pending = false;
}

void startWave(uint32_t now) {
    wave_running = true;
    wave_start_us = now;
    wave_next_us = now + WAVE_HALF_PERIOD_US;
    setMotorCommand(WAVE_VOLTS);
    Serial.println("wave started");
}

void stopWave() {
    wave_running = false;
    setMotorCommand(0.0f);
    Serial.println("wave stopped");
}

void updateWave(uint32_t now) {
    Action action = pending_action;
    pending_action = ACTION_NONE;

    if (action == ACTION_WAVE) {
        startWave(now);
    } else if (action == ACTION_OFF) {
        stopWave();
    }

    if (!wave_running) {
        return;
    }
    if (now - wave_start_us >= WAVE_DURATION_US) {
        stopWave();
        return;
    }
    if ((int32_t)(now - wave_next_us) >= 0) {
        wave_next_us += WAVE_HALF_PERIOD_US;
        setMotorCommand(-motor_command);
    }
}

void printStats() {
    uint32_t now = millis();
    if (now - stats_start_ms < 1000) {
        return;
    }

    uint32_t elapsed_ms = now - stats_start_ms;
    float rate = stats_samples * 1000.0f / elapsed_ms;
    uint32_t mean_uart = stats_samples
        ? stats_uart_total_us / stats_samples
        : 0;
    Serial.printf(
        "rate=%.1fHz uart_mean=%luus uart_max=%luus timeouts=%lu "
        "queue=%u drops=%lu sent=%lu notify_errors=%lu connected=%d\n",
        rate, mean_uart, stats_uart_max_us, uart_timeouts,
        uxQueueMessagesWaiting(packet_queue), queue_drops, packets_sent,
        notify_errors, ble_connected);

    stats_start_ms = now;
    stats_samples = 0;
    stats_uart_total_us = 0;
    stats_uart_max_us = 0;
}

void setup() {
    setCpuFrequencyMhz(240);
    Serial.begin(115200);
    pinMode(MOTOR_GND_PIN, OUTPUT);
    digitalWrite(MOTOR_GND_PIN, LOW);
    delay(1000);

    Serial1.begin(MOTOR_BAUD, SERIAL_8N1, RX_PIN, TX_PIN);
    delay(1000);
    if (!findMotor()) {
        while (true) {
            delay(1000);
        }
    }

    setupBle();
    stats_start_ms = millis();
    startMotorRequest();
    Serial.println("benchmark ready");
}

void loop() {
    readCom(com, *power, *motor);

    if (request_pending &&
        motor->obs_angular_displacement_.IsFresh() &&
        motor->obs_angular_velocity_.IsFresh()) {
        finishMotorRequest();
    } else if (request_pending &&
               micros() - request_start_us >= MOTOR_TIMEOUT_US) {
        timeoutMotorRequest();
    }

    if (!request_pending) {
        updateWave(micros());
        startMotorRequest();
    }
    printStats();
}
