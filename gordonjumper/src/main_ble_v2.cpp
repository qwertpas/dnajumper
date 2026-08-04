#include <Arduino.h>
#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <iq_module_communication.hpp>
#include <math.h>
#include <stdarg.h>
#include <strings.h>

constexpr int TX_PIN = 13;
constexpr int RX_PIN = 12;
constexpr int MOTOR_GND_PIN = 11;
constexpr int LED_PIN = 48;
constexpr uint32_t MOTOR_BAUD = 921600;
constexpr uint32_t MOTOR_TIMEOUT_US = 3000;
constexpr uint8_t MOTOR_TIMEOUT_LIMIT = 5;
constexpr float TARGET_LIMIT_RAD = 188.0f;
constexpr float VOLTAGE_LIMIT = 12.0f;
constexpr float VELOCITY_LIMIT = 300.0f;
constexpr float HOME_VELOCITY = 5.0f;
constexpr float TARGET_EPSILON = 0.01f;
constexpr uint8_t MAX_REBOUNDS = 8;

constexpr uint16_t PACKET_MAGIC = 0xD24B;
constexpr uint8_t SAMPLES_PER_PACKET = 12;
constexpr uint8_t TX_QUEUE_SIZE = 32;
constexpr uint8_t COMMAND_QUEUE_SIZE = 16;
constexpr uint16_t MAX_NOTIFY_SIZE = 182;

constexpr char BLE_NAME[] = "dnajumper-v2";
constexpr char SERVICE_UUID[] = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
constexpr char WRITE_UUID[] = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";
constexpr char NOTIFY_UUID[] = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";

enum ControlMode : uint8_t {
    MODE_VOLTAGE,
    MODE_VELOCITY,
};

enum OffMode : uint8_t {
    OFF_COAST,
    OFF_BRAKE,
};

enum MotionState : uint8_t {
    STATE_IDLE,
    STATE_MOVING,
    STATE_WAITING_REBOUND,
    STATE_HOMING,
};

struct Rebound {
    float threshold;
    uint32_t delay_us;
};

struct __attribute__((packed)) Sample {
    uint16_t offset_us;
    float angle;
    float velocity;
    uint16_t battery_mv;
};

struct __attribute__((packed)) PacketHeader {
    uint16_t magic;
    uint16_t packet_seq;
    uint32_t base_time_us;
    float command;
    float target;
    float rebound_trigger;
    uint16_t uart_timeouts;
    uint16_t queue_drops;
    uint8_t count;
    int8_t motor_id;
    uint8_t state;
    uint8_t mode;
};

struct __attribute__((packed)) StreamPacket {
    PacketHeader header;
    Sample samples[SAMPLES_PER_PACKET];
};

struct TxMessage {
    uint16_t length;
    uint8_t data[MAX_NOTIFY_SIZE];
};

struct Command {
    char text[96];
};

static_assert(sizeof(Sample) == 12);
static_assert(sizeof(PacketHeader) == 28);
static_assert(sizeof(StreamPacket) <= MAX_NOTIFY_SIZE);

GenericInterface com;
PowerMonitorClient *power = nullptr;
MultiTurnAngleControlClient *motor = nullptr;
int motor_id = -1;

BLEServer *ble_server = nullptr;
BLECharacteristic *ble_notify = nullptr;
QueueHandle_t tx_queue = nullptr;
QueueHandle_t command_queue = nullptr;
volatile bool ble_connected = false;
volatile bool stream_enabled = false;
volatile bool emergency_stop = false;
volatile uint16_t ble_mtu = 23;
volatile uint32_t notify_errors = 0;
volatile uint32_t command_drops = 0;
uint32_t queue_drops = 0;

float raw_angle = 0.0f;
float zero_angle = 0.0f;
float angle = 0.0f;
float velocity = 0.0f;
float battery = 0.0f;

ControlMode configured_mode = MODE_VOLTAGE;
float configured_value = 1.0f;
ControlMode active_mode = MODE_VOLTAGE;
float active_value = 1.0f;
MotionState motion_state = STATE_IDLE;
float target = 0.0f;
int direction = 0;
float reached_target = 0.0f;
int reached_direction = 0;
uint32_t reached_time_us = 0;

Rebound rebounds[MAX_REBOUNDS];
uint8_t rebound_head = 0;
uint8_t rebound_count = 0;
bool rebound_in_progress = false;

float motor_command = 0.0f;
bool motor_off = true;
OffMode off_mode = OFF_COAST;
bool control_dirty = true;

bool request_pending = false;
uint32_t request_start_us = 0;
uint8_t consecutive_timeouts = 0;
uint32_t uart_timeouts = 0;

StreamPacket current_packet{};
uint8_t current_count = 0;
uint16_t packet_seq = 0;

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
        if (!probeMotor(id)) {
            continue;
        }
        motor_id = id;
        power = new PowerMonitorClient(id);
        motor = new MultiTurnAngleControlClient(id);
        Serial.printf("motor=%d\n", id);
        return true;
    }
    Serial.println("motor not found");
    return false;
}

bool queueTx(const uint8_t *data, size_t length) {
    if (!ble_connected || length == 0 || length > MAX_NOTIFY_SIZE) {
        return false;
    }
    TxMessage message{};
    message.length = length;
    memcpy(message.data, data, length);
    if (xQueueSend(tx_queue, &message, 0) == pdTRUE) {
        return true;
    }
    ++queue_drops;
    return false;
}

void reply(const char *format, ...) {
    char text[MAX_NOTIFY_SIZE];
    va_list args;
    va_start(args, format);
    int length = vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    if (length <= 0) {
        return;
    }
    queueTx(reinterpret_cast<uint8_t *>(text),
            min((size_t)length, sizeof(text) - 1));
}

void bleSendTask(void *) {
    TxMessage message;
    while (true) {
        if (xQueueReceive(tx_queue, &message, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        if (!ble_connected) {
            continue;
        }
        ble_notify->setValue(message.data, message.length);
        ble_notify->notify();
    }
}

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer *server, esp_ble_gatts_cb_param_t *param) override {
        ble_connected = true;
        notify_errors = 0;
        ble_mtu = server->getPeerMTU(server->getConnId());
        server->updateConnParams(param->connect.remote_bda, 6, 12, 0, 400);
        Serial.println("ble connected");
    }

    void onDisconnect(BLEServer *server) override {
        ble_connected = false;
        stream_enabled = false;
        emergency_stop = true;
        ble_mtu = 23;
        xQueueReset(tx_queue);
        server->startAdvertising();
        Serial.println("ble disconnected");
    }

    void onMtuChanged(BLEServer *, esp_ble_gatts_cb_param_t *param) override {
        ble_mtu = param->mtu.mtu;
        Serial.printf("ble mtu=%u\n", ble_mtu);
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
        size_t length = min(characteristic->getLength(),
                            sizeof(Command::text) - 1);
        if (length == 0) {
            return;
        }
        Command command{};
        memcpy(command.text, characteristic->getData(), length);
        command.text[length] = '\0';
        if (xQueueSend(command_queue, &command, 0) != pdTRUE) {
            ++command_drops;
        }
    }
};

void setupBle() {
    tx_queue = xQueueCreate(TX_QUEUE_SIZE, sizeof(TxMessage));
    command_queue = xQueueCreate(COMMAND_QUEUE_SIZE, sizeof(Command));

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

    xTaskCreatePinnedToCore(bleSendTask, "ble-send", 4096,
                            nullptr, 2, nullptr, 0);
}

void flushStreamPacket() {
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
    size_t length = sizeof(PacketHeader) + current_count * sizeof(Sample);
    queueTx(reinterpret_cast<uint8_t *>(&current_packet), length);
    current_count = 0;
}

void queueSample(uint32_t now) {
    if (!ble_connected || !stream_enabled) {
        current_count = 0;
        return;
    }
    if (current_count == 0) {
        current_packet.header.magic = PACKET_MAGIC;
        current_packet.header.packet_seq = packet_seq++;
        current_packet.header.base_time_us = now;
        current_packet.header.command = motor_command;
        current_packet.header.target = NAN;
        current_packet.header.rebound_trigger = NAN;
        if (motion_state == STATE_MOVING ||
            motion_state == STATE_HOMING) {
            current_packet.header.target = target;
        } else if (motion_state == STATE_WAITING_REBOUND &&
                   rebound_count > 0) {
            current_packet.header.target = -reached_target;
            current_packet.header.rebound_trigger =
                reached_target -
                reached_direction * rebounds[rebound_head].threshold;
        }
        current_packet.header.motor_id = motor_id;
        current_packet.header.state = motion_state;
        current_packet.header.mode = active_mode;
    }

    current_packet.samples[current_count] = {
        static_cast<uint16_t>(
            now - current_packet.header.base_time_us),
        angle,
        velocity,
        static_cast<uint16_t>(constrain(
            lroundf(battery * 1000.0f), 0L, 65535L)),
    };
    if (++current_count == SAMPLES_PER_PACKET) {
        flushStreamPacket();
    }
}

void requestOff() {
    if (motor_off && !control_dirty) {
        return;
    }
    flushStreamPacket();
    motor_command = 0.0f;
    motor_off = true;
    control_dirty = true;
}

void requestDrive(ControlMode mode, float command) {
    if (!motor_off && active_mode == mode &&
        motor_command == command) {
        return;
    }
    flushStreamPacket();
    active_mode = mode;
    motor_command = command;
    motor_off = false;
    control_dirty = true;
}

void clearRebounds() {
    rebound_head = 0;
    rebound_count = 0;
}

bool addRebound(float threshold, uint32_t delay_ms) {
    if (rebound_count + (rebound_in_progress ? 1 : 0) >= MAX_REBOUNDS ||
        !isfinite(threshold) || threshold < 0.05f ||
        threshold > 50.0f || delay_ms > 10000) {
        return false;
    }
    uint8_t index = (rebound_head + rebound_count) % MAX_REBOUNDS;
    rebounds[index] = {threshold, delay_ms * 1000};
    ++rebound_count;
    return true;
}

Rebound &nextRebound() {
    return rebounds[rebound_head];
}

void popRebound() {
    rebound_head = (rebound_head + 1) % MAX_REBOUNDS;
    --rebound_count;
}

uint8_t reboundsLeft() {
    return rebound_count + (rebound_in_progress ? 1 : 0);
}

bool beginMove(float new_target, ControlMode mode, float value,
               MotionState state) {
    float error = new_target - angle;
    target = new_target;
    active_mode = mode;
    active_value = value;
    if (fabsf(error) <= TARGET_EPSILON) {
        requestOff();
        motion_state = STATE_IDLE;
        return false;
    }
    direction = error > 0.0f ? 1 : -1;
    motion_state = state;
    requestDrive(active_mode, direction * active_value);
    digitalWrite(LED_PIN, HIGH);
    return true;
}

void stopMotion(bool send_event) {
    clearRebounds();
    rebound_in_progress = false;
    motion_state = STATE_IDLE;
    direction = 0;
    target = angle;
    requestOff();
    digitalWrite(LED_PIN, LOW);
    if (send_event) {
        reply("EVENT STOPPED\n");
    }
}

void updateMotion(uint32_t now) {
    if (motion_state == STATE_MOVING ||
        motion_state == STATE_HOMING) {
        bool reached = direction * (target - angle) <= 0.0f;
        if (!reached) {
            return;
        }

        MotionState completed_state = motion_state;
        bool completed_rebound = rebound_in_progress;
        reached_target = target;
        reached_direction = direction;
        reached_time_us = now;
        requestOff();
        digitalWrite(LED_PIN, LOW);
        if (completed_rebound) {
            rebound_in_progress = false;
        }

        if (completed_state == STATE_HOMING) {
            motion_state = STATE_IDLE;
            reply("EVENT HOMED %.4f\n", angle);
        } else if (rebound_count > 0) {
            motion_state = STATE_WAITING_REBOUND;
            reply("EVENT TARGET %.4f WAIT %u\n",
                  reached_target, rebound_count);
        } else {
            motion_state = STATE_IDLE;
            reply("EVENT DONE %.4f\n", angle);
        }
        return;
    }

    if (motion_state != STATE_WAITING_REBOUND ||
        rebound_count == 0) {
        return;
    }

    Rebound rebound = nextRebound();
    if (now - reached_time_us < rebound.delay_us) {
        return;
    }
    bool reversed = reached_direction > 0
        ? angle <= reached_target - rebound.threshold
        : angle >= reached_target + rebound.threshold;
    if (!reversed) {
        return;
    }

    popRebound();
    float next_target = -reached_target;
    rebound_in_progress = true;
    if (!beginMove(next_target, active_mode, active_value, STATE_MOVING)) {
        rebound_in_progress = false;
    }
    reply("EVENT REBOUND %.4f REMAINING %u\n",
          next_target, reboundsLeft());
}

void startMotorRequest() {
    power->volts_.get(com);
    motor->obs_angular_displacement_.get(com);
    motor->obs_angular_velocity_.get(com);
    if (control_dirty) {
        if (motor_off) {
            if (off_mode == OFF_BRAKE) {
                motor->ctrl_brake_.set(com);
            } else {
                motor->ctrl_coast_.set(com);
            }
        } else if (active_mode == MODE_VOLTAGE) {
            motor->ctrl_volts_.set(com, motor_command);
        } else {
            motor->ctrl_velocity_.set(com, motor_command);
        }
        control_dirty = false;
    }
    sendCom(com);
    request_start_us = micros();
    request_pending = true;
}

void finishMotorRequest() {
    uint32_t now = micros();
    uint32_t uart_us = now - request_start_us;
    raw_angle = motor->obs_angular_displacement_.get_reply();
    angle = raw_angle - zero_angle;
    velocity = motor->obs_angular_velocity_.get_reply();
    battery = power->volts_.get_reply();

    queueSample(now);
    ++stats_samples;
    stats_uart_total_us += uart_us;
    stats_uart_max_us = max(stats_uart_max_us, uart_us);
    consecutive_timeouts = 0;
    request_pending = false;
    updateMotion(now);
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
    ++consecutive_timeouts;
    request_pending = false;
    if (consecutive_timeouts >= MOTOR_TIMEOUT_LIMIT &&
        motion_state != STATE_IDLE) {
        stopMotion(false);
        reply("ERROR MOTOR TIMEOUT\n");
    }
}

const char *modeName(ControlMode mode) {
    return mode == MODE_VOLTAGE ? "VOLTAGE" : "VELOCITY";
}

const char *offModeName() {
    return off_mode == OFF_BRAKE ? "BRAKE" : "COAST";
}

const char *stateName(MotionState state) {
    switch (state) {
        case STATE_MOVING: return "MOVING";
        case STATE_WAITING_REBOUND: return "WAITING";
        case STATE_HOMING: return "HOMING";
        default: return "IDLE";
    }
}

void processCommand(const char *text) {
    char operation[24] = {};
    if (sscanf(text, "%23s", operation) != 1) {
        return;
    }

    if (!strcasecmp(operation, "STREAM")) {
        if (ble_mtu < 185) {
            reply("ERROR MTU %u\n", ble_mtu);
            return;
        }
        stream_enabled = true;
        current_count = 0;
        reply("OK STREAM\n");
        return;
    }

    if (!strcasecmp(operation, "STOP") ||
        !strcasecmp(operation, "OFF")) {
        stopMotion(true);
        return;
    }

    if (!strcasecmp(operation, "OFF_MODE")) {
        char value[12] = {};
        if (sscanf(text, "%*s %11s", value) != 1) {
            reply("ERROR OFF_MODE\n");
            return;
        }
        if (!strcasecmp(value, "COAST")) {
            off_mode = OFF_COAST;
        } else if (!strcasecmp(value, "BRAKE")) {
            off_mode = OFF_BRAKE;
        } else {
            reply("ERROR OFF_MODE\n");
            return;
        }
        if (motion_state == STATE_IDLE) {
            control_dirty = true;
            requestOff();
        }
        reply("OK OFF_MODE %s\n", offModeName());
        return;
    }

    if (!strcasecmp(operation, "STATUS")) {
        reply(
            "OK STATUS STATE=%s MODE=%s OFF=%s SET=%.3f TARGET=%.4f "
            "ANGLE=%.4f VEL=%.3f VBAT=%.3f REBOUNDS=%u "
            "MOTOR=%d MTU=%u UART_TIMEOUTS=%lu TX_DROPS=%lu "
            "CMD_DROPS=%lu NOTIFY_ERRORS=%lu\n",
            stateName(motion_state), modeName(configured_mode), offModeName(),
            configured_value, target, angle, velocity, battery,
            reboundsLeft(), motor_id, ble_mtu, uart_timeouts,
            queue_drops, command_drops, notify_errors);
        return;
    }

    if (!strcasecmp(operation, "MODE")) {
        char mode[16] = {};
        float value = 0.0f;
        if (sscanf(text, "%*s %15s %f", mode, &value) != 2 ||
            !isfinite(value) || value <= 0.0f) {
            reply("ERROR MODE\n");
            return;
        }
        if (!strcasecmp(mode, "VOLTAGE") &&
            value <= VOLTAGE_LIMIT &&
            (battery <= 0.0f || value <= battery)) {
            configured_mode = MODE_VOLTAGE;
        } else if (!strcasecmp(mode, "VELOCITY") &&
                   value <= VELOCITY_LIMIT) {
            configured_mode = MODE_VELOCITY;
        } else {
            reply("ERROR MODE RANGE\n");
            return;
        }
        configured_value = value;
        if (motion_state == STATE_IDLE) {
            flushStreamPacket();
            active_mode = configured_mode;
            active_value = configured_value;
        }
        reply("OK MODE %s %.3f\n",
              modeName(configured_mode), configured_value);
        return;
    }

    if (!strcasecmp(operation, "MOVE")) {
        float new_target = 0.0f;
        int count = 0;
        float threshold = 2.0f;
        unsigned delay_ms = 500;
        int parsed = sscanf(text, "%*s %f %d %f %u",
                            &new_target, &count, &threshold, &delay_ms);
        if (parsed < 1 || !isfinite(new_target) ||
            fabsf(new_target) > TARGET_LIMIT_RAD ||
            count < 0 || count > MAX_REBOUNDS ||
            motion_state != STATE_IDLE) {
            reply("ERROR MOVE\n");
            return;
        }

        clearRebounds();
        for (int index = 0; index < count; ++index) {
            if (!addRebound(threshold, delay_ms)) {
                clearRebounds();
                reply("ERROR REBOUND\n");
                return;
            }
        }
        if (!beginMove(new_target, configured_mode,
                       configured_value, STATE_MOVING)) {
            clearRebounds();
            reply("EVENT DONE %.4f\n", angle);
            return;
        }
        reply("OK MOVE %.4f %s %.3f REBOUNDS=%u\n",
              target, modeName(active_mode), active_value,
              rebound_count);
        return;
    }

    if (!strcasecmp(operation, "ADD_REBOUND")) {
        float threshold = 0.0f;
        unsigned delay_ms = 0;
        if (sscanf(text, "%*s %f %u", &threshold, &delay_ms) != 2 ||
            (motion_state != STATE_MOVING &&
             motion_state != STATE_WAITING_REBOUND) ||
            !addRebound(threshold, delay_ms)) {
            reply("ERROR ADD_REBOUND\n");
            return;
        }
        reply("OK ADD_REBOUND %u\n", reboundsLeft());
        return;
    }

    if (!strcasecmp(operation, "REMOVE_REBOUND")) {
        if ((motion_state != STATE_MOVING &&
             motion_state != STATE_WAITING_REBOUND) ||
            rebound_count == 0) {
            reply("ERROR REMOVE_REBOUND\n");
            return;
        }
        --rebound_count;
        if (motion_state == STATE_WAITING_REBOUND &&
            rebound_count == 0) {
            motion_state = STATE_IDLE;
        }
        reply("OK REMOVE_REBOUND %u\n", reboundsLeft());
        return;
    }

    if (!strcasecmp(operation, "CLEAR_REBOUNDS")) {
        clearRebounds();
        if (motion_state == STATE_WAITING_REBOUND) {
            motion_state = STATE_IDLE;
        }
        reply("OK CLEAR_REBOUNDS\n");
        return;
    }

    if (!strcasecmp(operation, "HOME")) {
        if (motion_state != STATE_IDLE) {
            reply("ERROR BUSY\n");
            return;
        }
        clearRebounds();
        if (!beginMove(0.0f, MODE_VELOCITY,
                       HOME_VELOCITY, STATE_HOMING)) {
            reply("EVENT HOMED %.4f\n", angle);
            return;
        }
        reply("OK HOME %.4f\n", angle);
        return;
    }

    if (!strcasecmp(operation, "ZERO")) {
        if (motion_state != STATE_IDLE) {
            reply("ERROR BUSY\n");
            return;
        }
        zero_angle = raw_angle;
        angle = 0.0f;
        target = 0.0f;
        clearRebounds();
        requestOff();
        reply("OK ZERO\n");
        return;
    }

    reply("ERROR UNKNOWN\n");
}

void processCommands() {
    Command command;
    while (xQueueReceive(command_queue, &command, 0) == pdTRUE) {
        processCommand(command.text);
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
        ? stats_uart_total_us / stats_samples : 0;
    Serial.printf(
        "rate=%.1fHz uart_mean=%luus uart_max=%luus state=%s "
        "timeouts=%lu drops=%lu command=%.2f\n",
        rate, mean_uart, stats_uart_max_us,
        stateName(motion_state), uart_timeouts,
        queue_drops, motor_command);
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
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    delay(1000);

    Serial1.begin(MOTOR_BAUD, SERIAL_8N1, RX_PIN, TX_PIN);
    delay(1000);
    while (!findMotor()) {
        delay(500);
    }

    setupBle();
    stats_start_ms = millis();
    startMotorRequest();
    Serial.println("motor_ble_v2 ready");
}

void loop() {
    readCom(com, *power, *motor);

    if (request_pending &&
        motor->obs_angular_displacement_.IsFresh() &&
        motor->obs_angular_velocity_.IsFresh() &&
        power->volts_.IsFresh()) {
        finishMotorRequest();
    } else if (request_pending &&
               micros() - request_start_us >= MOTOR_TIMEOUT_US) {
        timeoutMotorRequest();
    }

    if (emergency_stop) {
        emergency_stop = false;
        stopMotion(false);
    }
    processCommands();

    if (!request_pending) {
        startMotorRequest();
    }
    printStats();
}
