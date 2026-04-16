
// // //Same setup with wifi
// #include <Arduino.h>
// #include <WiFi.h>
// #include <WiFiUdp.h>
// #include <iq_module_communication.hpp>

// // ---- Motor setup ----
// const int TX_PIN = 13; // ESP32 TX → Vertiq RX
// const int RX_PIN = 12; // ESP32 RX → Vertiq TX
// const int MTR_GND = 11;
// const int LED_PIN = 48;

// IqSerial ser(Serial1);
// PowerMonitorClient power(0);
// BrushlessDriveClient mot(0);
// // PropellerMotorControlClient prop(0); 
// MultiTurnAngleControlClient multi(0);

// // ---- Wi-Fi SoftAP + UDP ----
// WiFiUDP udp;
// const char *SSID = "ESP32-Motor";
// const uint16_t PORT = 9870;

// void udpReply(const String &message) {
//     udp.beginPacket(udp.remoteIP(), udp.remotePort());
//     udp.write((const uint8_t *)message.c_str(), message.length());
//     udp.endPacket();
// }


// void setup() {
//     setCpuFrequencyMhz(80);

//     Serial.begin(115200);
//     pinMode(MTR_GND, OUTPUT);
//     digitalWrite(MTR_GND, LOW);
//     pinMode(LED_PIN, OUTPUT);
//     digitalWrite(LED_PIN, LOW);
//     delay(1000);
//     Serial.println("\nStarting Vertiq Wi-Fi Control (SoftAP mode)");

//     // ---- Wi-Fi Access Point ----
//     WiFi.mode(WIFI_AP);
//     WiFi.softAP(SSID, nullptr, 6);
//     IPAddress ip = WiFi.softAPIP();
//     Serial.printf("SoftAP '%s' active\nConnect your laptop to Wi-Fi SSID: %s\n", SSID, SSID);
//     Serial.printf("ESP32 IP address: %s\n", ip.toString().c_str());

//     // ---- UDP listener ----
//     udp.begin(PORT);
//     Serial.printf("Listening for UDP commands on port %d\n", PORT);

//     // ---- Motor UART ----
//     Serial1.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);
//     ser.set(multi.ctrl_volts_, 0.0f);
//     Serial.println("Motor interface initialized");
// }

// float raw_angle = 0;
// float zero_angle = 0;
// float angle = 0;
// float target = 0;
// bool spinning = false;
// float start_err = 1;
// float set_voltage = 0.2f;

// float vbat = 0;
// float vel = 0;



// void loop() {
       
//     delay(1);

//     ser.get(power.volts_, vbat);
//     ser.get(multi.obs_angular_velocity_, vel);
//     ser.get(multi.obs_angular_displacement_, raw_angle);
//     angle = raw_angle - zero_angle;

//     float error = target - angle;
//     if(spinning && start_err*error>0){ //direction of error hasn't changed
//         if(error > 0){
//             ser.set(multi.ctrl_volts_, set_voltage);
//         }else{ 
//             ser.set(multi.ctrl_volts_, -set_voltage);
//         }
//         digitalWrite(LED_PIN, HIGH);
//     }else{
//         ser.set(multi.ctrl_volts_, 0.0f);
//         ser.set(multi.ctrl_brake_);
//         spinning = false;
//         digitalWrite(LED_PIN, LOW);
//     }


//     int packetSize = udp.parsePacket();
//     if (packetSize) {
//         char buf[64];
//         int len = udp.read(buf, sizeof(buf) - 1);
//         buf[len] = '\0';
//         String cmd(buf);
//         cmd.trim();
//         cmd.toUpperCase();

//         if (cmd.startsWith("T")) {
//             float val = cmd.substring(1).toFloat();
//             target = constrain(val, -188, 188); //30 rotations
//             start_err = target - angle;
//             spinning = true;
//             Serial.printf("TARGET SET %.1f, ERR %.1f\n", target, start_err);
//             udpReply("TARGET SET " + String(target, 1) + ", ERR" + String(start_err, 1) + "\n");

//         } else if(cmd.startsWith("V")) {
//             float val = cmd.substring(1).toFloat();
//             set_voltage = constrain(val, 0, vbat); 
//             Serial.printf("VOLTAGE SET %.2f / %.2f\n", set_voltage, vbat);
//             udpReply("VOLTAGE SET " + String(set_voltage, 2) + " / " + String(vbat, 2) + "\n");

//         } else if(cmd.startsWith("STOP") || cmd.startsWith("OFF")){
//             spinning = false;
//             ser.set(multi.ctrl_volts_, 0.0f);
//             ser.set(multi.ctrl_brake_);
//             udpReply("STOPPED\n");

//         }else if (cmd.startsWith("ZERO") || cmd == "Z") {
//             zero_angle = angle;
//             Serial.println("ZEROED");
//             udpReply("ZEROED\n");

//         } else if (cmd.startsWith("STATUS") || cmd == "S") {
            
//             udpReply(
//                 "STATUS " +
//                 String("SPINNING:") + String(spinning ? 1 : 0) +
//                 " " + String("TARGET:") + String(target, 1) +
//                 " " + String("ANGLE:") + String(angle, 1) +
//                 " " + String("VSET:") + String(set_voltage, 2) +
//                 " " + String("VBAT:") + String(vbat, 2) +
//                 " " + String("VEL:") + String(vel, 1) +
//                 "\n"
//             );

//         } else {
//             Serial.printf("Unknown cmd: '%s'\n", cmd.c_str());
//         }
//     }


// }
